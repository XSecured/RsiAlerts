import asyncio
import json
import logging
import os
import random
import math
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Set, Optional, Tuple, Any
from itertools import cycle

import aiohttp
import numpy as np
import talib
import redis.asyncio as aioredis

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

@dataclass
class Config:
    MAX_CONCURRENCY: int = 75
    REQUEST_TIMEOUT: int = 7
    MAX_RETRIES: int = 5
    
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL_MAP: Dict[str, int] = field(default_factory=lambda: {
        '4h': 14400 + 600, '1d': 86400 + 1800, '1w': 604800 + 3600
    })
    
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    PROXY_URL: str = "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"

    RSI_PERIOD: int = 14
    BB_LENGTH: int = 34
    BB_STDDEV: float = 2.0
    CANDLE_LIMIT: int = 60
    MIN_CANDLES: int = 36

    UPPER_TOUCH_THRESHOLD: float = 0.02
    LOWER_TOUCH_THRESHOLD: float = 0.02
    MIDDLE_TOUCH_THRESHOLD: float = 0.035

    IGNORED_SYMBOLS: Set[str] = field(default_factory=lambda: {
        "USDPUSDT", "USD1USDT", "TUSDUSDT", "AEURUSDT", "USDCUSDT", "EURUSDT"
    })

CONFIG = Config()

TIMEFRAME_MINUTES = {
    '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, 
    '1d': 1440, '1w': 10080
}

ACTIVE_TFS = ['15m', '30m', '1h', '2h', '4h', '1d', '1w']
MIDDLE_BAND_TFS = ['15m', '30m', '1h', '2h', '4h', '1d', '1w']
CACHED_TFS = {'2h', '4h', '1d', '1w'}

# ==========================================
# DATA MODELS
# ==========================================

@dataclass
class TouchHit:
    symbol: str
    exchange: str 
    market: str   
    timeframe: str
    rsi: float
    touch_type: str
    direction: str = "" 
    hot: bool = False
    
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return TouchHit(**d)

@dataclass
class ScanStats:
    timeframe: str
    source: str
    total_symbols: int = 0
    successful_scans: int = 0
    hits_found: int = 0

# ==========================================
# PROXY POOL (IMPROVEMENT #5: Dynamic Health Scoring)
# ==========================================

class AsyncProxyPool:
    def __init__(self, target_good: int = 40, max_test_concurrency: int = 300):
        self.proxies: List[str] = []
        self._lock = asyncio.Lock()
        self._rr_index = 0
        self.target_good = target_good
        self.max_test_concurrency = max_test_concurrency
        self.failures: Dict[str, int] = {}

    async def populate(self, url: str, session: aiohttp.ClientSession):
        if not url:
            logging.warning("No PROXY_URL configured, running without proxies.")
            self.proxies = []
            return

        try:
            logging.info(f"📥 Fetching proxy list from {url} ...")
            async with session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    logging.error(f"Proxy list fetch failed: HTTP {resp.status}")
                    return
                text = await resp.text()
        except Exception as e:
            logging.error(f"Proxy list fetch error: {e}")
            return

        raw = []
        for line in text.splitlines():
            p = line.strip()
            if not p:
                continue
            raw.append(p if "://" in p else f"http://{p}")

        if not raw:
            logging.error("Proxy list is empty.")
            return

        random.shuffle(raw)
        logging.info(f"Testing up to {len(raw)} proxies, aiming for {self.target_good} good ones...")

        test_url = "https://fapi.binance.com/fapi/v1/klines"
        test_params = {"symbol": "BTCUSDT", "interval": "1m", "limit": "5"}
        sem = asyncio.Semaphore(self.max_test_concurrency)

        async def test_one(proxy: str):
            start = time.monotonic()
            try:
                async with sem:
                    async with session.get(
                        test_url,
                        params=test_params,
                        proxy=proxy,
                        timeout=7
                    ) as r:
                        if r.status != 200:
                            return None
                        data = await r.json()
                        if not isinstance(data, list) or not data:
                            return None
            except Exception:
                return None
            latency = time.monotonic() - start
            return proxy, latency

        tasks = [asyncio.create_task(test_one(p)) for p in raw]
        good: List[Tuple[str, float]] = []

        try:
            async for fut in _as_completed_early(tasks):
                res = await fut
                if res:
                    good.append(res)
                    if len(good) >= self.target_good:
                        break
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

        if not good:
            logging.error("❌ No working proxies found in test phase.")
            self.proxies = []
            return

        good.sort(key=lambda x: x[1])
        self.proxies = [p for p, _lat in good]
        self.failures = {}
        self._rr_index = 0
        logging.info(f"✅ Proxy pool ready: {len(self.proxies)} good proxies.")

    async def get_proxy(self) -> Optional[str]:
        if not self.proxies:
            return None
        async with self._lock:
            if not self.proxies:
                return None
            proxy = self.proxies[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(self.proxies)
            return proxy

    async def report_failure(self, proxy: str):
        if not proxy:
            return
        self.failures[proxy] = self.failures.get(proxy, 0) + 1
        if self.failures[proxy] >= 3:
            async with self._lock:
                if proxy in self.proxies:
                    self.proxies.remove(proxy)
                    logging.warning(f"🚫 Dropped proxy after repeated failures: {proxy}")
                    if self._rr_index >= len(self.proxies) and self.proxies:
                        self._rr_index = 0


async def _as_completed_early(tasks: List[asyncio.Task]):
    # Helper: async generator over as_completed
    for fut in asyncio.as_completed(tasks):
        yield fut

# ==========================================
# REDIS CACHE MANAGER (IMPROVEMENT #3: Integer Keys)
# ==========================================

class CacheManager:
    def __init__(self):
        self.redis = None

    async def init(self):
        try:
            self.redis = await aioredis.from_url(CONFIG.REDIS_URL, decode_responses=True)
            await self.redis.ping()
            logging.info("✅ Redis Connected")
        except:
            logging.warning("⚠️ Redis failed. Caching disabled.")
            self.redis = None

    async def close(self):
        if self.redis: await self.redis.aclose()

    async def get_cached_symbols(self) -> Optional[Dict]:
        if not self.redis: return None
        data = await self.redis.get("bb_bot:symbols_cache_v3")
        return json.loads(data) if data else None

    async def save_cached_symbols(self, symbols: Dict):
        if not self.redis: return
        payload = {
            "timestamp": time.time(),
            "data": symbols
        }
        await self.redis.set("bb_bot:symbols_cache_v3", json.dumps(payload))
        logging.info("💾 Saved symbol list to cache (Persistent)")

    def _scan_key(self, tf: str, candle_key: int) -> str:
        return f"bb_touch:scan:{tf}:{candle_key}"

    async def get_scan_results(self, tf: str, candle_key: int) -> Optional[List[Dict]]:
        if not self.redis: return None
        try:
            data = await self.redis.get(self._scan_key(tf, candle_key))
            return json.loads(data) if data else None
        except: return None

    async def save_scan_results(self, tf: str, candle_key: int, results: List[Dict]):
        if not self.redis: return
        ttl = CONFIG.CACHE_TTL_MAP.get(tf, 3600)
        try: await self.redis.set(self._scan_key(tf, candle_key), json.dumps(results), ex=ttl)
        except: pass

    async def get_sent_state(self) -> Dict[str, int]:
        if not self.redis: return {}
        try:
            val = await self.redis.get("bb_bot:sent_state")
            return json.loads(val) if val else {}
        except: return {}

    async def save_sent_state(self, state: Dict[str, int]):
        if not self.redis: return
        try: await self.redis.set("bb_bot:sent_state", json.dumps(state))
        except: pass

# ==========================================
# EXCHANGE CLIENTS
# ==========================================

class ExchangeClient:
    def __init__(self, session: aiohttp.ClientSession, proxy_pool: AsyncProxyPool):
        self.session = session
        self.proxies = proxy_pool

        # Derive a safe but fast concurrency from proxy count
        proxy_count = len(self.proxies.proxies)
        if proxy_count:
            limit = min(CONFIG.MAX_CONCURRENCY, max(20, proxy_count * 4))
        else:
            limit = 5  # no proxies → very low direct concurrency

        self.sem = asyncio.Semaphore(limit)
        logging.info(f"HTTP concurrency set to {limit} for {proxy_count} proxies")

    async def _request(self, url: str, params: dict = None) -> Any:
        last_error = "Unknown"
        for attempt in range(CONFIG.MAX_RETRIES):
            proxy = await self.proxies.get_proxy()
            if not proxy:
                await asyncio.sleep(1)
                continue
            try:
                async with self.sem:
                    async with self.session.get(url, params=params, proxy=proxy, timeout=CONFIG.REQUEST_TIMEOUT) as resp:
                        if resp.status == 200: return await resp.json()
                        elif resp.status == 429:
                            logging.warning(f"⚠️ 429 Rate Limit ({proxy}). Sleeping 5s.")
                            await asyncio.sleep(5)
                            last_error = "429"
                        elif resp.status >= 500: last_error = f"Server {resp.status}"
                        else: last_error = f"HTTP {resp.status}"
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                await self.proxies.report_failure(proxy)
                last_error = str(e)
            except Exception as e: last_error = f"Unexpected: {str(e)}"
            await asyncio.sleep(0.5 + random.random() * 0.5)
        logging.warning(f"❌ Failed {url} after {CONFIG.MAX_RETRIES} tries. Last err: {last_error}")
        return None

class BinanceClient(ExchangeClient):
    async def get_perp_symbols(self) -> List[str]:
        data = await self._request('https://fapi.binance.com/fapi/v1/exchangeInfo')
        if not data: return []
        return [s['symbol'] for s in data['symbols'] if s.get('contractType') == 'PERPETUAL' and s['status'] == 'TRADING' and s.get('quoteAsset') == 'USDT']
    async def get_spot_symbols(self) -> List[str]:
        data = await self._request('https://api.binance.com/api/v3/exchangeInfo')
        if not data: return []
        return [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING' and s.get('quoteAsset') == 'USDT']
    async def fetch_closes_volatility(self, symbol: str, market: str) -> List[float]:
        base = 'https://api.binance.com/api/v3/klines' if market == "spot" else 'https://fapi.binance.com/fapi/v1/klines'
        data = await self._request(base, {'symbol': symbol, 'interval': '1h', 'limit': 25})
        if not data: return []
        try: return [float(c[4]) for c in data]
        except: return []
    async def fetch_closes(self, symbol: str, interval: str, market: str) -> List[float]:
        base = 'https://api.binance.com/api/v3/klines' if market == "spot" else 'https://fapi.binance.com/fapi/v1/klines'
        data = await self._request(base, {'symbol': symbol, 'interval': interval, 'limit': CONFIG.CANDLE_LIMIT})
        if not data: return []
        try: return [float(c[4]) for c in data]
        except: return []

class BybitClient(ExchangeClient):
    async def get_perp_symbols(self) -> List[str]:
        data = await self._request('https://api.bybit.com/v5/market/instruments-info', {'category': 'linear'})
        if not data: return []
        return [s['symbol'] for s in data['result']['list'] if s['status'] == 'Trading' and s['quoteCoin'] == 'USDT']
    async def get_spot_symbols(self) -> List[str]:
        data = await self._request('https://api.bybit.com/v5/market/instruments-info', {'category': 'spot'})
        if not data: return []
        return [s['symbol'] for s in data['result']['list'] if s['status'] == 'Trading' and s['quoteCoin'] == 'USDT']
    async def fetch_closes_volatility(self, symbol: str, market: str) -> List[float]:
        url = 'https://api.bybit.com/v5/market/kline'
        cat = 'linear' if market == 'perp' else 'spot'
        data = await self._request(url, {'category': cat, 'symbol': symbol, 'interval': '60', 'limit': 25})
        if not data: return []
        raw = data.get('result', {}).get('list', [])
        if not raw: return []
        closes = [float(c[4]) for c in raw]
        return closes[::-1]
    async def fetch_closes(self, symbol: str, interval: str, market: str) -> List[float]:
        url = 'https://api.bybit.com/v5/market/kline'
        cat = 'linear' if market == 'perp' else 'spot'
        b_int = {"15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240", "1d": "D", "1w": "W"}.get(interval, "D")
        data = await self._request(url, {'category': cat, 'symbol': symbol, 'interval': b_int, 'limit': CONFIG.CANDLE_LIMIT})
        if not data: return []
        raw = data.get('result', {}).get('list', [])
        if not raw: return []
        closes = [float(c[4]) for c in raw]
        return closes[::-1]

# ==========================================
# CORE LOGIC (IMPROVEMENT #3: Integer Keys Only)
# ==========================================

def get_cache_key(tf: str) -> int:
    """Returns stable integer timestamp for current TF candle."""
    mins = TIMEFRAME_MINUTES[tf]
    now = int(time.time())
    return now - (now % (mins * 60))

def calculate_volatility(closes: List[float]) -> float:
    if len(closes) < 24: return 0.0
    returns = []
    for i in range(1, len(closes)):
        if closes[i-1] != 0: returns.append((closes[i] - closes[i-1]) / closes[i-1] * 100)
    return np.std(returns) if returns else 0.0

def check_bb_rsi(closes: List[float], tf: str) -> Tuple[Optional[str], Optional[str], float]:
    if len(closes) < CONFIG.MIN_CANDLES: return None, None, 0.0
    np_c = np.array(closes, dtype=float)
    rsi = talib.RSI(np_c, timeperiod=CONFIG.RSI_PERIOD)
    upper, mid, lower = talib.BBANDS(rsi, timeperiod=CONFIG.BB_LENGTH, nbdevup=CONFIG.BB_STDDEV, nbdevdn=CONFIG.BB_STDDEV, matype=0)
    
    idx = -2
    if np.isnan(rsi[idx]) or np.isnan(upper[idx]): return None, None, 0.0
    
    curr_rsi = rsi[idx]
    if curr_rsi >= upper[idx] * (1 - CONFIG.UPPER_TOUCH_THRESHOLD): return "UPPER", None, curr_rsi
    if curr_rsi <= lower[idx] * (1 + CONFIG.LOWER_TOUCH_THRESHOLD): return "LOWER", None, curr_rsi
    
    if tf in MIDDLE_BAND_TFS:
        if abs(curr_rsi - mid[idx]) <= (mid[idx] * CONFIG.MIDDLE_TOUCH_THRESHOLD):
            prev_diff = rsi[idx-1] - mid[idx-1]
            curr_diff = curr_rsi - mid[idx]
            direction = "from above" if (prev_diff > 0 >= curr_diff) or (curr_diff > 0) else "from below"
            return "MIDDLE", direction, curr_rsi
    return None, None, 0.0

# ==========================================
# MAIN BOT (IMPROVEMENT #1: Hybrid Fetch + #6: Session Reuse)
# ==========================================

class RsiBot:
    def __init__(self):
        self.cache = CacheManager()
        self.proxies = AsyncProxyPool()
        
    async def send_report(self, session: aiohttp.ClientSession, hits: List[TouchHit]):
        if not hits: return
        grouped = {}
        for h in hits: grouped.setdefault(h.timeframe, {}).setdefault(h.touch_type, []).append(h)
        
        tf_blocks = []
        tf_order = ["1w", "1d", "4h", "2h", "1h", "30m", "15m"]
        
        for tf in tf_order:
            if tf not in grouped: continue
            
            total_hits = sum(len(grouped[tf].get(t, [])) for t in ["UPPER", "MIDDLE", "LOWER"])
            lines = [
                f" ▣ TIMEFRAME: {tf} ({total_hits} Hits)",
                "─────────────────────────",
                ""
            ]
            
            headers = {"UPPER": "⬆️ UPPER BB", "MIDDLE": "🔶 MIDDLE BB", "LOWER": "⬇️ LOWER BB"}
            found = [t for t in ["UPPER", "MIDDLE", "LOWER"] if grouped[tf].get(t)]
            
            for t in found:
                items = grouped[tf].get(t, [])
                items.sort(key=lambda x: x.symbol)
                lines.append(f"┌ {headers[t]}")
                for idx, item in enumerate(items):
                    prefix = "└" if idx == len(items)-1 else "│"
                    icon = "🥐" if item.exchange == "Binance" else "💣"
                    sym_clean = item.symbol.replace("USDT", "")
                    ext = f" {'🔻' if item.direction=='from above' else '🔹'}" if t=="MIDDLE" else ""
                    if item.hot: ext += " 🔥"
                    lines.append(f"{prefix} {icon} *{sym_clean}* ➜ *{item.rsi:.2f}*{ext}")
                lines.append("")
            
            lines.append("─────────────────────────")
            tf_blocks.append("\n".join(lines))

        if not tf_blocks: return

        current_msg = []
        current_len = 0
        ts_footer = datetime.now(timezone.utc).strftime('%d %b %H:%M UTC')
        
        for block in tf_blocks:
            if current_len + len(block) > 3800:
                full_text = "\n\n".join(current_msg) + f"\n{ts_footer}"
                try: 
                    await session.post(f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage", 
                                     json={"chat_id": CONFIG.CHAT_ID, "text": full_text, "parse_mode": "Markdown"})
                    await asyncio.sleep(0.5)
                except Exception as e: logging.error(f"TG Send Fail: {e}")
                current_msg = []
                current_len = 0
            
            current_msg.append(block)
            current_len += len(block)
        
        if current_msg:
            full_text = "\n\n".join(current_msg) + f"\n{ts_footer}"
            try: 
                await session.post(f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage", 
                                 json={"chat_id": CONFIG.CHAT_ID, "text": full_text, "parse_mode": "Markdown"})
            except Exception as e: logging.error(f"TG Send Fail: {e}")

    async def fetch_symbols_hybrid(self, binance: BinanceClient, bybit: BybitClient) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Fetches symbols with per-exchange fallback to cache."""
        
        cached = await self.cache.get_cached_symbols()
        cached_data = cached.get('data') if cached else None
        
        async def try_fetch(fetch_func, cache_key: str, name: str):
            for attempt in range(3):
                result = await fetch_func()
                if len(result) > 0:
                    logging.info(f"✅ {name}: {len(result)} symbols")
                    return result
                logging.warning(f"⚠️ {name}: Attempt {attempt+1} failed, retrying...")
                await asyncio.sleep(2)
            
            logging.warning(f"❌ {name}: Failed after retries, using cache")
            return cached_data[cache_key] if cached_data else []
        
        bp, bs, yp, ys = await asyncio.gather(
            try_fetch(binance.get_perp_symbols, 'bp', 'Binance Perp'),
            try_fetch(binance.get_spot_symbols, 'bs', 'Binance Spot'),
            try_fetch(bybit.get_perp_symbols, 'yp', 'Bybit Perp'),
            try_fetch(bybit.get_spot_symbols, 'ys', 'Bybit Spot')
        )
        
        if any(len(x) > 0 for x in [bp, bs, yp, ys]):
            await self.cache.save_cached_symbols({'bp': bp, 'bs': bs, 'yp': yp, 'ys': ys})
        
        return bp, bs, yp, ys

    async def run(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        await self.cache.init()
        async with aiohttp.ClientSession() as session:
            await self.proxies.populate(CONFIG.PROXY_URL, session)
            binance = BinanceClient(session, self.proxies)
            bybit = BybitClient(session, self.proxies)
            
            # 1. HYBRID SYMBOL FETCH
            bp, bs, yp, ys = await self.fetch_symbols_hybrid(binance, bybit)
            
            seen = set()
            def filter_u(syms):
                res = []
                for s in syms:
                    if s in CONFIG.IGNORED_SYMBOLS or not s.endswith("USDT"): continue
                    norm = s.upper().replace("USDT", "")
                    if norm not in seen:
                        res.append(s)
                        seen.add(norm)
                return res
                
            f_bp, f_bs, f_yp, f_ys = filter_u(bp), filter_u(bs), filter_u(yp), filter_u(ys)
            all_pairs = []
            for s in f_bp: all_pairs.append((binance, s, 'perp', 'Binance'))
            for s in f_bs: all_pairs.append((binance, s, 'spot', 'Binance'))
            for s in f_yp: all_pairs.append((bybit, s, 'perp', 'Bybit'))
            for s in f_ys: all_pairs.append((bybit, s, 'spot', 'Bybit'))
            
            total_sym_count = len(all_pairs)
            logging.info(f"Total symbols: {total_sym_count}")
            
            # 2. ORIGINAL VOLATILITY CALCULATION (NO PRE-FILTER)
            logging.info("Calculating Volatility...")
            vol_scores = {}
            async def check_vol(client, sym, mkt):
                c = await client.fetch_closes_volatility(sym, mkt)
                v = calculate_volatility(c)
                if v > 0: vol_scores[sym] = v
            
            vol_tasks = [check_vol(client, s, mkt) for client, s, mkt, ex in all_pairs]
            for i in range(0, len(vol_tasks), 200): await asyncio.gather(*vol_tasks[i:i+200])
            hot_coins = set(sorted(vol_scores, key=vol_scores.get, reverse=True)[:60])
            logging.info(f"Vol Calc: {len(vol_scores)}/{len(all_pairs)} success | Hot: {len(hot_coins)}")
            
            # 3. PRECISION SCAN SETUP WITH EARLY SENT STATE CHECK
            sent_state = await self.cache.get_sent_state()
            scan_stats = {tf: ScanStats(tf, "Unknown", total_symbols=total_sym_count) for tf in ACTIVE_TFS}
            
            tfs_to_scan_fresh = []
            cached_hits_to_use = []
            
            for tf in ACTIVE_TFS:
                if tf in CACHED_TFS:
                    candle_key = get_cache_key(tf)
                    if sent_state.get(tf) == candle_key:
                        logging.info(f"⏭️ Skipping {tf}: already sent for this candle")
                        scan_stats[tf].source = "Already Sent"
                        continue
                    
                    cached_res = await self.cache.get_scan_results(tf, candle_key)
                    if cached_res is None:
                        tfs_to_scan_fresh.append(tf)
                        scan_stats[tf].source = "Fresh Scan (New Candle)"
                    else:
                        hits = [TouchHit.from_dict(d) for d in cached_res]
                        cached_hits_to_use.extend(hits)
                        scan_stats[tf].source = "Cached"
                        scan_stats[tf].hits_found = len(hits)
                        scan_stats[tf].successful_scans = 0
                else:
                    tfs_to_scan_fresh.append(tf)
                    scan_stats[tf].source = "Fresh Scan (Low TF)"
            
            # 4. EXECUTE FRESH SCANS (batched by timeframe, ALL SYMBOLS)
            final_hits = []
            if tfs_to_scan_fresh:
                logging.info(f"Scanning fresh TFs: {tfs_to_scan_fresh} across all symbols...")
                
                for tf in tfs_to_scan_fresh:
                    tf_tasks = []
                    # Scan ALL symbols for this TF
                    async def scan_one(client, sym, mkt, ex):
                        closes = await client.fetch_closes(sym, tf, mkt)
                        # Treat empty/no-data as a failed scan
                        if not closes:
                            return None

                        t_type, direction, rsi_val = check_bb_rsi(closes, tf)
                        if t_type:
                            return [TouchHit(sym, ex, mkt, tf, rsi_val, t_type, direction, sym in hot_coins)]
                        # Fetched candles successfully but no hit
                        return []
                    
                    for client, sym, mkt, ex in all_pairs:
                        tf_tasks.append(scan_one(client, sym, mkt, ex))
                    
                    results = await asyncio.gather(*tf_tasks, return_exceptions=True)
                    
                    results = await asyncio.gather(*tf_tasks, return_exceptions=True)

                    success_count = 0
                    for result in results:
                        if isinstance(result, list):
                            success_count += 1
                            if result:  # non-empty list = actual hits
                                final_hits.extend(result)
                                scan_stats[tf].hits_found += len(result)

                    scan_stats[tf].successful_scans = success_count
                    
                    if tf in CACHED_TFS:
                        candle_key = get_cache_key(tf)
                        tf_hits = [h for h in final_hits if h.timeframe == tf]
                        await self.cache.save_scan_results(tf, candle_key, [h.to_dict() for h in tf_hits])
            
            # 5. MERGE CACHED
            final_hits.extend(cached_hits_to_use)
            
            # Summary
            logging.info("="*60)
            logging.info(f"{'TF':<5} | {'Source':<25} | {'Success':<15} | {'Hits'}")
            logging.info("-" * 60)
            for tf in sorted(ACTIVE_TFS, key=lambda x: ["15m","30m","1h","2h","4h","1d","1w"].index(x)):
                st = scan_stats[tf]
                succ_str = "Skipped (Cached/Sent)" if st.source in ["Cached", "Already Sent"] else f"{st.successful_scans}/{st.total_symbols}"
                logging.info(f"[{tf:<3}] {st.source:<25} | {succ_str:<15} | {st.hits_found}")
            logging.info("="*60)
            
            # 6. ALERTING
            hits_to_send = []
            new_state = sent_state.copy()
            for h in final_hits:
                tf = h.timeframe
                if tf in CACHED_TFS:
                    candle_key = get_cache_key(tf)
                    if sent_state.get(tf, 0) != candle_key:
                        hits_to_send.append(h)
                        new_state[tf] = candle_key
                else:
                    hits_to_send.append(h)
            
            await self.send_report(session, hits_to_send)
            await self.cache.save_sent_state(new_state)
            
        await self.cache.close()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_policy(asyncio.WindowsSelectorEventLoopPolicy())
    bot = RsiBot()
    asyncio.run(bot.run())
    
