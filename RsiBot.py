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
    MAX_CONCURRENCY: int = 50
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
    def __init__(self, max_pool_size=20):
        self.proxies: List[str] = []
        self.max_pool_size = max_pool_size
        self._lock = asyncio.Lock()
        self.health: Dict[str, Dict[str, Any]] = {}  # proxy -> stats

    async def populate(self, url: str, session: aiohttp.ClientSession):
        if not url: return
        raw = []
        try:
            logging.info(f"📥 Fetching proxies from {url}...")
            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    for line in text.splitlines():
                        p = line.strip()
                        if p: raw.append(p if "://" in p else f"http://{p}")
        except Exception as e:
            logging.error(f"❌ Proxy fetch failed: {e}")
            return

        logging.info(f"🔎 Validating {len(raw)} proxies...")
        self.proxies = []
        self.health = {}
        
        sem = asyncio.Semaphore(100)
        async def protected_test(p):
            async with sem: return await self._test_proxy(p, session)

        tasks = [asyncio.create_task(protected_test(p)) for p in raw]
        for future in asyncio.as_completed(tasks):
            try:
                proxy, is_good = await future
                if is_good:
                    self.proxies.append(proxy)
                    self.health[proxy] = {"strikes": 0, "uses": 0, "cooldown_until": 0}
                    if len(self.proxies) >= self.max_pool_size: break
            except: pass
        
        for t in tasks:
            if not t.done(): t.cancel()
        await asyncio.sleep(0.1)
        
        if self.proxies:
            logging.info(f"✅ Proxy Pool Ready: {len(self.proxies)} proxies.")
        else:
            logging.error("❌ NO WORKING PROXIES FOUND!")

    async def _test_proxy(self, proxy: str, session: aiohttp.ClientSession) -> Tuple[str, bool]:
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {"symbol": "BTCUSDT", "interval": "1m", "limit": "2"}
            async with session.get(url, params=params, proxy=proxy, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        return proxy, True
                return proxy, False
        except:
            return proxy, False

    async def get_proxy(self) -> Optional[str]:
        if not self.proxies: return None
        
        async with self._lock:
            now = time.time()
            available = [p for p in self.proxies if self.health.get(p, {}).get("cooldown_until", 0) < now]
            
            if not available:
                proxy = min(self.proxies, key=lambda p: self.health[p]["cooldown_until"])
                return proxy
            
            def health_score(p):
                h = self.health[p]
                uses = max(h["uses"], 1)
                return h["strikes"] / uses
            
            proxy = min(available, key=health_score)
            self.health[proxy]["uses"] += 1
            return proxy

    async def report_failure(self, proxy: str):
        async with self._lock:
            h = self.health.setdefault(proxy, {"strikes": 0, "uses": 0, "cooldown_until": 0})
            h["strikes"] += 1
            
            success_rate = 1 - (h["strikes"] / max(h["uses"], 1))
            
            if h["uses"] > 10 and success_rate < 0.6:
                if proxy in self.proxies:
                    self.proxies.remove(proxy)
                    del self.health[proxy]
                    logging.warning(f"🚫 Banned {proxy} ({success_rate:.1%} success)")
            else:
                h["cooldown_until"] = time.time() + 300

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
        limit = CONFIG.MAX_CONCURRENCY if proxy_pool.proxies else 5
        self.sem = asyncio.Semaphore(limit)

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

        tf_order = ["1w", "1d", "4h", "2h", "1h", "30m", "15m"]
        ts_footer = datetime.now(timezone.utc).strftime('%d %b %H:%M UTC')

        # Helper to shorten names
        def clean_name(s):
            s = s.replace("USDT", "")
            s = re.sub(r"^(10+|100+|1000+|1M)", "", s) 
            return s

        for tf in tf_order:
            if tf not in grouped: continue
            
            total_hits = sum(len(grouped[tf].get(t, [])) for t in ["UPPER", "MIDDLE", "LOWER"])
            if total_hits == 0: continue

            tf_lines = []
            # We pad the title with a special space to FORCE the bubble to be wide
            # This prevents the "narrowing" effect on Android
            header_pad = " " * 15 
            tf_lines.append(f"⏱ *{tf} Timeframe* ({total_hits}){header_pad}")

            targets = ["UPPER", "MIDDLE", "LOWER"]
            for t in targets:
                items = grouped[tf].get(t, [])
                if not items: continue
                
                items.sort(key=lambda x: x.rsi, reverse=(t != "LOWER"))
                
                if t == "UPPER":    header = "\n🔼 *UPPER BAND*"
                elif t == "MIDDLE": header = "\n💠 *MIDDLE BAND*"
                else:               header = "\n🔽 *LOWER BAND*"
                tf_lines.append(header)

                # --- Smart Grid Logic ---
                current_line = []
                current_char_count = 0
                MAX_CHARS = 26 # Slightly tighter to ensure 3 columns fit without wrap

                for item in items:
                    sym = clean_name(item.symbol)
                    fire = "🔥" if item.hot else ""
                    
                    # Direction arrows for Middle BB
                    dir_arrow = ""
                    if t == "MIDDLE":
                        dir_arrow = "↓" if item.direction == "from above" else ""

                    # Construct the cell: "BTC 70.1"
                    # Using a hyphen instead of bullet
                    cell = f"{sym} `{item.rsi:.1f}`{dir_arrow}{fire}"
                    cell_len = len(sym) + 6 
                    
                    if current_char_count + cell_len > MAX_CHARS:
                        # Separator is now "  |  " for cleaner look than hyphen
                        tf_lines.append("  |  ".join(current_line))
                        current_line = []
                        current_char_count = 0
                    
                    current_line.append(cell)
                    current_char_count += cell_len + 5 # +5 for "  |  "

                if current_line:
                    tf_lines.append("  |  ".join(current_line))

            tf_lines.append("")

            # --- Send Logic ---
            current_msg = []
            current_len = 0

            async def flush():
                nonlocal current_msg, current_len
                if not current_msg: return
                text = "\n".join(current_msg) + f"\n\n{ts_footer}"
                try:
                    await session.post(f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage",
                                     json={"chat_id": CONFIG.CHAT_ID, "text": text, "parse_mode": "Markdown"})
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logging.error(f"TG Send Fail: {e}")
                current_msg = []
                current_len = 0

            for line in tf_lines:
                if current_len + len(line) + 10 > 3800:
                    await flush()
                current_msg.append(line)
                current_len += len(line) + 1
            
            await flush()

    async def fetch_symbols_hybrid(self, binance: BinanceClient, bybit: BybitClient) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Fetches symbols with per-exchange fallback to cache."""
        
        cached = await self.cache.get_cached_symbols()
        cached_data = cached.get('data') if cached else None
        
        async def try_fetch(fetch_func, cache_key: str, name: str):
            # Only try ONCE per exchange. 
            # The internal _request already retries 5 times. That is enough.
            result = await fetch_func()
            
            if result and len(result) > 0:
                logging.info(f"✅ {name}: {len(result)} symbols")
                return result
            
            # If live fetch fails, immediately check cache
            if cached_data and cache_key in cached_data:
                logging.warning(f"⚠️ {name}: Live fetch failed. Using CACHE ({len(cached_data[cache_key])} syms)")
                return cached_data[cache_key]
            
            logging.error(f"❌ {name}: Failed and NO CACHE available.")
            return []
        
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
            await asyncio.gather(*vol_tasks)
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
                        if not closes: return None
                        t_type, direction, rsi_val = check_bb_rsi(closes, tf)
                        if t_type:
                            return [TouchHit(sym, ex, mkt, tf, rsi_val, t_type, direction, sym in hot_coins)]
                        return []
                    
                    for client, sym, mkt, ex in all_pairs:
                        tf_tasks.append(scan_one(client, sym, mkt, ex))
                    
                    results = await asyncio.gather(*tf_tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, list):
                            final_hits.extend(result)
                            scan_stats[tf].hits_found += len(result)
                    
                    scan_stats[tf].successful_scans = len([r for r in results if r is not None and not isinstance(r, Exception)])
                    
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
