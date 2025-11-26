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
    MAX_CONCURRENCY: int = 200
    REQUEST_TIMEOUT: int = 5
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
CACHED_TFS = {'4h', '1d', '1w'}

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
    source: str # 'Fresh' or 'Cache'
    total_symbols: int = 0
    successful_scans: int = 0
    hits_found: int = 0
    details: str = ""

# ==========================================
# PROXY POOL
# ==========================================

class AsyncProxyPool:
    def __init__(self, max_pool_size=25):
        self.proxies: List[str] = []
        self.max_pool_size = max_pool_size
        self.iterator = None
        self._lock = asyncio.Lock()
        self.failures: Dict[str, int] = {} # Track strikes

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
        self.failures = {} # Reset failures on repopulate
        random.shuffle(raw)
        
        sem = asyncio.Semaphore(250)
        async def protected_test(p):
            async with sem: return await self._test_proxy(p, session)

        tasks = [asyncio.create_task(protected_test(p)) for p in raw]
        for future in asyncio.as_completed(tasks):
            try:
                proxy, is_good = await future
                if is_good:
                    self.proxies.append(proxy)
                    if len(self.proxies) >= self.max_pool_size: break
            except: pass
        
        for t in tasks:
            if not t.done(): t.cancel()
        await asyncio.sleep(0.1)
        
        if self.proxies:
            self.iterator = cycle(self.proxies)
            logging.info(f"✅ Proxy Pool Ready: {len(self.proxies)} proxies.")
        else:
            logging.error("❌ NO WORKING PROXIES FOUND!")

    async def _test_proxy(self, proxy: str, session: aiohttp.ClientSession) -> Tuple[str, bool]:
        try:
            async with session.get("https://api.binance.com/api/v3/time", proxy=proxy, timeout=5) as resp:
                return proxy, resp.status == 200
        except: return proxy, False

    async def get_proxy(self) -> Optional[str]:
        if not self.proxies: return None
        async with self._lock: 
            if not self.proxies: return None # Double check inside lock
            return next(self.iterator)

    async def report_failure(self, proxy: str):
        """Report a proxy failure. 3 strikes = Banned."""
        async with self._lock:
            self.failures[proxy] = self.failures.get(proxy, 0) + 1
            strikes = self.failures[proxy]
            
            if strikes >= 3:
                if proxy in self.proxies:
                    self.proxies.remove(proxy)
                    # Rebuild cycle logic
                    if self.proxies:
                        self.iterator = cycle(self.proxies)
                    logging.warning(f"🚫 Banned Proxy {proxy} (3 strikes)")

# ==========================================
# REDIS CACHE MANAGER
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
        if self.redis: await self.redis.close()

    def _scan_key(self, tf: str, iso_time: str) -> str:
        return f"bb_touch:scan:{tf}:{iso_time}"

    async def get_scan_results(self, tf: str, iso_time: str) -> Optional[List[Dict]]:
        if not self.redis: return None
        try:
            data = await self.redis.get(self._scan_key(tf, iso_time))
            return json.loads(data) if data else None
        except: return None

    async def save_scan_results(self, tf: str, iso_time: str, results: List[Dict]):
        if not self.redis: return
        ttl = CONFIG.CACHE_TTL_MAP.get(tf, 3600)
        try:
            await self.redis.set(self._scan_key(tf, iso_time), json.dumps(results), ex=ttl)
        except: pass

    async def get_sent_state(self) -> Dict[str, str]:
        if not self.redis: return {}
        try:
            val = await self.redis.get("bb_bot:sent_state")
            return json.loads(val) if val else {}
        except: return {}

    async def save_sent_state(self, state: Dict[str, str]):
        if not self.redis: return
        try:
            await self.redis.set("bb_bot:sent_state", json.dumps(state))
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
        """Robust request with proxy rotation & blacklisting."""
        for attempt in range(CONFIG.MAX_RETRIES):
            proxy = await self.proxies.get_proxy()
            if not proxy:
                # No proxies left? Wait a bit and retry (maybe repopulate happens externally?)
                await asyncio.sleep(1)
                continue

            try:
                async with self.sem:
                    async with self.session.get(url, params=params, proxy=proxy, timeout=CONFIG.REQUEST_TIMEOUT) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        elif resp.status == 429:
                            logging.warning(f"⚠️ 429 Rate Limit ({proxy}). Sleeping 5s.")
                            await asyncio.sleep(5)
                        elif resp.status >= 500:
                            # Server Error, try next proxy immediately
                            pass
                        else:
                            pass # 4xx errors
            
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                # Network error = Strike for proxy
                await self.proxies.report_failure(proxy)
                # Only log if it's a "clean" error message to reduce noise
                logging.warning(f"❌ Failed {url} after {CONFIG.MAX_RETRIES} tries. Last err: {last_error}")
            
            # Jittered backoff
            await asyncio.sleep(0.5 + random.random() * 0.5)
            
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
# CORE LOGIC
# ==========================================

def get_candle_open_iso(tf: str) -> str:
    now = datetime.now(timezone.utc)
    mins = TIMEFRAME_MINUTES.get(tf, 60)
    total = now.hour * 60 + now.minute
    floored = (total // mins) * mins
    open_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=floored)
    return open_time.isoformat()

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
# MAIN BOT
# ==========================================

class RsiBot:
    def __init__(self):
        self.cache = CacheManager()
        self.proxies = AsyncProxyPool()
        
    async def send_report(self, hits: List[TouchHit]):
        if not hits: return
        grouped = {}
        for h in hits: grouped.setdefault(h.timeframe, {}).setdefault(h.touch_type, []).append(h)
        
        messages = []
        # Sort timeframes logically
        tf_order = ["1w", "1d", "4h", "2h", "1h", "30m", "15m"]
        
        for tf in tf_order:
            if tf not in grouped: continue
            
            # Header
            lines = [
                f"▣ TIMEFRAME: {tf}",
                "────────────────",
                ""
            ]
            
            headers = {"UPPER": "⬆️ UPPER BB", "MIDDLE": "🔶 MIDDLE BB", "LOWER": "⬇️ LOWER BB"}
            types_found = [t for t in ["UPPER", "MIDDLE", "LOWER"] if grouped[tf].get(t)]
            
            for i, t_type in enumerate(types_found):
                items = grouped[tf].get(t_type, [])
                items.sort(key=lambda x: x.symbol)
                
                lines.append(f"┌ {headers[t_type]}")
                
                for idx, item in enumerate(items):
                    is_last = (idx == len(items) - 1)
                    prefix = "└" if is_last else "│"
                    
                    # Exchange Icon
                    icon = "🟡" if item.exchange == "Binance" else "⚫"
                    sym = item.symbol 
                    
                    # Value
                    val = f"{item.rsi:.2f}"
                    
                    # Extras
                    extras = ""
                    if t_type == "MIDDLE":
                        arrow = "🔻" if item.direction == "from above" else "🔹"
                        extras = f" ({arrow})"
                    
                    if item.hot:
                        extras += " 🔥"
                    
                    # Line: │ 🟡 BTCUSDT | 75.42 🔥
                    lines.append(f"{prefix} {icon} {sym} | {val}{extras}")
                
                lines.append("") # Spacer
                
            lines.append("────────────────")
            
            # Footer Date: 26 Nov 18:45 UTC
            ts = datetime.utcnow().strftime('%d %b %H:%M UTC')
            lines.append(ts)
            
            messages.append("\n".join(lines))
            
        async with aiohttp.ClientSession() as session:
            for msg in messages:
                for chunk in [msg[i:i+4000] for i in range(0, len(msg), 4000)]:
                    try:
                        await session.post(f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage", 
                                         json={"chat_id": CONFIG.CHAT_ID, "text": chunk, "parse_mode": "Markdown"})
                        await asyncio.sleep(0.5)
                    except: pass

    async def fetch_all_symbols_robust(self, binance: BinanceClient, bybit: BybitClient):
        """Critical function to fetch symbols with retry logic."""
        results = {"bp": [], "bs": [], "yp": [], "ys": []}
        
        for attempt in range(3): # Try up to 3 times to get a sane list
            logging.info(f"Fetching Symbols (Attempt {attempt+1})...")
            
            # Fetch in parallel
            t_bp = asyncio.create_task(binance.get_perp_symbols())
            t_bs = asyncio.create_task(binance.get_spot_symbols())
            t_yp = asyncio.create_task(bybit.get_perp_symbols())
            t_ys = asyncio.create_task(bybit.get_spot_symbols())
            
            bp, bs, yp, ys = await asyncio.gather(t_bp, t_bs, t_yp, t_ys)
            
            results = {"bp": bp, "bs": bs, "yp": yp, "ys": ys}
            
            counts = [len(x) for x in [bp, bs, yp, ys]]
            log_msg = f"BinPerp: {counts[0]} | BinSpot: {counts[1]} | BybPerp: {counts[2]} | BybSpot: {counts[3]}"
            
            if all(c > 0 for c in counts):
                logging.info(f"✅ Symbols fetched successfully: {log_msg}")
                return bp, bs, yp, ys
            else:
                logging.warning(f"⚠️ Partial symbol fetch failure: {log_msg}. Retrying...")
                await asyncio.sleep(2)
        
        logging.error("❌ Critical: Failed to fetch all symbol lists after retries. Proceeding with partial data.")
        return results["bp"], results["bs"], results["yp"], results["ys"]

    async def run(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        await self.cache.init()
        
        async with aiohttp.ClientSession() as session:
            await self.proxies.populate(CONFIG.PROXY_URL, session)
            binance = BinanceClient(session, self.proxies)
            bybit = BybitClient(session, self.proxies)
            
            # ROBUST FETCH
            bp, bs, yp, ys = await self.fetch_all_symbols_robust(binance, bybit)
            
            # Filter & Deduplicate
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
            logging.info(f"Total symbols after deduplication: {total_sym_count}")
            
            # Volatility Check
            logging.info("Calculating Volatility...")
            vol_scores = {}
            async def check_vol(client, sym, mkt):
                c = await client.fetch_closes_volatility(sym, mkt)
                v = calculate_volatility(c)
                if v > 0: vol_scores[sym] = v
            
            vol_tasks = [check_vol(client, s, mkt) for client, s, mkt, ex in all_pairs]
            # Batch volatility
            for i in range(0, len(vol_tasks), 200): 
                await asyncio.gather(*vol_tasks[i:i+200])
            
            hot_coins = set(sorted(vol_scores, key=vol_scores.get, reverse=True)[:60])
            logging.info(f"Volatility Calc: {len(vol_scores)}/{len(all_pairs)} success | Hot Coins: {len(hot_coins)}")
            
            # Scan Setup
            sent_state = await self.cache.get_sent_state()
            tfs_to_scan_fresh = set()
            tfs_fresh_map = {}
            cached_hits_to_use = []
            scan_stats = {tf: ScanStats(tf, "Unknown", total_symbols=total_sym_count) for tf in ACTIVE_TFS}
            
            for tf in ACTIVE_TFS:
                if tf in CACHED_TFS:
                    iso = get_candle_open_iso(tf)
                    cached_res = await self.cache.get_scan_results(tf, iso)
                    if cached_res is None:
                        tfs_to_scan_fresh.add(tf)
                        tfs_fresh_map[tf] = iso
                        scan_stats[tf].source = "Fresh Scan (New Candle)"
                    else:
                        hits = [TouchHit.from_dict(d) for d in cached_res]
                        cached_hits_to_use.extend(hits)
                        scan_stats[tf].source = "Cached"
                        scan_stats[tf].hits_found = len(hits)
                        scan_stats[tf].successful_scans = 0 
                else:
                    tfs_to_scan_fresh.add(tf)
                    scan_stats[tf].source = "Fresh Scan (Low TF)"

            # Fresh Scans
            final_hits = []
            if tfs_to_scan_fresh:
                logging.info(f"Scanning fresh: {tfs_to_scan_fresh}")
                scan_tasks = []
                async def scan_one(client, sym, mkt, ex, tfs):
                    hits = []
                    scanned_tfs = []
                    for tf in tfs:
                        closes = await client.fetch_closes(sym, tf, mkt)
                        if closes: 
                            scanned_tfs.append(tf)
                            t_type, direction, rsi_val = check_bb_rsi(closes, tf)
                            if t_type:
                                hits.append(TouchHit(sym, ex, mkt, tf, rsi_val, t_type, direction, sym in hot_coins))
                    return hits, scanned_tfs, ex

                for client, sym, mkt, ex in all_pairs:
                    scan_tasks.append(scan_one(client, sym, mkt, ex, list(tfs_to_scan_fresh)))
                
                for f in asyncio.as_completed(scan_tasks):
                    res_hits, res_tfs, res_ex = await f
                    final_hits.extend(res_hits)
                    for tf in res_tfs: scan_stats[tf].successful_scans += 1
                    for h in res_hits: scan_stats[h.timeframe].hits_found += 1

                for tf, iso in tfs_fresh_map.items():
                    tf_hits = [h for h in final_hits if h.timeframe == tf]
                    await self.cache.save_scan_results(tf, iso, [h.to_dict() for h in tf_hits])

            # Merge Cached
            for h in cached_hits_to_use:
                h.hot = (h.symbol in hot_coins)
                final_hits.append(h)

            # Summary Log
            logging.info("="*60)
            logging.info(f"{'TF':<5} | {'Source':<25} | {'Success':<15} | {'Hits'}")
            logging.info("-" * 60)
            for tf in sorted(ACTIVE_TFS, key=lambda x: ["15m","30m","1h","2h","4h","1d","1w"].index(x)):
                st = scan_stats[tf]
                succ_str = "Skipped (Cached)" if st.source == "Cached" else f"{st.successful_scans}/{st.total_symbols}"
                logging.info(f"[{tf:<3}] {st.source:<25} | {succ_str:<15} | {st.hits_found}")
            logging.info("="*60)

            # Send
            hits_to_send = []
            new_state = sent_state.copy()
            for h in final_hits:
                tf = h.timeframe
                if tf in CACHED_TFS:
                    iso = get_candle_open_iso(tf)
                    if sent_state.get(tf, "") != iso:
                        hits_to_send.append(h)
                        new_state[tf] = iso
                else:
                    hits_to_send.append(h)
            
            await self.send_report(hits_to_send)
            await self.cache.save_sent_state(new_state)
            
        await self.cache.close()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    bot = RsiBot()
    asyncio.run(bot.run())
