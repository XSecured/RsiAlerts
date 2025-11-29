import asyncio
import json
import logging
import os
import random
import time
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Set, Optional, Tuple, Any
from itertools import cycle
import aiohttp
import numpy as np
import redis.asyncio as aioredis
import talib
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

@dataclass
class Config:
    # System
    MAX_CONCURRENCY: int = 75
    REQUEST_TIMEOUT: int = 7
    MAX_RETRIES: int = 5
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Telegram
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # URLs (Moved from code to Config)
    PROXY_URL: str = "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"
    URLS: Dict[str, str] = field(default_factory=lambda: {
        'binance_perp_info': 'https://fapi.binance.com/fapi/v1/exchangeInfo',
        'binance_spot_info': 'https://api.binance.com/api/v3/exchangeInfo',
        'binance_perp_kline': 'https://fapi.binance.com/fapi/v1/klines',
        'binance_spot_kline': 'https://api.binance.com/api/v3/klines',
        'bybit_info': 'https://api.bybit.com/v5/market/instruments-info',
        'bybit_kline': 'https://api.bybit.com/v5/market/kline',
    })

    # Trading Params
    RSI_PERIOD: int = 14
    BB_LENGTH: int = 34
    BB_STDDEV: float = 2.0
    CANDLE_LIMIT: int = 60
    MIN_CANDLES: int = 36
    
    # Thresholds
    UPPER_TOUCH_THRESHOLD: float = 0.02
    LOWER_TOUCH_THRESHOLD: float = 0.02
    MIDDLE_TOUCH_THRESHOLD: float = 0.035
    
    # Cache & Timeframes
    CACHE_TTL_MAP: Dict[str, int] = field(default_factory=lambda: {
        '4h': 14400 + 600, '1d': 86400 + 1800, '1w': 604800 + 3600
    })
    
    IGNORED_SYMBOLS: Set[str] = field(default_factory=lambda: {
        "USDPUSDT", "USD1USDT", "TUSDUSDT", "AEURUSDT", "USDCUSDT", "EURUSDT", 
        "SUSDUSDT", "BUSDUSDT", "FDUSDUSDT"
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
# 2. DATA MODELS
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
# 3. OPTIMIZED PROXY POOL
# ==========================================

class AsyncProxyPool:
    def __init__(self, max_pool_size=20):
        self.proxies: List[str] = []
        self.max_pool_size = max_pool_size
        self._lock = asyncio.Lock()
        # Structure: {proxy: {"strikes": 0, "uses": 0, "cooldown_until": 0, "score": 1.0}}
        self.health: Dict[str, Dict[str, Any]] = {} 

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

        logging.info(f"🔎 Validating {len(raw)} proxies (Timeout: 7s)...")
        self.proxies = []
        self.health = {}

        sem = asyncio.Semaphore(200)

        async def protected_test(p):
            async with sem: return await self._test_proxy(p, session)

        tasks = [asyncio.create_task(protected_test(p)) for p in raw]
        
        for future in asyncio.as_completed(tasks):
            try:
                proxy, is_good = await future
                if is_good:
                    self.proxies.append(proxy)
                    self.health[proxy] = {"strikes": 0, "uses": 0, "cooldown_until": 0, "score": 1.0}
                    if len(self.proxies) >= self.max_pool_size: break
            except Exception:
                pass

        # Cancel remaining
        for t in tasks:
            if not t.done(): t.cancel()
        
        if self.proxies:
            logging.info(f"✅ Proxy Pool Ready: {len(self.proxies)} proxies.")
        else:
            logging.error("❌ NO WORKING PROXIES FOUND! Bot may fail.")

    async def _test_proxy(self, proxy: str, session: aiohttp.ClientSession) -> Tuple[str, bool]:
        try:
            # Test against Binance Futures for real latency check
            url = CONFIG.URLS['binance_perp_kline']
            params = {"symbol": "BTCUSDT", "interval": "1m", "limit": "2"}
            # ✅ FIX #3: Reduced timeout from 7s to 4s
            async with session.get(url, params=params, proxy=proxy, timeout=4) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        return proxy, True
            return proxy, False
        except:
            return proxy, False

    async def get_proxy(self) -> Optional[str]:
        """
        Optimized selection: Randomly picks from the top 25% healthiest proxies.
        Avoids O(N) scan on every request.
        """
        if not self.proxies: return None
        
        now = time.time()
        
        # Filter available
        available = [p for p in self.proxies if self.health[p]["cooldown_until"] < now]
        if not available:
            # If all cooled down, pick the one ending soonest
            return min(self.proxies, key=lambda p: self.health[p]["cooldown_until"])

        # Quick Selection: Pick from random sample of best performers
        # Instead of sorting entire list every time (O(N log N)), we just sample.
        candidates = available if len(available) < 5 else random.sample(available, min(len(available), 10))
        
        # Small local optimization
        best = max(candidates, key=lambda p: self.health[p]['score'])
        
        self.health[best]["uses"] += 1
        return best

    async def report_failure(self, proxy: str):
        async with self._lock:
            if proxy not in self.health: return
            
            h = self.health[proxy]
            h["strikes"] += 1
            
            # Dynamic Score Update
            uses = max(h["uses"], 1)
            success_rate = 1 - (h["strikes"] / uses)
            h["score"] = success_rate
            
            # Ban Condition
            if h["uses"] > 10 and success_rate < 0.6:
                if proxy in self.proxies:
                    self.proxies.remove(proxy)
                    del self.health[proxy]
                    logging.warning(f"🚫 Banned {proxy} (Success: {success_rate:.1%})")
            else:
                # Soft Cooldown (30s to 300s depending on severity)
                penalty = 30 * h["strikes"]
                h["cooldown_until"] = time.time() + min(penalty, 300)

# ==========================================
# 4. ROBUST CACHE MANAGER
# ==========================================

class CacheManager:
    def __init__(self):
        self.redis = None

    async def init(self):
        try:
            self.redis = await aioredis.from_url(CONFIG.REDIS_URL, decode_responses=True)
            await self.redis.ping()
            logging.info("✅ Redis Connected")
        except Exception as e:
            logging.warning(f"⚠️ Redis connection failed: {e}. Caching disabled.")
            self.redis = None

    async def close(self):
        if self.redis: 
            await self.redis.aclose()

    # --- Helpers ---
    async def _get_json(self, key: str):
        if not self.redis: return None
        try:
            data = await self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logging.error(f"Cache Read Error ({key}): {e}")
            return None

    async def _set_json(self, key: str, data: Any, ttl: int = 0):
        if not self.redis: return
        try:
            payload = json.dumps(data)
            if ttl > 0:
                await self.redis.set(key, payload, ex=ttl)
            else:
                await self.redis.set(key, payload)
        except Exception as e:
            logging.error(f"Cache Write Error ({key}): {e}")

    # --- Domain Methods ---
    
    async def get_cached_symbols(self) -> Optional[Dict]:
        return await self._get_json("bb_bot:symbols_cache_v3")

    async def save_cached_symbols(self, symbols: Dict):
        payload = {"timestamp": time.time(), "data": symbols}
        await self._set_json("bb_bot:symbols_cache_v3", payload)

    def _scan_key(self, tf: str, candle_key: int) -> str:
        return f"bb_touch:scan:{tf}:{candle_key}"

    async def get_scan_results(self, tf: str, candle_key: int) -> Optional[List[Dict]]:
        return await self._get_json(self._scan_key(tf, candle_key))

    async def save_scan_results(self, tf: str, candle_key: int, results: List[Dict]):
        ttl = CONFIG.CACHE_TTL_MAP.get(tf, 3600)
        await self._set_json(self._scan_key(tf, candle_key), results, ttl=ttl)

    async def get_sent_state(self) -> Dict[str, int]:
        res = await self._get_json("bb_bot:sent_state")
        return res if res else {}

    async def save_sent_state(self, state: Dict[str, int]):
        await self._set_json("bb_bot:sent_state", state)

# ==========================================
# 5. MODULAR EXCHANGE CLIENTS (DRY)
# ==========================================

class BaseExchange:
    def __init__(self, session: aiohttp.ClientSession, proxy_pool: AsyncProxyPool, name: str):
        self.session = session
        self.proxies = proxy_pool
        self.name = name
        # Limit concurrency based on proxy availability
        limit = CONFIG.MAX_CONCURRENCY if proxy_pool.proxies else 5
        self.sem = asyncio.Semaphore(limit)

    async def _fetch_with_retry(self, url: str, params: dict = None) -> Any:
        for _ in range(CONFIG.MAX_RETRIES):
            proxy = await self.proxies.get_proxy()
            if not proxy:
                await asyncio.sleep(1)
                continue
            
            try:
                async with self.sem:
                    async with self.session.get(url, params=params, proxy=proxy, timeout=CONFIG.REQUEST_TIMEOUT) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        elif resp.status == 429:
                            logging.warning(f"⚠️ 429 Rate Limit ({self.name}). Sleeping 5s.")
                            await asyncio.sleep(5)
                        elif resp.status >= 500:
                            pass # Server error, retry
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                await self.proxies.report_failure(proxy)
            except Exception as e:
                logging.error(f"Unexpected fetch error: {e}")
            
            # Jitter sleep
            await asyncio.sleep(0.5 + random.random())
            
        return None

    async def get_symbols(self, market_type: str) -> List[str]:
        raise NotImplementedError

    async def fetch_ohlcv(self, symbol: str, interval: str, market_type: str, limit: int) -> List[float]:
        raise NotImplementedError

class BinanceClient(BaseExchange):
    def __init__(self, session, pool):
        super().__init__(session, pool, "Binance")

    async def get_symbols(self, market_type: str) -> List[str]:
        url = CONFIG.URLS['binance_perp_info'] if market_type == 'perp' else CONFIG.URLS['binance_spot_info']
        data = await self._fetch_with_retry(url)
        if not data: return []
        
        try:
            if market_type == 'perp':
                return [s['symbol'] for s in data['symbols'] if s.get('contractType') == 'PERPETUAL' and s['status'] == 'TRADING' and s.get('quoteAsset') == 'USDT']
            else:
                return [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING' and s.get('quoteAsset') == 'USDT']
        except Exception as e:
            logging.error(f"Binance Parse Error: {e}")
            return []

    async def fetch_ohlcv(self, symbol: str, interval: str, market_type: str, limit: int) -> List[float]:
        url = CONFIG.URLS['binance_perp_kline'] if market_type == 'perp' else CONFIG.URLS['binance_spot_kline']
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        data = await self._fetch_with_retry(url, params)
        
        if not data: return []
        try:
            return [float(c[4]) for c in data] # Close price
        except: 
            return []

class BybitClient(BaseExchange):
    def __init__(self, session, pool):
        super().__init__(session, pool, "Bybit")

    async def get_symbols(self, market_type: str) -> List[str]:
        url = CONFIG.URLS['bybit_info']
        cat = 'linear' if market_type == 'perp' else 'spot'
        data = await self._fetch_with_retry(url, {'category': cat})
        if not data: return []
        
        try:
            return [s['symbol'] for s in data['result']['list'] if s['status'] == 'Trading' and s['quoteCoin'] == 'USDT']
        except Exception as e:
            logging.error(f"Bybit Parse Error: {e}")
            return []

    async def fetch_ohlcv(self, symbol: str, interval: str, market_type: str, limit: int) -> List[float]:
        url = CONFIG.URLS['bybit_kline']
        cat = 'linear' if market_type == 'perp' else 'spot'
        
        # Bybit interval mapping
        b_int = {"15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240", "1d": "D", "1w": "W"}.get(interval, "D")
        if interval == '1h' and limit == 25: b_int = "60" # Volatility check specifically
        
        params = {'category': cat, 'symbol': symbol, 'interval': b_int, 'limit': limit}
        data = await self._fetch_with_retry(url, params)
        
        if not data: return []
        try:
            raw = data.get('result', {}).get('list', [])
            if not raw: return []
            closes = [float(c[4]) for c in raw]
            return closes[::-1] # Bybit returns descending order
        except:
            return []

# ==========================================
# 6. NON-BLOCKING & VECTORIZED CORE LOGIC
# ==========================================

# Global Thread Pool for CPU tasks
cpu_executor = ThreadPoolExecutor(max_workers=4)

# ✅ FIX #2: Memoized cache key
_cache_key_memo = {}  # {tf: key}
def get_cache_key(tf: str) -> int:
    """Returns stable integer timestamp for current TF candle."""
    if tf in _cache_key_memo:
        return _cache_key_memo[tf]
    
    mins = TIMEFRAME_MINUTES[tf]
    now = int(time.time())
    key = now - (now % (mins * 60))
    _cache_key_memo[tf] = key
    return key

def calculate_volatility_vectorized(closes: List[float]) -> float:
    """Vectorized Volatility Calc (Suggestion #4)"""
    if len(closes) < 24: return 0.0
    try:
        prices = np.array(closes, dtype=np.float64)
        # Ignore division errors for robustness
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_changes = np.diff(prices) / prices[:-1] * 100
        
        pct_changes = pct_changes[np.isfinite(pct_changes)]
        if len(pct_changes) == 0: return 0.0
        return float(np.std(pct_changes))
    except Exception:
        return 0.0

def check_bb_rsi_sync(closes: List[float], tf: str) -> Tuple[Optional[str], Optional[str], float]:
    """Synchronous CPU bound logic to be run in executor"""
    try:
        if len(closes) < CONFIG.MIN_CANDLES: return None, None, 0.0
        
        np_c = np.array(closes, dtype=float)
        rsi = talib.RSI(np_c, timeperiod=CONFIG.RSI_PERIOD)
        upper, mid, lower = talib.BBANDS(rsi, timeperiod=CONFIG.BB_LENGTH, 
                                         nbdevup=CONFIG.BB_STDDEV, nbdevdn=CONFIG.BB_STDDEV, matype=0)
        
        idx = -2
        if np.isnan(rsi[idx]) or np.isnan(upper[idx]): return None, None, 0.0
        
        curr_rsi = rsi[idx]
        
        # Check Bounds
        if curr_rsi >= upper[idx] * (1 - CONFIG.UPPER_TOUCH_THRESHOLD): 
            return "UPPER", None, curr_rsi
        if curr_rsi <= lower[idx] * (1 + CONFIG.LOWER_TOUCH_THRESHOLD): 
            return "LOWER", None, curr_rsi
            
        # Check Middle
        if tf in MIDDLE_BAND_TFS:
            if abs(curr_rsi - mid[idx]) <= (mid[idx] * CONFIG.MIDDLE_TOUCH_THRESHOLD):
                prev_diff = rsi[idx-1] - mid[idx-1]
                curr_diff = curr_rsi - mid[idx]
                direction = "from above" if (prev_diff > 0 >= curr_diff) or (curr_diff > 0) else "from below"
                return "MIDDLE", direction, curr_rsi
                
        return None, None, 0.0
    except Exception as e:
        return None, None, 0.0

# ==========================================
# 7. REFACTORED MAIN BOT (LIFECYCLE METHODS)
# ==========================================

class RsiBot:
    def __init__(self):
        self.cache = CacheManager()
        self.proxies = AsyncProxyPool()
        self.session: Optional[aiohttp.ClientSession] = None
        self.binance: Optional[BinanceClient] = None
        self.bybit: Optional[BybitClient] = None
        
        # ✅ FIX #4: Add lock for all_pairs mutation
        self._pairs_lock = asyncio.Lock()
        
        # State
        self.all_pairs = [] # List of tuples (client, symbol, market, exchange_name)
        self.hot_coins = set()
        self.sent_state = {}

    async def initialize(self):
        """Step 1: Bootup"""
        global _cache_key_memo  # Reset memoization for each run
        _cache_key_memo = {}
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        await self.cache.init()
        
        # Reuse session (Suggestion #6 in previous context, standard here)
        self.session = aiohttp.ClientSession()
        await self.proxies.populate(CONFIG.PROXY_URL, self.session)
        
        self.binance = BinanceClient(self.session, self.proxies)
        self.bybit = BybitClient(self.session, self.proxies)

    async def shutdown(self):
        if self.session: await self.session.close()
        await self.cache.close()
        logging.info("👋 Bot Shutdown Complete")

    async def sync_market_data(self):
        """Step 2: Hybrid Symbol Fetch"""
        logging.info("🔄 Syncing Market Pairs...")
        
        async def fetch_safe(func, mkt, name, cache_key):
            cached = await self.cache.get_cached_symbols()
            cached_val = cached['data'].get(cache_key, []) if cached else []
            
            # Try fresh fetch
            try:
                res = await func(mkt)
                if res: 
                    logging.info(f"✅ {name}: {len(res)} symbols")
                    return res
            except Exception as e:
                logging.error(f"❌ {name} fetch error: {e}")
            
            # Fallback
            if cached_val:
                logging.warning(f"⚠️ Using Cached for {name}")
                return cached_val
            return []

        bp, bs, yp, ys = await asyncio.gather(
            fetch_safe(self.binance.get_symbols, 'perp', 'Binance Perp', 'bp'),
            fetch_safe(self.binance.get_symbols, 'spot', 'Binance Spot', 'bs'),
            fetch_safe(self.bybit.get_symbols, 'perp', 'Bybit Perp', 'yp'),
            fetch_safe(self.bybit.get_symbols, 'spot', 'Bybit Spot', 'ys')
        )

        # Save successful fetch
        if any([bp, bs, yp, ys]):
            await self.cache.save_cached_symbols({'bp': bp, 'bs': bs, 'yp': yp, 'ys': ys})

        # Compile and Filter
        seen = set()
        
        # ✅ FIX #4: Use lock when mutating shared state
        async with self._pairs_lock:
            self.all_pairs = []  # Reset and rebuild
            
            def add_pairs(client, syms, mkt, ex_name):
                for s in syms:
                    if s in CONFIG.IGNORED_SYMBOLS or not s.endswith("USDT"): continue
                    norm = s.upper().replace("USDT", "")
                    if norm not in seen:
                        self.all_pairs.append((client, s, mkt, ex_name))
                        seen.add(norm)

            add_pairs(self.binance, bp, 'perp', 'Binance')
            add_pairs(self.binance, bs, 'spot', 'Binance')
            add_pairs(self.bybit, yp, 'perp', 'Bybit')
            add_pairs(self.bybit, ys, 'spot', 'Bybit')
            
            total_symbols = len(self.all_pairs)
        
        logging.info(f"📊 Total Unique Symbols: {total_symbols}")

    async def analyze_volatility(self):
        """Step 3: Vectorized Volatility Analysis"""
        logging.info("🔥 Calculating Market Volatility...")
        vol_scores = {}

        async def process_vol(client, sym, mkt):
            # Only need 25 candles for vol check
            closes = await client.fetch_ohlcv(sym, '1h', mkt, 25)
            if not closes: return
            
            v = calculate_volatility_vectorized(closes)
            if v > 0: vol_scores[sym] = v

        # Batch execution
        tasks = [process_vol(c, s, m, e) for c, s, m, e in self.all_pairs]
        
        # Chunking to avoid memory explosion
        chunk_size = 200
        for i in range(0, len(tasks), chunk_size):
            await asyncio.gather(*tasks[i:i+chunk_size], return_exceptions=True)
            
        self.hot_coins = set(sorted(vol_scores, key=vol_scores.get, reverse=True)[:60])
        logging.info(f"🔥 Hot Coins Identified: {len(self.hot_coins)}")

    async def scan_targets(self) -> List[TouchHit]:
        """Step 4: Scan Logic with Early Exits"""
        self.sent_state = await self.cache.get_sent_state()
        
        # Determine what to scan
        tfs_to_scan = []
        cached_hits = []
        
        for tf in ACTIVE_TFS:
            # Check if already sent
            candle_key = get_cache_key(tf)
            if tf in CACHED_TFS:
                if str(self.sent_state.get(tf)) == str(candle_key):
                    logging.info(f"⏭️ Skipping {tf}: Already alerted.")
                    continue
                    
                # Check if we have results cached
                res = await self.cache.get_scan_results(tf, candle_key)
                if res:
                    logging.info(f"📦 Loaded {tf} from Cache.")
                    cached_hits.extend([TouchHit.from_dict(r) for r in res])
                    continue
            
            tfs_to_scan.append(tf)

        if not tfs_to_scan:
            logging.info("💤 Nothing new to scan.")
            return cached_hits

        logging.info(f"🕵️ Scanning Fresh: {tfs_to_scan}")
        
        fresh_hits = []
        loop = asyncio.get_running_loop()

        # ✅ FIX #4: Use lock when reading shared state
        async with self._pairs_lock:
            pairs_snapshot = self.all_pairs.copy()

        # Scan Loop
        for tf in tfs_to_scan:
            tf_hits = []
            
            async def scan_single(client, sym, mkt, ex_name):
                closes = await client.fetch_ohlcv(sym, tf, mkt, CONFIG.CANDLE_LIMIT)
                if not closes: return None
                
                # Offload CPU heavy talib to thread pool
                t_type, direction, rsi_val = await loop.run_in_executor(
                    cpu_executor, check_bb_rsi_sync, closes, tf
                )
                
                if t_type:
                    return TouchHit(sym, ex_name, mkt, tf, rsi_val, t_type, direction, sym in self.hot_coins)
                return None

            tasks = [scan_single(c, s, m, e) for c, s, m, e in pairs_snapshot]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for r in results:
                if isinstance(r, TouchHit):
                    tf_hits.append(r)
                elif isinstance(r, Exception):
                    # logging.error(f"Scan Error: {r}") # Optional: noisy
                    pass

            fresh_hits.extend(tf_hits)
            
            # Save results for High TFs
            if tf in CACHED_TFS:
                candle_key = get_cache_key(tf)
                await self.cache.save_scan_results(tf, candle_key, [h.to_dict() for h in tf_hits])

        return cached_hits + fresh_hits

    async def dispatch_alerts(self, hits: List[TouchHit]):
        """Step 5: Alerting"""
        if not hits: return
        
        # Filter hits that need sending
        to_send = []
        updated_state = self.sent_state.copy()
        
        # Group by TF to check against state
        by_tf = {}
        for h in hits: by_tf.setdefault(h.timeframe, []).append(h)
        
        for tf, items in by_tf.items():
            if tf in CACHED_TFS:
                candle_key = get_cache_key(tf)
                # Only send if not sent yet
                if str(updated_state.get(tf)) != str(candle_key):
                    to_send.extend(items)
                    updated_state[tf] = candle_key
            else:
                # Low TFs always send
                to_send.extend(items)
        
        if not to_send: return

        await self._send_telegram(to_send)
        await self.cache.save_sent_state(updated_state)

    async def _send_telegram(self, hits: List[TouchHit]):
        # Grouping Logic
        grouped = {}
        for h in hits: grouped.setdefault(h.timeframe, {}).setdefault(h.touch_type, []).append(h)
        
        blocks = []
        headers = {"UPPER": "⬆️ UPPER BB", "MIDDLE": "🔶 MIDDLE BB", "LOWER": "⬇️ LOWER BB"}
        
        for tf in ["1w", "1d", "4h", "2h", "1h", "30m", "15m"]:
            if tf not in grouped: continue
            
            total = sum(len(grouped[tf].get(t, [])) for t in headers)
            lines = [f" ▣ TIMEFRAME: {tf} ({total} Hits)", "─────────────────────────", ""]
            
            for t_type, label in headers.items():
                items = grouped[tf].get(t_type, [])
                if not items: continue
                items.sort(key=lambda x: x.symbol)
                
                lines.append(f"┌ {label}")
                for idx, item in enumerate(items):
                    prefix = "└" if idx == len(items)-1 else "│"
                    icon = "🥐" if item.exchange == "Binance" else "💣"
                    sym = item.symbol.replace("USDT", "")
                    ext = f" {'🔻' if item.direction=='from above' else '🔹'}" if t_type=="MIDDLE" else ""
                    if item.hot: ext += " 🔥"
                    
                    lines.append(f"{prefix} {icon} *{sym}* ➜ *{item.rsi:.2f}*{ext}")
                lines.append("")
            
            lines.append("─────────────────────────")
            blocks.append("\n".join(lines))

        # Send Chunks
        full_text = []
        curr_len = 0
        ts = datetime.now(timezone.utc).strftime('%d %b %H:%M UTC')
        
        async def flush(txt):
            if not txt: return
            body = "\n\n".join(txt) + f"\n{ts}"
            try:
                url = f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage"
                await self.session.post(url, json={
                    "chat_id": CONFIG.CHAT_ID, 
                    "text": body, 
                    "parse_mode": "Markdown"
                })
                await asyncio.sleep(0.5)
            except Exception as e:
                logging.error(f"Telegram Fail: {e}")

        for b in blocks:
            if curr_len + len(b) > 3800:
                await flush(full_text)
                full_text = []
                curr_len = 0
            full_text.append(b)
            curr_len += len(b)
            
        await flush(full_text)

    async def run(self):
        """Main Lifecycle"""
        try:
            await self.initialize()
            await self.sync_market_data()
            await self.analyze_volatility()
            hits = await self.scan_targets()
            await self.dispatch_alerts(hits)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.exception("CRITICAL CRASH IN MAIN LOOP")
        finally:
            await self.shutdown()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    bot = RsiBot()
    asyncio.run(bot.run())
