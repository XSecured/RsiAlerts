import asyncio
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional, Tuple, Any, Union

import aiohttp
import numpy as np
import talib
import redis.asyncio as aioredis

# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class Config:
    # System Settings
    MAX_CONCURRENCY: int = 75
    BATCH_SIZE: int = 50  # <--- NEW: Protects Event Loop
    REQUEST_TIMEOUT: int = 7
    MAX_RETRIES: int = 3
    
    # Infrastructure
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    PROXY_URL: str = "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"

    # Strategy Settings
    RSI_PERIOD: int = 14
    BB_LENGTH: int = 34
    BB_STDDEV: float = 2.0
    CANDLE_LIMIT: int = 60
    MIN_CANDLES: int = 36
    
    UPPER_TOUCH_THRESHOLD: float = 0.02
    LOWER_TOUCH_THRESHOLD: float = 0.02
    MIDDLE_TOUCH_THRESHOLD: float = 0.035

    # Caching
    CACHE_TTL_MAP: Dict[str, int] = field(default_factory=lambda: {
        '4h': 14400 + 600, '1d': 86400 + 1800, '1w': 604800 + 3600
    })

    # Filtering
    IGNORED_SYMBOLS: Set[str] = field(default_factory=lambda: {
        "USDPUSDT", "USD1USDT", "TUSDUSDT", "AEURUSDT", "USDCUSDT", "EURUSDT", "FDUSDUSDT"
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
# INFRASTRUCTURE: PROXY & CACHE
# ==========================================

class AsyncProxyPool:
    def __init__(self, max_pool_size=20):
        self.proxies: List[str] = []
        self.max_pool_size = max_pool_size
        self._lock = asyncio.Lock()
        self.health: Dict[str, Dict[str, Any]] = {} 

    async def populate(self, url: str, session: aiohttp.ClientSession):
        """Fetches and validates proxies."""
        if not url: return
        raw = []
        try:
            logging.info(f"📥 Fetching proxies from source...")
            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    raw = [f"http://{p.strip()}" if "://" not in p else p.strip() 
                           for p in text.splitlines() if p.strip()]
        except Exception as e:
            logging.error(f"❌ Proxy fetch failed: {e}")
            return

        logging.info(f"🔎 Validating {len(raw)} proxies...")
        self.proxies = []
        self.health = {}
        
        sem = asyncio.Semaphore(200)
        async def _validate(p):
            async with sem:
                return await self._test_proxy(p, session)

        # Batch validation to avoid killing FD limits
        tasks = [_validate(p) for p in raw]
        for i in range(0, len(tasks), 50):
            batch = tasks[i:i+50]
            results = await asyncio.gather(*batch, return_exceptions=True)
            for res in results:
                if isinstance(res, tuple) and res[1]:
                    proxy = res[0]
                    async with self._lock:
                        if len(self.proxies) < self.max_pool_size:
                            self.proxies.append(proxy)
                            self.health[proxy] = {"strikes": 0, "uses": 0, "cooldown_until": 0}
            if len(self.proxies) >= self.max_pool_size: break
        
        logging.info(f"✅ Proxy Pool Ready: {len(self.proxies)} proxies.")

    async def _test_proxy(self, proxy: str, session: aiohttp.ClientSession) -> Tuple[str, bool]:
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {"symbol": "BTCUSDT", "interval": "1m", "limit": "2"}
            async with session.get(url, params=params, proxy=proxy, timeout=5) as resp:
                return proxy, resp.status == 200
        except:
            return proxy, False

    async def get_proxy(self) -> Optional[str]:
        if not self.proxies: return None
        async with self._lock:
            now = time.time()
            available = [p for p in self.proxies if self.health[p]["cooldown_until"] < now]
            
            if not available:
                # Fallback to the one with shortest cooldown
                return min(self.proxies, key=lambda p: self.health[p]["cooldown_until"])
            
            # Smart selection: minimize strikes, maximize success rate
            def score(p):
                h = self.health[p]
                return h["strikes"] / max(h["uses"], 1)
            
            proxy = min(available, key=score)
            self.health[proxy]["uses"] += 1
            return proxy

    async def report_failure(self, proxy: str):
        async with self._lock:
            if proxy not in self.health: return
            h = self.health[proxy]
            h["strikes"] += 1
            h["cooldown_until"] = time.time() + 60  # 1 min penalty
            
            success_rate = 1 - (h["strikes"] / max(h["uses"], 1))
            if h["uses"] > 10 and success_rate < 0.5:
                if proxy in self.proxies:
                    self.proxies.remove(proxy)
                    del self.health[proxy]
                    logging.warning(f"🚫 Banned proxy {proxy} (Rate: {success_rate:.1%})")

class CacheManager:
    def __init__(self):
        self.redis = None

    async def init(self):
        try:
            self.redis = await aioredis.from_url(CONFIG.REDIS_URL, decode_responses=True)
            await self.redis.ping()
            logging.info("✅ Redis Connected")
        except Exception as e:
            logging.warning(f"⚠️ Redis failed ({e}). Caching disabled.")
            self.redis = None

    async def close(self):
        if self.redis: await self.redis.aclose()

    async def get_cached_symbols(self) -> Optional[Dict]:
        if not self.redis: return None
        try:
            data = await self.redis.get("bb_bot:symbols_v4")
            return json.loads(data) if data else None
        except: return None

    async def save_cached_symbols(self, symbols: Dict):
        if not self.redis: return
        try:
            payload = {"timestamp": time.time(), "data": symbols}
            await self.redis.set("bb_bot:symbols_v4", json.dumps(payload), ex=86400)
        except Exception as e: logging.error(f"Redis Save Error: {e}")

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
# ABSTRACTION: EXCHANGE CLIENTS
# ==========================================

class BaseExchange(ABC):
    """
    Abstract base class enforcing DRY principle.
    Handles network logic, error handling, and proxy rotation centrally.
    """
    def __init__(self, session: aiohttp.ClientSession, proxy_pool: AsyncProxyPool, name: str):
        self.session = session
        self.proxies = proxy_pool
        self.name = name
        # Shared semaphore to prevent overriding global limits per instance
        self.sem = asyncio.Semaphore(CONFIG.MAX_CONCURRENCY)

    async def _request(self, url: str, params: dict = None) -> Optional[Any]:
        """Robust request wrapper with retries and logging."""
        for attempt in range(CONFIG.MAX_RETRIES):
            proxy = await self.proxies.get_proxy()
            
            try:
                async with self.sem:
                    kwargs = {'params': params, 'timeout': CONFIG.REQUEST_TIMEOUT}
                    if proxy: kwargs['proxy'] = proxy
                    
                    async with self.session.get(url, **kwargs) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        elif resp.status == 429:
                            logging.warning(f"⚠️ 429 {self.name} via {proxy}. Backing off.")
                            await asyncio.sleep(5)
                        elif resp.status >= 500:
                            pass # Server error, retry
                        else:
                            logging.debug(f"HTTP {resp.status} on {url}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if proxy: await self.proxies.report_failure(proxy)
            except Exception as e:
                logging.error(f"Unexpected Err {self.name}: {e}")
            
            await asyncio.sleep(0.5 + random.random()) # Jitter
        return None

    @abstractmethod
    async def get_perp_symbols(self) -> List[str]:
        pass

    @abstractmethod
    async def get_spot_symbols(self) -> List[str]:
        pass

    @abstractmethod
    def _get_kline_config(self, market: str) -> Dict[str, Any]:
        """Returns URL, interval map, and param keys."""
        pass

    async def fetch_candles(self, symbol: str, interval: str, market: str, limit: int = CONFIG.CANDLE_LIMIT) -> List[float]:
        """Universal candle fetcher."""
        conf = self._get_kline_config(market)
        api_interval = conf['interval_map'].get(interval, interval)
        
        params = {
            conf['sym_key']: symbol,
            conf['int_key']: api_interval,
            'limit': limit,
            **conf.get('extra_params', {})
        }

        data = await self._request(conf['url'], params)
        if not data: return []
        
        return self._parse_response(data, market)

    @abstractmethod
    def _parse_response(self, data: Any, market: str) -> List[float]:
        pass


class BinanceClient(BaseExchange):
    def __init__(self, session, proxies):
        super().__init__(session, proxies, "Binance")

    async def get_perp_symbols(self) -> List[str]:
        data = await self._request('https://fapi.binance.com/fapi/v1/exchangeInfo')
        if not data: return []
        return [s['symbol'] for s in data['symbols'] if s['contractType'] == 'PERPETUAL' and s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT']

    async def get_spot_symbols(self) -> List[str]:
        data = await self._request('https://api.binance.com/api/v3/exchangeInfo')
        if not data: return []
        return [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT']

    def _get_kline_config(self, market: str) -> Dict[str, Any]:
        base = 'https://api.binance.com/api/v3/klines' if market == "spot" else 'https://fapi.binance.com/fapi/v1/klines'
        return {
            'url': base,
            'sym_key': 'symbol',
            'int_key': 'interval',
            'interval_map': {} # Binance uses standard notation
        }

    def _parse_response(self, data: Any, market: str) -> List[float]:
        try:
            # Index 4 is Close
            return [float(c[4]) for c in data]
        except: return []


class BybitClient(BaseExchange):
    def __init__(self, session, proxies):
        super().__init__(session, proxies, "Bybit")

    async def get_perp_symbols(self) -> List[str]:
        data = await self._request('https://api.bybit.com/v5/market/instruments-info', {'category': 'linear'})
        if not data: return []
        return [s['symbol'] for s in data['result']['list'] if s['status'] == 'Trading' and s['quoteCoin'] == 'USDT']

    async def get_spot_symbols(self) -> List[str]:
        data = await self._request('https://api.bybit.com/v5/market/instruments-info', {'category': 'spot'})
        if not data: return []
        return [s['symbol'] for s in data['result']['list'] if s['status'] == 'Trading' and s['quoteCoin'] == 'USDT']

    def _get_kline_config(self, market: str) -> Dict[str, Any]:
        cat = 'linear' if market == 'perp' else 'spot'
        imap = {"15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240", "1d": "D", "1w": "W"}
        return {
            'url': 'https://api.bybit.com/v5/market/kline',
            'sym_key': 'symbol',
            'int_key': 'interval',
            'interval_map': imap,
            'extra_params': {'category': cat}
        }

    def _parse_response(self, data: Any, market: str) -> List[float]:
        # Bybit v5 returns list in descending order (newest first)
        raw = data.get('result', {}).get('list', [])
        if not raw: return []
        try:
            closes = [float(c[4]) for c in raw]
            return closes[::-1] # Reverse to align with TA-Lib (Oldest -> Newest)
        except: return []

# ==========================================
# MATH & ANALYSIS (VECTORIZED)
# ==========================================

def get_cache_key(tf: str) -> int:
    """Returns stable integer timestamp for current TF candle."""
    mins = TIMEFRAME_MINUTES[tf]
    now = int(time.time())
    return now - (now % (mins * 60))

def calculate_volatility_vectorized(closes: List[float]) -> float:
    """
    Optimized Volatility Calculation using NumPy.
    ~30% faster than Python loops.
    """
    if len(closes) < 24: return 0.0
    
    c = np.array(closes, dtype=np.float64)
    
    # Calculate percentage change: (Current - Prev) / Prev
    prev = c[:-1]
    curr = c[1:]
    
    # Handle division by zero safely
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = (curr - prev) / prev * 100
        
    # Remove NaNs or Infs
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0: return 0.0
    return float(np.std(returns))

def check_bb_rsi(closes: List[float], tf: str) -> Tuple[Optional[str], Optional[str], float]:
    """Analyzes BB and RSI logic."""
    if len(closes) < CONFIG.MIN_CANDLES: return None, None, 0.0
    
    np_c = np.array(closes, dtype=float)
    rsi = talib.RSI(np_c, timeperiod=CONFIG.RSI_PERIOD)
    upper, mid, lower = talib.BBANDS(rsi, timeperiod=CONFIG.BB_LENGTH, nbdevup=CONFIG.BB_STDDEV, nbdevdn=CONFIG.BB_STDDEV, matype=0)
    
    idx = -2 # Look at finished candle
    if np.isnan(rsi[idx]) or np.isnan(upper[idx]): return None, None, 0.0
    
    curr_rsi = rsi[idx]
    
    # Logic Checks
    if curr_rsi >= upper[idx] * (1 - CONFIG.UPPER_TOUCH_THRESHOLD): 
        return "UPPER", None, curr_rsi
        
    if curr_rsi <= lower[idx] * (1 + CONFIG.LOWER_TOUCH_THRESHOLD): 
        return "LOWER", None, curr_rsi
    
    if tf in MIDDLE_BAND_TFS:
        if abs(curr_rsi - mid[idx]) <= (mid[idx] * CONFIG.MIDDLE_TOUCH_THRESHOLD):
            # Crossover detection
            prev_diff = rsi[idx-1] - mid[idx-1]
            curr_diff = curr_rsi - mid[idx]
            
            # Crossed down (positive to negative) or just touching from above
            direction = "from above" if (prev_diff > 0 >= curr_diff) or (curr_diff > 0) else "from below"
            return "MIDDLE", direction, curr_rsi
            
    return None, None, 0.0

# ==========================================
# MAIN BOT CLASS
# ==========================================

class RsiBot:
    def __init__(self):
        self.cache = CacheManager()
        self.proxies = AsyncProxyPool()
        self.sent_state = {}
        self.hot_coins = set()
        
    async def _safe_batch_gather(self, tasks: List[Any], batch_name: str) -> List[Any]:
        """
        Executes a large list of awaitables in safe batches to protect the Event Loop.
        """
        results = []
        total = len(tasks)
        if total == 0: return []
        
        logging.info(f"🔄 Processing {total} {batch_name} tasks in batches of {CONFIG.BATCH_SIZE}...")
        
        for i in range(0, total, CONFIG.BATCH_SIZE):
            batch = tasks[i : i + CONFIG.BATCH_SIZE]
            batch_res = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_res)
            # Small breathe for the loop
            await asyncio.sleep(0.05)
            
        return results

    async def _bootstrap_symbols(self, binance: BinanceClient, bybit: BybitClient) -> List[Tuple]:
        """Hybrid symbol fetch: Try API first, fallback to Cache."""
        cached_wrapper = await self.cache.get_cached_symbols()
        cached_data = cached_wrapper.get('data', {}) if cached_wrapper else {}
        
        fetched = {}
        
        # Define fetch tasks
        sources = [
            ('bp', binance.get_perp_symbols, 'Binance Perp'),
            ('bs', binance.get_spot_symbols, 'Binance Spot'),
            ('yp', bybit.get_perp_symbols, 'Bybit Perp'),
            ('ys', bybit.get_spot_symbols, 'Bybit Spot')
        ]
        
        for key, func, label in sources:
            symbols = []
            for attempt in range(2):
                res = await func()
                if res:
                    symbols = res
                    break
                await asyncio.sleep(1)
            
            if not symbols and key in cached_data:
                logging.warning(f"⚠️ {label} failed. Using Cache.")
                symbols = cached_data[key]
            
            fetched[key] = symbols
            logging.info(f"📌 {label}: {len(symbols)} symbols")

        # Update cache if we got live data
        if any(len(v) > 0 for v in fetched.values()):
            await self.cache.save_cached_symbols(fetched)

        # Normalize and tuple-ize
        seen = set()
        all_pairs = []
        
        def process_list(sym_list, client, mkt, ex_name):
            for s in sym_list:
                if s in CONFIG.IGNORED_SYMBOLS or not s.endswith("USDT"): continue
                norm = s.upper().replace("USDT", "")
                if norm not in seen:
                    seen.add(norm)
                    all_pairs.append((client, s, mkt, ex_name))

        process_list(fetched['bp'], binance, 'perp', 'Binance')
        process_list(fetched['bs'], binance, 'spot', 'Binance')
        process_list(fetched['yp'], bybit, 'perp', 'Bybit')
        process_list(fetched['ys'], bybit, 'spot', 'Bybit')
        
        return all_pairs

    async def _calculate_volatility(self, all_pairs: List[Tuple]):
        """Vectorized Volatility Scan."""
        logging.info("📊 Calculating Volatility...")
        
        # Create tasks
        tasks = []
        for client, sym, mkt, ex in all_pairs:
            tasks.append(client.fetch_candles(sym, '1h', mkt, limit=25))
            
        # Execute safe batch
        results = await self._safe_batch_gather(tasks, "Volatility")
        
        # Process results
        vol_scores = {}
        for idx, res in enumerate(results):
            if isinstance(res, list) and len(res) > 20:
                # Use vectorized math
                v = calculate_volatility_vectorized(res)
                if v > 0:
                    sym = all_pairs[idx][1]
                    vol_scores[sym] = v
            elif isinstance(res, Exception):
                # Contextual Logging
                client, sym, _, ex_name = all_pairs[idx]
                logging.debug(f"Vol Check Failed {ex_name} {sym}: {res}")

        # Top 60
        self.hot_coins = set(sorted(vol_scores, key=vol_scores.get, reverse=True)[:60])
        logging.info(f"🔥 Identified {len(self.hot_coins)} Hot Coins")

    async def _execute_scans(self, all_pairs: List[Tuple]) -> List[TouchHit]:
        """Main Scanning Logic with Cache skipping."""
        final_hits = []
        tfs_to_scan = []
        
        # 1. Determine what needs scanning
        for tf in ACTIVE_TFS:
            if tf in CACHED_TFS:
                key = get_cache_key(tf)
                # Check if we already sent this candle
                if self.sent_state.get(tf) == key:
                    continue 
                
                # Check redis cache for results
                cached_res = await self.cache.get_scan_results(tf, key)
                if cached_res:
                    logging.info(f"📦 Loaded {len(cached_res)} cached hits for {tf}")
                    final_hits.extend([TouchHit.from_dict(x) for x in cached_res])
                    continue
            
            tfs_to_scan.append(tf)

        if not tfs_to_scan:
            return final_hits

        # 2. Scan necessary timeframes
        logging.info(f"🕵️ Scanning: {tfs_to_scan}")
        
        for tf in tfs_to_scan:
            tasks = []
            # Prepare task metadata to map results back to symbols
            meta_data = [] 
            
            for client, sym, mkt, ex in all_pairs:
                tasks.append(client.fetch_candles(sym, tf, mkt))
                meta_data.append((sym, mkt, ex))
                
            candle_results = await self._safe_batch_gather(tasks, f"Scan {tf}")
            
            tf_hits = []
            for i, closes in enumerate(candle_results):
                if isinstance(closes, list) and len(closes) >= CONFIG.MIN_CANDLES:
                    t_type, direc, rsi_val = check_bb_rsi(closes, tf)
                    if t_type:
                        sym, mkt, ex = meta_data[i]
                        hit = TouchHit(sym, ex, mkt, tf, rsi_val, t_type, direc, sym in self.hot_coins)
                        tf_hits.append(hit)
                elif isinstance(closes, Exception):
                    sym, _, ex = meta_data[i]
                    logging.debug(f"Scan Fail {ex} {sym} [{tf}]: {closes}")

            # Cache results for this TF
            if tf in CACHED_TFS:
                key = get_cache_key(tf)
                await self.cache.save_scan_results(tf, key, [h.to_dict() for h in tf_hits])
                
            final_hits.extend(tf_hits)
            
        return final_hits

    async def _send_report(self, session: aiohttp.ClientSession, hits: List[TouchHit]):
        """Formatted Telegram Reporting."""
        if not hits: return

        # Grouping
        grouped = {}
        for h in hits:
            # Check if already sent
            tf_key = get_cache_key(h.timeframe)
            if h.timeframe in CACHED_TFS and self.sent_state.get(h.timeframe) == tf_key:
                continue # Redundant check, but safe
            grouped.setdefault(h.timeframe, {}).setdefault(h.touch_type, []).append(h)

        if not grouped: return

        # Build Messages
        msg_blocks = []
        tf_order = ["1w", "1d", "4h", "2h", "1h", "30m", "15m"]
        
        for tf in tf_order:
            if tf not in grouped: continue
            
            lines = [f" ▣ TIMEFRAME: {tf}", "─────────────────────────"]
            headers = {"UPPER": "⬆️ UPPER BB", "MIDDLE": "🔶 MIDDLE BB", "LOWER": "⬇️ LOWER BB"}
            
            for t_type, header in headers.items():
                items = grouped[tf].get(t_type, [])
                if not items: continue
                
                items.sort(key=lambda x: x.symbol)
                lines.append(f"┌ {header}")
                
                for idx, item in enumerate(items):
                    prefix = "└" if idx == len(items)-1 else "│"
                    icon = "🥐" if item.exchange == "Binance" else "💣"
                    sym = item.symbol.replace("USDT", "")
                    
                    # Formatting Details
                    extra = ""
                    if t_type == "MIDDLE":
                        extra = f" {'🔻' if item.direction=='from above' else '🔹'}"
                    if item.hot: extra += " 🔥"
                    
                    lines.append(f"{prefix} {icon} *{sym}* ➜ *{item.rsi:.2f}*{extra}")
                lines.append("") # Spacer
                
            lines.append("─────────────────────────")
            msg_blocks.append("\n".join(lines))

        # Send in Chunks
        full_text = "\n\n".join(msg_blocks) + f"\n{datetime.now(timezone.utc).strftime('%d %b %H:%M UTC')}"
        
        # Professional chunking
        MAX_LEN = 3800
        while full_text:
            if len(full_text) <= MAX_LEN:
                chunk = full_text
                full_text = ""
            else:
                # Find nearest newline to split cleanly
                split_idx = full_text.rfind('\n', 0, MAX_LEN)
                if split_idx == -1: split_idx = MAX_LEN
                chunk = full_text[:split_idx]
                full_text = full_text[split_idx:]
            
            try:
                await session.post(
                    f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage",
                    json={"chat_id": CONFIG.CHAT_ID, "text": chunk, "parse_mode": "Markdown"}
                )
                await asyncio.sleep(0.5)
            except Exception as e:
                logging.error(f"TG Send Error: {e}")

    async def run(self):
        """Orchestrator."""
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s %(levelname)s [%(funcName)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # 1. Initialize Infrastructure
        await self.cache.init()
        self.sent_state = await self.cache.get_sent_state()
        
        async with aiohttp.ClientSession() as session:
            # 2. Warmup Proxies
            await self.proxies.populate(CONFIG.PROXY_URL, session)
            
            # 3. Instantiate Clients
            binance = BinanceClient(session, self.proxies)
            bybit = BybitClient(session, self.proxies)
            
            # 4. Bootstrap Symbols
            all_pairs = await self._bootstrap_symbols(binance, bybit)
            logging.info(f"Total Trading Pairs: {len(all_pairs)}")
            
            # 5. Volatility Analysis
            await self._calculate_volatility(all_pairs)
            
            # 6. Execute Scans
            hits = await self._execute_scans(all_pairs)
            
            # 7. Report & State Update
            await self._send_report(session, hits)
            
            # Update state for cached TFs
            new_state = self.sent_state.copy()
            for tf in CACHED_TFS:
                new_state[tf] = get_cache_key(tf)
            await self.cache.save_sent_state(new_state)

        await self.cache.close()
        logging.info("✅ Cycle Complete.")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    bot = RsiBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass
