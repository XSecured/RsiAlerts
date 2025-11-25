import asyncio
import aiohttp
import logging
import os
import time
import json
import random
import re
import math
import signal
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any, Set, NamedTuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial, wraps

# 3rd Party Libs
import numpy as np
import talib
import redis.asyncio as aioredis

# Try to use uvloop for maximum performance if available
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# === CONFIGURATION ===

@dataclass(frozen=True)
class AppConfig:
    # API Endpoints
    BINANCE_FUTURES_API: str = "https://fapi.binance.com/fapi/v1"
    BINANCE_SPOT_API: str = "https://api.binance.com/api/v3"
    PROXY_TEST_URL: str = "https://api.binance.com/api/v3/time"
    
    # Resources
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Strategy Params
    RSI_PERIOD: int = 14
    BB_LENGTH: int = 34
    BB_STDDEV: float = 2.0
    
    # Thresholds
    UPPER_TOUCH_THRESHOLD: float = 0.02
    LOWER_TOUCH_THRESHOLD: float = 0.02
    MIDDLE_TOUCH_THRESHOLD: float = 0.035
    
    # Limits
    CANDLE_LIMIT: int = 60
    MIN_CANDLES_TALIB: int = 36  # max(14, 34) + 2
    BATCH_SIZE: int = 100  # Increased batch size due to async efficiency
    
    # Logic Toggles
    TIMEFRAMES: Dict[str, bool] = field(default_factory=lambda: {
        '3m': False, '5m': False, '15m': True, '30m': True, 
        '1h': True, '2h': True, '4h': True, '1d': True, '1w': True
    })
    
    MIDDLE_BAND_ENABLED: Dict[str, bool] = field(default_factory=lambda: {
        '3m': False, '5m': False, '15m': True, '30m': True, 
        '1h': True, '2h': True, '4h': True, '1d': True, '1w': True
    })
    
    # Cache Settings
    CACHE_TTL: Dict[str, int] = field(default_factory=lambda: {
        '4h': 15000, '1d': 88000, '1w': 605000
    })
    CACHED_TFS: Set[str] = field(default_factory=lambda: {'4h', '1d', '1w'})
    RELAXED_LIMIT_TFS: Set[str] = field(default_factory=lambda: {'1d', '1w'})

    # Proxy Sources
    PROXY_SOURCES: List[str] = field(default_factory=lambda: [
        "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"
    ])

CONFIG = AppConfig()

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RsiBot")

# === DATA STRUCTURES ===

class ScanResult(NamedTuple):
    symbol: str
    timeframe: str
    rsi: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    touch_type: str  # UPPER, LOWER, MIDDLE
    timestamp: str
    hot: bool
    direction: Optional[str]
    market_type: str # FUTURES or SPOT

# === UTILS & DECORATORS ===

def retry_async(retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Robust retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_exception = e
                    if attempt == retries:
                        break
                    # logger.debug(f"Retry {attempt+1}/{retries} for {func.__name__} due to {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            raise last_exception
        return wrapper
    return decorator

# === CORE SERVICES ===

class ProxyManager:
    """
    Reactive Proxy Manager.
    Instead of actively checking health (wasteful), it maintains a scored list.
    Proxies are rotated on failure. High-scoring proxies are preferred.
    """
    def __init__(self):
        self.proxies: List[str] = []
        self.blacklist: Set[str] = set()
        self._lock = asyncio.Lock()
        
        # Load environment blocked proxies
        raw_blocked = os.getenv("PROXY_BLOCK_PERM", "")
        self.env_blocked = {p.strip() for p in raw_blocked.split(",") if p.strip()}

    async def fetch_proxies(self):
        """Fetches fresh proxies from sources."""
        raw_proxies = set()
        async with aiohttp.ClientSession() as session:
            for source in CONFIG.PROXY_SOURCES:
                try:
                    async with session.get(source, timeout=10) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            for line in text.splitlines():
                                p = line.strip()
                                if p and p not in self.env_blocked:
                                    if "://" not in p:
                                        p = f"http://{p}"
                                    raw_proxies.add(p)
                except Exception as e:
                    logger.warning(f"Failed to fetch proxy list from {source}: {e}")
        
        async with self._lock:
            self.proxies = list(raw_proxies)
            self.blacklist.clear()
            random.shuffle(self.proxies)
            logger.info(f"Proxy Pool Refreshed: {len(self.proxies)} proxies available.")

    async def get_proxy(self) -> Optional[str]:
        async with self._lock:
            # Simple strategy: Return a random one from the top 50% to balance load
            if not self.proxies:
                return None
            
            # Filter out blacklisted locally
            candidates = [p for p in self.proxies if p not in self.blacklist]
            if not candidates:
                # If all blacklisted, clear blacklist and recycle
                self.blacklist.clear()
                candidates = self.proxies
                
            return random.choice(candidates[:50] if len(candidates) > 50 else candidates)

    async def report_failure(self, proxy: str):
        """Mark a proxy as bad temporarily."""
        async with self._lock:
            self.blacklist.add(proxy)
            if proxy in self.proxies:
                # Move to end of list
                self.proxies.remove(proxy)
                self.proxies.append(proxy)


class NetworkClient:
    """
    Centralized HTTP Client utilizing aiohttp with connection pooling
    and automatic proxy rotation.
    """
    def __init__(self, proxy_manager: ProxyManager):
        self.proxy_manager = proxy_manager
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        conn = aiohttp.TCPConnector(
            limit=0, # Unlimited connections (OS limited)
            ttl_dns_cache=300,
            ssl=False # Performance trade-off, strictly verify in sensitive apps
        )
        self.session = aiohttp.ClientSession(connector=conn)

    async def close(self):
        if self.session:
            await self.session.close()

    async def request(self, method: str, url: str, params: dict = None, attempts: int = 3) -> Any:
        for _ in range(attempts):
            proxy = await self.proxy_manager.get_proxy()
            try:
                async with self.session.request(
                    method, url, params=params, proxy=proxy, timeout=15
                ) as resp:
                    if resp.status == 451 or resp.status == 403:
                        # Geo-block or Forbidden
                        await self.proxy_manager.report_failure(proxy)
                        continue
                    
                    resp.raise_for_status()
                    return await resp.json()
            except Exception:
                await self.proxy_manager.report_failure(proxy)
                await asyncio.sleep(0.2)
        
        # If all proxy attempts fail, try direct (if not running in restricted region)
        # or raise error
        return None

class CacheService:
    """Redis Cache Manager."""
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self):
        try:
            self.redis = await aioredis.from_url(
                CONFIG.REDIS_URL, encoding="utf-8", decode_responses=True
            )
            await self.redis.ping()
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis = None

    async def close(self):
        if self.redis:
            await self.redis.close()

    def _key(self, tf: str, open_time: str) -> str:
        return f"bot:scan:{tf}:{open_time}"

    async def get_results(self, tf: str, open_time: str) -> Optional[List[Dict]]:
        if not self.redis: return None
        data = await self.redis.get(self._key(tf, open_time))
        return json.loads(data) if data else None

    async def set_results(self, tf: str, open_time: str, data: List[Dict]):
        if not self.redis: return
        ttl = CONFIG.CACHE_TTL.get(tf, 3600)
        await self.redis.set(self._key(tf, open_time), json.dumps(data), ex=ttl)

    async def get_state(self) -> Dict[str, str]:
        if not self.redis: return {}
        data = await self.redis.get("bot:sent_state")
        return json.loads(data) if data else {}

    async def set_state(self, state: Dict[str, str]):
        if not self.redis: return
        await self.redis.set("bot:sent_state", json.dumps(state))

class TechnicalAnalysisEngine:
    """
    Handles heavy lifting.
    Uses ProcessPoolExecutor to run Talib calculations outside the Event Loop.
    """
    def __init__(self):
        self.executor = ProcessPoolExecutor(max_workers=4)

    def shutdown(self):
        self.executor.shutdown()

    @staticmethod
    def _calculate_cpu_bound(closes: List[float]) -> Tuple[float, float, float, float]:
        """Runs in a separate process."""
        try:
            np_closes = np.array(closes, dtype=float)
            
            # RSI
            rsi_arr = talib.RSI(np_closes, timeperiod=CONFIG.RSI_PERIOD)
            
            # BB
            upper, middle, lower = talib.BBANDS(
                rsi_arr, 
                timeperiod=CONFIG.BB_LENGTH,
                nbdevup=CONFIG.BB_STDDEV, 
                nbdevdn=CONFIG.BB_STDDEV, 
                matype=0
            )
            
            # Return last valid values
            # We look at index -2 (completed candle) usually, or -1 if live
            # The logic in original script was -2
            idx = -2
            if len(np_closes) < abs(idx): 
                return (np.nan,)*4
                
            return (rsi_arr[idx], upper[idx], middle[idx], lower[idx])
        except Exception:
            return (np.nan,)*4

    async def analyze(self, closes: List[float]) -> Tuple[float, float, float, float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, self._calculate_cpu_bound, closes
        )

class NotificationService:
    """Fully Async Telegram Notifier."""
    def __init__(self):
        self.base_url = f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage"

    async def send(self, message: str, session: aiohttp.ClientSession):
        if not CONFIG.TELEGRAM_TOKEN or not CONFIG.TELEGRAM_CHAT_ID:
            return

        payload = {
            'chat_id': CONFIG.TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }
        
        try:
            async with session.post(self.base_url, json=payload, timeout=10) as resp:
                if resp.status != 200:
                    logger.error(f"Telegram failed: {resp.status} - {await resp.text()}")
        except Exception as e:
            logger.error(f"Telegram connection error: {e}")

# === MAIN LOGIC ===

class RsiBot:
    def __init__(self):
        self.proxy_manager = ProxyManager()
        self.network = NetworkClient(self.proxy_manager)
        self.cache = CacheService()
        self.ta_engine = TechnicalAnalysisEngine()
        self.notifier = NotificationService()
        self.hot_coins: Set[str] = set()

    async def initialize(self):
        await self.cache.connect()
        await self.proxy_manager.fetch_proxies()
        await self.network.start()

    async def shutdown(self):
        await self.network.close()
        await self.cache.close()
        self.ta_engine.shutdown()

    # --- Market Data Fetching ---

    async def get_symbols(self) -> Tuple[List[str], List[str]]:
        """Fetch Futures and Spot symbols concurrently."""
        f_task = self.network.request('GET', f"{CONFIG.BINANCE_FUTURES_API}/exchangeInfo")
        s_task = self.network.request('GET', f"{CONFIG.BINANCE_SPOT_API}/exchangeInfo")
        
        f_data, s_data = await asyncio.gather(f_task, s_task)
        
        f_symbols = set()
        if f_data:
            for s in f_data.get('symbols', []):
                if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING':
                    f_symbols.add(s['symbol'])
        
        s_symbols = set()
        if s_data:
            for s in s_data.get('symbols', []):
                if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING':
                    s_symbols.add(s['symbol'])
        
        # Spot symbols not in Futures
        s_unique = list(s_symbols - f_symbols)
        return list(f_symbols), sorted(s_unique)

    async def get_candles(self, symbol: str, interval: str, is_futures: bool) -> List[float]:
        url = f"{CONFIG.BINANCE_FUTURES_API}/klines" if is_futures else f"{CONFIG.BINANCE_SPOT_API}/klines"
        # Determine limit based on logic
        limit = CONFIG.CANDLE_LIMIT
        
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        data = await self.network.request('GET', url, params=params)
        
        if not data: return []
        # Return closes
        return [float(k[4]) for k in data]

    # --- Strategy ---

    async def update_volatility_rankings(self, all_symbols: List[Tuple[str, bool]]):
        """Calculates volatility for ALL symbols efficiently."""
        logger.info("Updating volatility rankings...")
        scores = []
        
        # Limit concurrency to avoid getting banned even with proxies
        sem = asyncio.Semaphore(100) 

        async def check_vol(sym, is_fut):
            async with sem:
                closes = await self.get_candles(sym, '1h', is_fut)
                if len(closes) >= 24:
                    # Calculate pct change manually to avoid pandas overhead
                    returns = [(closes[i] - closes[i-1])/closes[i-1] for i in range(1, len(closes))]
                    if returns:
                        vol = float(np.std(returns))
                        return (sym, vol)
            return None

        # Process in chunks
        tasks = [check_vol(s, f) for s, f in all_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in results:
            if r and isinstance(r, tuple):
                scores.append(r)
        
        scores.sort(key=lambda x: x[1], reverse=True)
        self.hot_coins = {x[0] for x in scores[:60]}
        logger.info(f"Top Volatile Coins Updated: {len(self.hot_coins)}")

    async def process_symbol_timeframe(
        self, symbol: str, timeframe: str, is_futures: bool
    ) -> Optional[ScanResult]:
        
        closes = await self.get_candles(symbol, timeframe, is_futures)
        
        # Validation
        min_req = CONFIG.MIN_CANDLES_TALIB
        if timeframe not in CONFIG.RELAXED_LIMIT_TFS:
             if len(closes) < CONFIG.CANDLE_LIMIT: return None
        elif len(closes) < min_req:
            return None

        # Offload Math
        rsi, upper, middle, lower = await self.ta_engine.analyze(closes)
        
        if np.isnan(rsi): return None

        # Logic
        touch_type = None
        direction = None
        
        # Check Upper
        if rsi >= upper * (1 - CONFIG.UPPER_TOUCH_THRESHOLD):
            touch_type = "UPPER"
        # Check Lower
        elif rsi <= lower * (1 + CONFIG.LOWER_TOUCH_THRESHOLD):
            touch_type = "LOWER"
        # Check Middle (if enabled)
        elif CONFIG.MIDDLE_BAND_ENABLED.get(timeframe):
            if abs(rsi - middle) <= middle * CONFIG.MIDDLE_TOUCH_THRESHOLD:
                touch_type = "MIDDLE"
                # Determine direction (simplified for perf, relying on single point check)
                direction = "from above" if rsi > middle else "from below"

        if touch_type:
            # Reconstruct timestamp logic
            # Note: kline[0] is open time. We used -2 index (previous closed candle)
            # Efficient timestamp gen
            now = datetime.utcnow()
            return ScanResult(
                symbol=symbol,
                timeframe=timeframe,
                rsi=round(rsi, 2),
                bb_upper=upper,
                bb_middle=middle,
                bb_lower=lower,
                touch_type=touch_type,
                timestamp=now.strftime('%Y-%m-%d %H:%M:%S UTC'),
                hot=(symbol in self.hot_coins),
                direction=direction,
                market_type='FUTURES' if is_futures else 'SPOT'
            )
        return None

    # --- Orchestration ---

    async def run_scan(self):
        start_ts = time.time()
        
        # 1. Get State & Symbols
        sent_state = await self.cache.get_state()
        f_syms, s_syms = await self.get_symbols()
        all_pairs = [(s, True) for s in f_syms] + [(s, False) for s in s_syms]
        
        logger.info(f"Scanning {len(all_pairs)} pairs across {len(CONFIG.TIMEFRAMES)} TFs.")

        # 2. Update Volatility (Background task or pre-requisite? Let's do pre-req)
        await self.update_volatility_rankings(all_pairs)

        # 3. Determine Timeframes to scan
        # Calculate current candle open times
        now = datetime.utcnow()
        tf_open_times = {}
        
        # Minutes map
        tf_min = {
            '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, 
            '1d': 1440, '1w': 10080
        }

        # Identify which TFs are "new" (candle just closed/opened)
        to_alert_cache_tfs = set()
        
        for tf, enabled in CONFIG.TIMEFRAMES.items():
            if not enabled: continue
            if tf not in tf_min: continue # skip non-standard if any
            
            total_mins = now.hour * 60 + now.minute
            mins = tf_min[tf]
            intervals = total_mins // mins
            current_open = (now.replace(hour=0, minute=0, second=0, microsecond=0) + 
                          timedelta(minutes=intervals*mins)).isoformat()
            
            tf_open_times[tf] = current_open
            
            if tf in CONFIG.CACHED_TFS:
                last_sent = sent_state.get(tf)
                if last_sent != current_open:
                    to_alert_cache_tfs.add(tf)
        
        # 4. Scanning Logic
        # We scan:
        # A. Uncached TFs (Always)
        # B. Cached TFs (Only if they aren't in Redis OR if we need to alert/refresh them)
        
        active_tfs = [t for t, e in CONFIG.TIMEFRAMES.items() if e]
        uncached_tfs = [t for t in active_tfs if t not in CONFIG.CACHED_TFS]
        cached_tfs_to_scan = []
        
        # Check cache existence
        cached_results = {}
        for tf in CONFIG.CACHED_TFS:
            if tf in active_tfs:
                res = await self.cache.get_results(tf, tf_open_times[tf])
                if res is not None:
                    # Hydrate objects
                    cached_results[tf] = [ScanResult(**r) for r in res]
                    # Update hot status dynamically even if cached
                    for r in cached_results[tf]:
                         # NamedTuple is immutable, slight hack or reconstruct
                         pass # Skip for speed, or re-create list
                else:
                    cached_tfs_to_scan.append(tf)

        tfs_to_process_fresh = uncached_tfs + cached_tfs_to_scan
        
        final_results = []
        
        # Batch Processor
        if tfs_to_process_fresh:
            logger.info(f"Fresh scanning TFs: {tfs_to_process_fresh}")
            
            # Create a massive list of jobs
            # (symbol, tf, is_futures)
            jobs = []
            for tf in tfs_to_process_fresh:
                for sym, is_fut in all_pairs:
                    jobs.append((sym, tf, is_fut))
            
            # Run concurrently with semaphore
            sem = asyncio.Semaphore(CONFIG.BATCH_SIZE)
            
            async def worker(job):
                async with sem:
                    try:
                        return await self.process_symbol_timeframe(*job)
                    except Exception as e:
                        # logger.debug(f"Scan error {job}: {e}")
                        return None

            results = await asyncio.gather(*[worker(j) for j in jobs], return_exceptions=True)
            
            # Group fresh results by TF for caching
            fresh_grouped = {tf: [] for tf in tfs_to_process_fresh}
            
            for res in results:
                if isinstance(res, ScanResult):
                    final_results.append(res)
                    fresh_grouped[res.timeframe].append(asdict(res))

            # Save to Cache
            for tf, data in fresh_grouped.items():
                if tf in CONFIG.CACHED_TFS:
                    await self.cache.set_results(tf, tf_open_times[tf], data)
        
        # Add cached results to final list
        for tf, items in cached_results.items():
            final_results.extend(items)

        # 5. Filtering & Alerting
        # We alert if:
        # A. It's an uncached TF
        # B. It's a cached TF AND it's in `to_alert_cache_tfs` (New candle)
        
        alerts_to_send = []
        
        # Sort for display
        final_results.sort(key=lambda x: (x.timeframe, x.symbol))
        
        # Group by Timeframe
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in final_results:
            grouped[r.timeframe].append(r)
            
        # Format and queue messages
        for tf, items in grouped.items():
            should_send = (tf in uncached_tfs) or (tf in to_alert_cache_tfs)
            if not should_send:
                continue
                
            msg = self.format_message(tf, items)
            alerts_to_send.append(msg)
            
            # Update state if sent
            if tf in CONFIG.CACHED_TFS:
                sent_state[tf] = tf_open_times[tf]

        # Dispatch Alerts
        if alerts_to_send:
            for msg_chunk in alerts_to_send:
                for chunk in self.split_msg(msg_chunk):
                    await self.notifier.send(chunk, self.network.session)
        else:
            logger.info("No new alerts to trigger.")

        # Save State
        await self.cache.set_state(sent_state)
        
        logger.info(f"Run completed in {time.time() - start_ts:.2f}s")

    def format_message(self, timeframe: str, results: List[ScanResult]) -> str:
        if not results: return ""
        
        header = f"*🔍 BB Touches on {timeframe} ({len(results)} pairs)*\n"
        
        upper = [r for r in results if r.touch_type == 'UPPER']
        middle = [r for r in results if r.touch_type == 'MIDDLE']
        lower = [r for r in results if r.touch_type == 'LOWER']
        
        lines = [header]
        
        def _fmt(r: ScanResult):
            icon = "🔥" if r.hot else ""
            mkt = "[]" if r.market_type == 'FUTURES' else "[]"
            extra = ""
            if r.touch_type == "MIDDLE":
                arrow = "🔻" if r.direction == "from above" else "🔹"
                extra = f" ({arrow})"
            return f"• *{r.symbol}* {mkt} RSI:{r.rsi:.1f}{extra} {icon}"

        if upper:
            lines.append("\n*⬆️ UPPER BAND:*")
            lines.extend([_fmt(r) for r in upper])
            
        if middle:
            lines.append("\n*🔶 MIDDLE BAND:*")
            lines.extend([_fmt(r) for r in middle])
            
        if lower:
            lines.append("\n*⬇️ LOWER BAND:*")
            lines.extend([_fmt(r) for r in lower])
            
        lines.append(f"\n_Generated: {datetime.utcnow().strftime('%H:%M UTC')}_")
        return "\n".join(lines)

    def split_msg(self, text: str, limit: int = 4000) -> List[str]:
        if len(text) <= limit: return [text]
        chunks = []
        curr = ""
        for line in text.split('\n'):
            if len(curr) + len(line) + 1 > limit:
                chunks.append(curr)
                curr = line
            else:
                curr += "\n" + line if curr else line
        if curr: chunks.append(curr)
        return chunks

async def main():
    bot = RsiBot()
    try:
        await bot.initialize()
        await bot.run_scan()
    except Exception as e:
        logger.critical(f"Fatal Bot Error: {e}", exc_info=True)
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
