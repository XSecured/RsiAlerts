import asyncio
import logging
import os
import time
import random
import re
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict, Any, Set, Final, Union
from itertools import cycle

# === 3rd Party High-Performance Libs ===
import aiohttp
import numpy as np
import talib
import redis.asyncio as aioredis

# Try to import high-performance replacements
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass  # Fallback to standard loop

try:
    import orjson as json  # Rust-based fast JSON
except ImportError:
    import json

# === MONKEY-PATCH FOR HTTPS PROXIES ===
import ssl
import aiohttp

if hasattr(aiohttp, 'ClientSession'):
    original_create_connection = aiohttp.connector.TCPConnector._create_connection

    async def _create_connection(self, req, traces, timeout):
        proxy_host = req.proxy.split(':')[1][2:]  # Extract proxy host
        proxy_port = int(req.proxy.split(':')[2])  # Extract proxy port
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        return await original_create_connection(self, req, traces, timeout)

    aiohttp.connector.TCPConnector._create_connection = _create_connection

# === CONFIGURATION ===

@dataclass(frozen=True)
class Config:
    # API Endpoints
    BINANCE_FUTURES_EXCHANGE: Final[str] = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    BINANCE_FUTURES_KLINES: Final[str] = "https://fapi.binance.com/fapi/v1/klines"
    BINANCE_SPOT_EXCHANGE: Final[str] = "https://api.binance.com/api/v3/exchangeInfo"
    BINANCE_SPOT_KLINES: Final[str] = "https://api.binance.com/api/v3/klines"
    PROXY_TEST_URL: Final[str] = "https://api.binance.com/api/v3/time"
    
    # Strategy Params
    RSI_PERIOD: Final[int] = 14
    BB_LENGTH: Final[int] = 34
    BB_STDDEV: Final[float] = 2.0
    UPPER_TOUCH_THRESHOLD: Final[float] = 0.02
    MIDDLE_TOUCH_THRESHOLD: Final[float] = 0.035
    LOWER_TOUCH_THRESHOLD: Final[float] = 0.02
    
    # Data Fetching
    CANDLE_LIMIT: Final[int] = 60
    MIN_CANDLES_TALIB: Final[int] = 36  # max(14, 34) + 2
    
    # Proxies
    PROXY_SOURCES: Final[List[str]] = field(default_factory=lambda: [
        "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"
    ])
    PROXY_BLOCK_PERM: Final[str] = os.getenv("PROXY_BLOCK_PERM", "201.174.239.25:8080")
    
    # Redis
    REDIS_URL: Final[str] = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_PREFIX: Final[str] = "bb_touch"
    
    # Telegram
    BOT_TOKEN: Final[str] = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID: Final[str] = os.getenv("TELEGRAM_CHAT_ID", "")

    # Timeframes Configuration
    TIMEFRAMES_TOGGLE: Final[Dict[str, bool]] = field(default_factory=lambda: {
        '3m': False, '5m': False, '15m': True, '30m': True,
        '1h': True, '2h': True, '4h': True, '1d': True, '1w': True,
    })
    
    MIDDLE_BAND_TOGGLE: Final[Dict[str, bool]] = field(default_factory=lambda: {
        '3m': False, '5m': False, '15m': True, '30m': True,
        '1h': True, '2h': True, '4h': True, '1d': True, '1w': True,
    })
    
    TIMEFRAME_MAP: Final[Dict[str, int]] = field(default_factory=lambda: {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
        '12h': 720, '1d': 1440, '1w': 10080
    })

    CACHE_TTL: Final[Dict[str, int]] = field(default_factory=lambda: {
        '4h': 14400 + 600, '1d': 86400 + 1800, '1w': 604800 + 3600
    })
    
    CACHED_TFS: Final[Set[str]] = field(default_factory=lambda: {'4h', '1d', '1w'})
    HIGH_TF_RELAX: Final[Set[str]] = field(default_factory=lambda: {'1d', '1w'})

CONF = Config()

# === LOGGING ===

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("BB_Bot")

# === UTILS ===

def get_candle_open(timeframe: str, now: Optional[datetime] = None) -> datetime:
    """Calculates accurate candle open time."""
    if now is None:
        now = datetime.now(timezone.utc)
    
    minutes = CONF.TIMEFRAME_MAP.get(timeframe)
    if not minutes:
        raise ValueError(f"Invalid timeframe: {timeframe}")
        
    total_mins = now.hour * 60 + now.minute
    intervals = total_mins // minutes
    open_minutes = intervals * minutes
    
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=open_minutes)

def normalize_proxy(p: str) -> str:
    p = p.strip()
    return p if "://" in p else f"http://{p}"

# === PROXY MANAGER ===

# === PROXY MANAGER ===

class ProxyManager:
    """
    High-performance Proxy Rotator with health tracking and async locking.
    """
    def __init__(self):
        self.proxies: List[str] = []
        self.blacklist: Set[str] = set()
        self.failures: Dict[str, int] = {}
        self._cycle = None
        self._lock = asyncio.Lock()
        self.perm_block = {normalize_proxy(p) for p in CONF.PROXY_BLOCK_PERM.split(",") if p.strip()}
        
    async def initialize(self, session: aiohttp.ClientSession):
        """Fetches and verifies proxies initially."""
        await self._fetch_proxies(session)
        if not self.proxies:
            logger.warning("No proxies found initially. Retrying...")
            await self._fetch_proxies(session)

    async def _fetch_proxies(self, session: aiohttp.ClientSession):
        raw_proxies = set()
        for url in CONF.PROXY_SOURCES:
            try:
                logger.info(f"Fetching proxies from {url}...")
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        if text:
                            for line in text.splitlines():
                                p = line.strip()
                                if p: 
                                    p_norm = normalize_proxy(p)
                                    if p_norm not in self.perm_block and p_norm not in self.blacklist:
                                        raw_proxies.add(p_norm)
                        else:
                            logger.warning(f"No content returned from {url}.")
                    else:
                        logger.error(f"Failed to fetch proxies: HTTP {resp.status}")
            except Exception as e:
                logger.error(f"Proxy fetch failed from {url}: {e}")

        # Fast concurrent validation
        valid_proxies = await self._validate_concurrently(session, list(raw_proxies))
        
        async with self._lock:
            self.proxies = valid_proxies
            self._cycle = cycle(self.proxies)
            # Reset blacklist if pool is critically low
            if len(self.proxies) < 5:
                self.blacklist.clear()
                self.failures.clear()
            logger.info(f"Proxy Pool Refreshed: {len(self.proxies)} active proxies.")

    async def _validate_concurrently(self, session: aiohttp.ClientSession, proxies: List[str]) -> List[str]:
        """Checks proxies in parallel with high concurrency."""
        valid = []
        sem = asyncio.Semaphore(100) # High concurrency for checking

        async def check(p):
            async with sem:
                try:
                    start = time.perf_counter()
                    async with session.get(CONF.PROXY_TEST_URL, proxy=p, timeout=5) as resp:
                        if 200 <= resp.status < 300:
                            # Prefer fast proxies
                            return p if (time.perf_counter() - start) < 3.0 else None
                except Exception:
                    pass
            return None

        tasks = [check(p) for p in proxies]
        results = await asyncio.gather(*tasks)
        return [p for p in results if p]

    async def get(self) -> Optional[str]:
        """Get next proxy."""
        async with self._lock:
            if not self.proxies:
                return None
            # Simple round-robin is O(1) and sufficient
            for _ in range(len(self.proxies)):
                p = next(self._cycle)
                if p not in self.blacklist:
                    return p
            return None

    async def report_failure(self, proxy: str):
        """Mark a proxy as failed."""
        async with self._lock:
            self.failures[proxy] = self.failures.get(proxy, 0) + 1
            if self.failures[proxy] >= 3:
                self.blacklist.add(proxy)
                if proxy in self.proxies:
                    # We don't remove from list to avoid mutating while cycling, 
                    # just blacklist it so get() skips it.
                    pass
                logger.debug(f"Blacklisted proxy: {proxy}")

# === REDIS CACHE ===

class Cache:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self):
        try:
            self.redis = await aioredis.from_url(
                CONF.REDIS_URL, encoding="utf-8", decode_responses=True
            )
            await self.redis.ping()
            logger.info("Redis connected.")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis = None

    async def close(self):
        if self.redis:
            await self.redis.close()

    async def get_json(self, key: str) -> Any:
        if not self.redis: return None
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def set_json(self, key: str, value: Any, ttl: int = None):
        if not self.redis: return
        await self.redis.set(key, json.dumps(value), ex=ttl)

    async def get_sent_state(self) -> Dict[str, str]:
        res = await self.get_json(f"{CONF.CACHE_PREFIX}:sent_state")
        return res if res else {}

    async def save_sent_state(self, state: Dict[str, str]):
        await self.set_json(f"{CONF.CACHE_PREFIX}:sent_state", state)

# === MARKET DATA ENGINE ===

class MarketDataEngine:
    def __init__(self, proxy_manager: ProxyManager):
        self.pm = proxy_manager
        # Optimized connector
        connector = aiohttp.TCPConnector(
            limit=200, # High concurrency
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            ssl=False # Trade-off for speed if using http proxies, otherwise True
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            json_serialize=lambda x: json.dumps(x).decode()
        )
        self.sem = asyncio.Semaphore(50) # Rate limit protection

    async def close(self):
        await self.session.close()

    async def _request(self, url: str, params: dict = None, retries: int = 3) -> Any:
        """Robust request wrapper with proxy rotation."""
        for attempt in range(retries):
            proxy = await self.pm.get()
            if not proxy and attempt > 0:
                # If out of proxies, try one direct request or wait
                await asyncio.sleep(1)
                continue

            try:
                async with self.sem: # Semaphore controls concurrency
                    async with self.session.get(
                        url, params=params, proxy=proxy, timeout=10
                    ) as resp:
                        if resp.status == 429: # Rate limit
                            logger.warning("Rate limit hit (429). Cooling down.")
                            await asyncio.sleep(5)
                            continue
                        if resp.status == 418: # IP Ban
                            if proxy: await self.pm.report_failure(proxy)
                            continue
                        
                        resp.raise_for_status()
                        # Use orjson for fast parsing
                        return json.loads(await resp.read())
            except Exception:
                if proxy: await self.pm.report_failure(proxy)
                if attempt == retries - 1: raise
                # Backoff
                await asyncio.sleep(0.5 * (attempt + 1))
        return None

    async def get_symbols(self) -> Tuple[List[str], List[str]]:
        """Fetches Futures and Spot symbols concurrently."""
        fut_task = self._request(CONF.BINANCE_FUTURES_EXCHANGE)
        spot_task = self._request(CONF.BINANCE_SPOT_EXCHANGE)
        
        res_fut, res_spot = await asyncio.gather(fut_task, spot_task, return_exceptions=True)
        
        futures_syms = set()
        if isinstance(res_fut, dict):
            for s in res_fut.get('symbols', []):
                if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING':
                    futures_syms.add(s['symbol'])
        
        spot_syms = []
        if isinstance(res_spot, dict):
            for s in res_spot.get('symbols', []):
                if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING' and s['symbol'] not in futures_syms:
                    spot_syms.append(s['symbol'])
                    
        return sorted(list(futures_syms)), sorted(spot_syms)

    async def get_volatility_rankings(self, symbols_map: List[Tuple[str, bool]], top_n: int = 60) -> Set[str]:
        """
        Calculates volatility.
        Optimization: Uses asyncio.gather with semaphore to fetch 1h klines in parallel.
        Algorithm: StdDev of returns over last 24 hours.
        """
        scores = []
        
        async def fetch_vol(symbol: str, is_futures: bool):
            url = CONF.BINANCE_FUTURES_KLINES if is_futures else CONF.BINANCE_SPOT_KLINES
            try:
                # Fetch just enough data for 24h calculation
                data = await self._request(url, {'symbol': symbol, 'interval': '1h', 'limit': 25})
                if not data or len(data) < 25: return None
                
                # Fast numpy calculation
                closes = np.array([float(x[4]) for x in data])
                # Calculate pct change
                returns = np.diff(closes) / closes[:-1]
                vol = np.std(returns)
                return (symbol, vol)
            except Exception:
                return None

        # Batch processing
        tasks = [fetch_vol(s, f) for s, f in symbols_map]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in results:
            if isinstance(r, tuple):
                scores.append(r)
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top = {x[0] for x in scores[:top_n]}
        logger.info(f"Identified {len(top)} volatile assets.")
        return top

    async def fetch_klines(self, symbol: str, interval: str, is_futures: bool) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        url = CONF.BINANCE_FUTURES_KLINES if is_futures else CONF.BINANCE_SPOT_KLINES
        try:
            data = await self._request(url, {'symbol': symbol, 'interval': interval, 'limit': CONF.CANDLE_LIMIT})
            if not data: return None, None
            
            # Zero-copy optimization: Parse directly to float list then array
            # Index 4 is Close, Index 0 is Open Time
            closes = np.array([float(x[4]) for x in data], dtype=np.float64)
            times = [int(x[0]) for x in data]
            return closes, times
        except Exception:
            return None, None

# === TECHNICAL ANALYSIS ===

class TechnicalAnalysis:
    @staticmethod
    def analyze(symbol: str, timeframe: str, closes: np.ndarray, timestamps: List[int], is_hot: bool, is_futures: bool) -> Optional[Dict]:
        """
        Pure math function. Returns result dict if touch detected, else None.
        """
        if len(closes) < CONF.MIN_CANDLES_TALIB: return None
        
        # TALIB Calculation (C-speed)
        rsi = talib.RSI(closes, timeperiod=CONF.RSI_PERIOD)
        upper, middle, lower = talib.BBANDS(rsi, timeperiod=CONF.BB_LENGTH, nbdevup=CONF.BB_STDDEV, nbdevdn=CONF.BB_STDDEV, matype=0)
        
        # Analyze the second to last candle (last closed candle)
        idx = -2
        c_rsi, c_up, c_mid, c_low = rsi[idx], upper[idx], middle[idx], lower[idx]
        
        if np.isnan(c_rsi) or np.isnan(c_up): return None
        
        touch_type = None
        direction = None
        
        # Logic Checks
        if c_rsi >= c_up * (1 - CONF.UPPER_TOUCH_THRESHOLD):
            touch_type = "UPPER"
        elif c_rsi <= c_low * (1 + CONF.LOWER_TOUCH_THRESHOLD):
            touch_type = "LOWER"
        elif CONF.MIDDLE_BAND_TOGGLE.get(timeframe, False):
            if abs(c_rsi - c_mid) <= c_mid * CONF.MIDDLE_TOUCH_THRESHOLD:
                # Check previous candle for direction
                p_rsi, p_mid = rsi[idx-1], middle[idx-1]
                prev_diff = p_rsi - p_mid
                curr_diff = c_rsi - c_mid
                
                if prev_diff > 0 >= curr_diff: direction = "from above"
                elif prev_diff < 0 <= curr_diff: direction = "from below"
                # Only trigger middle if not upper/lower
                if not touch_type: touch_type = "MIDDLE"

        if touch_type:
            ts_str = datetime.fromtimestamp(timestamps[idx] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'rsi': float(c_rsi), # convert numpy float to python float for json
                'bb_upper': float(c_up),
                'bb_middle': float(c_mid),
                'bb_lower': float(c_low),
                'touch_type': touch_type,
                'timestamp': ts_str,
                'hot': is_hot,
                'direction': direction,
                'market_type': 'FUTURES' if is_futures else 'SPOT'
            }
        return None

# === TELEGRAM SENDER ===

async def send_telegram(session: aiohttp.ClientSession, message: str):
    """Async Telegram Sender."""
    if not CONF.BOT_TOKEN or not CONF.CHAT_ID: return
    
    url = f"https://api.telegram.org/bot{CONF.BOT_TOKEN}/sendMessage"
    
    # Split logic generator
    def chunk_msg(text, limit=4000):
        lines = text.split('\n')
        chunk = ""
        for line in lines:
            if len(chunk) + len(line) + 1 > limit:
                yield chunk
                chunk = line
            else:
                chunk = (chunk + "\n" + line) if chunk else line
        if chunk: yield chunk

    for chunk in chunk_msg(message):
        payload = {
            'chat_id': CONF.CHAT_ID,
            'text': chunk,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    logger.error(f"Telegram Error: {await resp.text()}")
        except Exception as e:
            logger.error(f"Telegram Exception: {e}")
        await asyncio.sleep(0.2) # Rate limit nice-ness

# === MAIN BOT LOGIC ===

class BB_Bot:
    def __init__(self):
        self.cache = Cache()
        self.proxies = ProxyManager()
        self.mde = None # Initialized in run

    async def run(self):
        start_ts = time.perf_counter()
        
        # 1. Init Resources
        await self.cache.connect()
        self.mde = MarketDataEngine(self.proxies)
        await self.proxies.initialize(self.mde.session)
        
        try:
            # 2. Get Symbols
            f_syms, s_syms = await self.mde.get_symbols()
            all_syms = [(s, True) for s in f_syms] + [(s, False) for s in s_syms]
            logger.info(f"Symbols: {len(f_syms)} Futures, {len(s_syms)} Spot.")

            # 3. Volatility (The heavy lifting)
            hot_coins = await self.mde.get_volatility_rankings(all_syms)

            # 4. Determine Workload
            active_tfs = [tf for tf, on in CONF.TIMEFRAMES_TOGGLE.items() if on]
            sent_state = await self.cache.get_sent_state()
            
            # Calculate current open times
            now = datetime.now(timezone.utc)
            current_opens = {tf: get_candle_open(tf, now).isoformat() for tf in active_tfs}
            
            # Determine which Cached TFs need processing/sending
            # Logic: Non-cached TFs always run. Cached TFs run if cache missing OR new candle.
            tfs_to_scan_fresh = []
            tfs_from_cache = []
            tfs_to_notify = set() # Which TFs we should send alerts for

            for tf in active_tfs:
                is_cached_type = tf in CONF.CACHED_TFS
                candle_open = current_opens[tf]
                
                # Should we notify for this TF?
                # Yes, if it's not a cached type (always notify)
                # OR if it is a cached type and the candle time changed since last send
                if not is_cached_type or sent_state.get(tf) != candle_open:
                    tfs_to_notify.add(tf)

                if is_cached_type:
                    # Check if data exists in Redis
                    cache_key = f"{CONF.CACHE_PREFIX}:scan:{tf}:{candle_open}"
                    cached_data = await self.cache.get_json(cache_key)
                    
                    if cached_data is not None:
                        # Update hot status in cached data
                        for item in cached_data: item['hot'] = item['symbol'] in hot_coins
                        tfs_from_cache.extend(cached_data)
                    else:
                        tfs_to_scan_fresh.append(tf)
                else:
                    tfs_to_scan_fresh.append(tf)

            # 5. Scan Execution
            results = list(tfs_from_cache)
            
            if tfs_to_scan_fresh:
                logger.info(f"Scanning fresh: {tfs_to_scan_fresh}")
                fresh_results = await self._scan_parallel(all_syms, tfs_to_scan_fresh, hot_coins)
                
                # Save cached types
                grouped_fresh = {}
                for r in fresh_results:
                    grouped_fresh.setdefault(r['timeframe'], []).append(r)
                
                for tf in tfs_to_scan_fresh:
                    if tf in CONF.CACHED_TFS:
                        data = grouped_fresh.get(tf, [])
                        key = f"{CONF.CACHE_PREFIX}:scan:{tf}:{current_opens[tf]}"
                        await self.cache.set_json(key, data, ttl=CONF.CACHE_TTL[tf])
                
                results.extend(fresh_results)

            # 6. Filtering & Reporting
            final_results = [r for r in results if r['timeframe'] in tfs_to_notify]
            
            if final_results:
                msgs = self._format_results(final_results)
                for msg in msgs:
                    await send_telegram(self.mde.session, msg)
                
                # Update state for sent cached TFs
                for tf in tfs_to_notify:
                    if tf in CONF.CACHED_TFS:
                        sent_state[tf] = current_opens[tf]
                await self.cache.save_sent_state(sent_state)
            else:
                logger.info("No new alerts to send.")

        except Exception as e:
            logger.exception("Fatal Error in Run")
            await send_telegram(self.mde.session, f"🚨 *Bot Crash*: {str(e)}")
        finally:
            await self.mde.close()
            await self.cache.close()
            logger.info(f"Run finished in {time.perf_counter() - start_ts:.2f}s")

    async def _scan_parallel(self, symbols: List[Tuple[str, bool]], timeframes: List[str], hot_coins: Set[str]) -> List[Dict]:
        """
        Massively parallel scanner.
        """
        results = []
        
        async def worker(sym, is_fut, tf):
            # Optimize: Relax candle limit for high TFs if needed, but config handles it
            closes, times = await self.mde.fetch_klines(sym, tf, is_fut)
            if closes is None: return None
            
            return TechnicalAnalysis.analyze(sym, tf, closes, times, sym in hot_coins, is_fut)

        # Create all tasks
        tasks = []
        for tf in timeframes:
            for sym, is_fut in symbols:
                tasks.append(worker(sym, is_fut, tf))
        
        # Execute with gather (concurrency controlled by mde.sem)
        # Chunking to avoid memory spikes if millions of tasks
        chunk_size = 1000
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i+chunk_size]
            res = await asyncio.gather(*chunk, return_exceptions=True)
            for r in res:
                if isinstance(r, dict): results.append(r)
                
        return results

    def _format_results(self, results: List[Dict]) -> List[str]:
        if not results: return []
        
        # Sort order
        tf_order = {k: v for k, v in CONF.TIMEFRAME_MAP.items()}
        
        grouped = {}
        for r in results: grouped.setdefault(r['timeframe'], []).append(r)
        
        sorted_tfs = sorted(grouped.keys(), key=lambda x: tf_order.get(x, 0), reverse=True)
        
        messages = []
        ts = datetime.now(timezone.utc).strftime('%H:%M UTC')
        
        for tf in sorted_tfs:
            items = grouped[tf]
            header = f"🔥 *BB Touches on {tf}* ({len(items)})"
            
            lines = [header]
            
            # Group by type
            by_type = {'UPPER': [], 'MIDDLE': [], 'LOWER': []}
            for i in items: by_type[i['touch_type']].append(i)
            
            for t_type in ['UPPER', 'MIDDLE', 'LOWER']:
                if not by_type[t_type]: continue
                
                lines.append(f"\n*{t_type}*:")
                # Sort by RSI strength (Upper: desc, Lower: asc)
                sorted_items = sorted(by_type[t_type], key=lambda x: x['rsi'], reverse=(t_type=='UPPER'))
                
                for item in sorted_items:
                    market = "F" if item['market_type'] == 'FUTURES' else "S"
                    hot_icon = "⚡" if item['hot'] else ""
                    arrow = ""
                    if t_type == 'MIDDLE':
                        arrow = "⬆️" if "below" in item.get('direction', '') else "⬇️"
                    
                    line = f"`{item['symbol']:<10}` [{market}] RSI:{item['rsi']:.1f} {hot_icon} {arrow}"
                    lines.append(line)
            
            lines.append(f"\n_Scan: {ts}_")
            messages.append("\n".join(lines))
            
        return messages

if __name__ == "__main__":
    # Graceful Exit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    bot = BB_Bot()

    def signal_handler():
        logger.info("Stopping...")
        pending = asyncio.all_tasks(loop)
        for t in pending: t.cancel()
    
    if sys.platform != 'win32':
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(bot.run())
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()
