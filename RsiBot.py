import aiohttp
import asyncio
import numpy as np
import talib
import logging
import os
import json
import itertools
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass
import redis.asyncio as aioredis
import uvloop

# Set the event loop to uvloop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# === CONFIG & CONSTANTS ===

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

CACHE_DIR = "bb_touch_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Binance API endpoints
BINANCE_FUTURES_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_SPOT_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
BINANCE_SPOT_KLINES = "https://api.binance.com/api/v3/klines"
PROXY_TEST_URL = "https://api.binance.com/api/v3/time"

# RSI and BB parameters
RSI_PERIOD = 14
BB_LENGTH = 34
BB_STDDEV = 2
CANDLE_LIMIT = 60
HIGH_TIMEFRAMES_RELAX_CANDLE_LIMIT = {'1d', '1w'}
MIN_CANDLES_FOR_TALIB = max(RSI_PERIOD, BB_LENGTH) + 2

# Touch thresholds
UPPER_TOUCH_THRESHOLD = 0.02
MIDDLE_TOUCH_THRESHOLD = 0.035
LOWER_TOUCH_THRESHOLD = 0.02

# Proxy sources
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"
]

TIMEFRAMES_TOGGLE = {
    '3m': False,
    '5m': False,
    '15m': True,
    '30m': True,
    '1h': True,
    '2h': True,
    '4h': True,
    '1d': True,
    '1w': True,
}

TIMEFRAME_MINUTES_MAP = {
    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
    '12h': 720, '1d': 1440, '1w': 10080
}

CACHE_TTL_SECONDS = {
    '4h': 4 * 3600 + 600,
    '1d': 24 * 3600 + 1800,
    '1w': 7 * 24 * 3600 + 3600,
}

CACHED_TFS = {'4h', '1d', '1w'}

# === DATA CLASSES ===

@dataclass
class CacheEntry:
    symbol: str
    timeframe: str
    rsi: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    touch_type: str
    timestamp: str
    hot: bool
    direction: Optional[str]
    market_type: str

# === Cache Manager using Redis ===

class CacheManager:
    """Manages caching of scan results and bot state using Redis."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis: Optional[aioredis.Redis] = None

    async def initialize(self) -> None:
        """Initializes the Redis connection."""
        if not self.redis:
            try:
                self.redis = await aioredis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
                await self.redis.ping()  # Test connection
                logging.info(f"Connected to Redis at {self.redis_url}")
            except Exception as e:
                logging.error(f"Failed to connect to Redis at {self.redis_url}: {e}")

    async def close(self) -> None:
        """Closes the Redis connection."""
        if self.redis:
            await self.redis.close()
            self.redis = None
            logging.info("Redis connection closed")

    def _scan_key(self, timeframe: str, candle_open_iso: str) -> str:
        """Generates a Redis key for scan results."""
        return f"bb_touch:scan:{timeframe}:{candle_open_iso}"

    def _sent_state_key(self) -> str:
        """Generates a Redis key for the sent state."""
        return "bb_touch:sent_state"

    async def get_scan_results(self, timeframe: str, candle_open_iso: str) -> Optional[List[CacheEntry]]:
        """Retrieves scan results from cache."""
        if not self.redis:
            return None
        key = self._scan_key(timeframe, candle_open_iso)
        data = await self.redis.get(key)
        if data:
            try:
                return [CacheEntry(**entry) for entry in json.loads(data)]
            except json.JSONDecodeError:
                logging.warning(f"Failed to decode cache data for {key}.")
        return None

    async def set_scan_results(self, timeframe: str, candle_open_iso: str, results: List[CacheEntry], ttl_seconds: int) -> None:
        """Stores scan results in cache."""
        if not self.redis:
            return
        key = self._scan_key(timeframe, candle_open_iso)
        await self.redis.set(key, json.dumps([entry.__dict__ for entry in results]), ex=ttl_seconds)
        logging.info(f"Cache set for {key} with TTL {ttl_seconds}s")

    async def get_sent_state(self) -> Dict[str, str]:
        """Retrieves the last sent state from cache."""
        if not self.redis:
            return {}
        key = self._sent_state_key()
        data = await self.redis.get(key)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                logging.warning("Failed to decode sent state.")
        return {}

    async def set_sent_state(self, state: Dict[str, str]) -> None:
        """Stores the current sent state in cache."""
        if not self.redis:
            return
        key = self._sent_state_key()
        await self.redis.set(key, json.dumps(state))
        logging.info("Sent state updated")

# === UTILITY FUNCTIONS ===

def get_active_timeframes() -> List[str]:
    """Returns a list of enabled timeframes."""
    return [tf for tf, enabled in TIMEFRAMES_TOGGLE.items() if enabled]

def get_latest_candle_open(timeframe: str, now: Optional[datetime] = None) -> datetime:
    """Calculates the opening time of the current candle for a given timeframe."""
    if now is None:
        now = datetime.utcnow()
    interval_minutes = TIMEFRAME_MINUTES_MAP.get(timeframe)
    if interval_minutes is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    total_minutes = now.hour * 60 + now.minute
    intervals_passed = total_minutes // interval_minutes
    candle_open_minutes = intervals_passed * interval_minutes
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=candle_open_minutes)

def normalize_proxy(proxy: str) -> str:
    """Normalizes proxy format."""
    return proxy.strip() if "://" in proxy else f"http://{proxy.strip()}"

# === PROXY MANAGEMENT ===

class ProxyManager:
    def __init__(self, sources: List[str] = PROXY_SOURCES, max_pool_size: int = 20, min_working: int = 5):
        self.sources = sources
        self.max_pool_size = max_pool_size
        self.min_working = min_working
        self.proxies: List[str] = []
        self.blacklisted = set()

    async def fetch_proxies(self) -> List[str]:
        """Fetches proxies from the defined sources."""
        proxies = []
        for source in self.sources:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(source, timeout=10) as resp:
                        resp.raise_for_status()
                        proxies.extend([
                            normalize_proxy(line) for line in (await resp.text()).splitlines() if line.strip()
                        ])
            except Exception as e:
                logging.error(f"Error fetching proxies from {source}: {e}")
        return proxies

    async def update_proxies(self) -> None:
        """Updates the proxy pool."""
        fetched_proxies = await self.fetch_proxies()
        self.proxies = [p for p in fetched_proxies if p not in self.blacklisted]
        if len(self.proxies) < self.min_working:
            logging.warning("Not enough working proxies, consider clearing blacklist.")
            self.blacklisted.clear()

    async def get_next_proxy(self) -> Optional[str]:
        """Gets the next available proxy."""
        if not self.proxies:
            await self.update_proxies()
        return self.proxies[0] if self.proxies else None

# === ASYNC REQUESTS & SCANNING ===

async def make_request_async(session: aiohttp.ClientSession, url: str, params: Optional[Dict[str, Any]] = None, proxy_manager: Optional[ProxyManager] = None, max_attempts: int = 4) -> Any:
    """Makes a request with proxy handling."""
    attempt = 0
    while attempt < max_attempts:
        proxy = await proxy_manager.get_next_proxy() if proxy_manager else None
        try:
            async with session.get(url, params=params, proxy=proxy, timeout=15) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            logging.error(f"Request failed: {e}")
            attempt += 1
            if attempt >= max_attempts:
                raise RuntimeError(f"Request to {url} failed after {max_attempts} attempts")

async def get_all_tradable_usdt_symbols_async(session: aiohttp.ClientSession, proxy_manager: ProxyManager) -> Tuple[List[str], List[str]]:
    """Fetches tradable USDT symbols from both Futures and Spot, prioritizing Futures."""
    futures_symbols_set: Set[str] = set()
    spot_symbols_set: Set[str] = set()

    # Fetch Futures symbols
    try:
        futures_data = await make_request_async(session, BINANCE_FUTURES_EXCHANGE_INFO, proxy_manager=proxy_manager)
        futures_symbols_set = {
            s['symbol'] for s in futures_data.get('symbols', [])
            if s.get('contractType') == 'PERPETUAL' and s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'
        }
        logging.info(f"Fetched {len(futures_symbols_set)} Futures USDT perpetual symbols.")
    except Exception as e:
        logging.error(f"Error fetching Futures exchange info: {e}")

    # Fetch Spot symbols
    try:
        spot_data = await make_request_async(session, BINANCE_SPOT_EXCHANGE_INFO, proxy_manager=proxy_manager)
        spot_symbols_set = {
            s['symbol'] for s in spot_data.get('symbols', [])
            if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'
        }
        logging.info(f"Fetched {len(spot_symbols_set)} Spot USDT symbols.")
    except Exception as e:
        logging.error(f"Error fetching Spot exchange info: {e}")

    # Filter Spot symbols to exclude those already on Futures
    spot_symbols_filtered = sorted(spot_symbols_set - futures_symbols_set)
    futures_symbols_list = sorted(futures_symbols_set)

    logging.info(f"Filtered Spot symbols (not on Futures): {len(spot_symbols_filtered)}")

    return futures_symbols_list, spot_symbols_filtered

async def fetch_klines_async(session: aiohttp.ClientSession, symbol: str, interval: str, proxy_manager: ProxyManager, limit: int = CANDLE_LIMIT, is_futures: bool = True) -> Tuple[Optional[List[float]], Optional[List[int]]]:
    """Fetches klines for a given symbol and interval from either Futures or Spot API."""
    endpoint = BINANCE_FUTURES_KLINES if is_futures else BINANCE_SPOT_KLINES
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        data = await make_request_async(session, endpoint, params=params, proxy_manager=proxy_manager)
        closes = [float(k[4]) for k in data]
        timestamps = [int(k[0]) for k in data]
        return closes, timestamps
    except Exception as e:
        logging.error(f"Error fetching klines for {symbol} {interval} (Futures: {is_futures}): {e}")
        return None, None

async def get_daily_change_percent_async(session: aiohttp.ClientSession, symbol: str, proxy_manager: ProxyManager, is_futures: bool = True) -> Optional[float]:
    """Fetches the daily price change percentage for a symbol."""
    endpoint = BINANCE_FUTURES_KLINES if is_futures else BINANCE_SPOT_KLINES
    params = {'symbol': symbol, 'interval': '1d', 'limit': 1}
    try:
        data = await make_request_async(session, endpoint, params=params, proxy_manager=proxy_manager)
        if not data or len(data) < 1:
            return None
        daily_open = float(data[-1][1])
        current_price = float(data[-1][4])
        if daily_open == 0:
            return None
        return (current_price - daily_open) / daily_open * 100
    except Exception as e:
        logging.warning(f"Could not fetch daily change for {symbol} (Futures: {is_futures}): {e}")
        return None

async def calculate_volatility_rankings(session: aiohttp.ClientSession, symbols: List[Tuple[str, bool]], proxy_manager: ProxyManager, top_n: int = 60) -> Set[str]:
    """Calculate 24h volatility for all symbols and return top N most volatile."""
    volatility_scores = {}

    async def get_volatility(symbol: str, is_futures: bool) -> None:
        try:
            data = await fetch_klines_async(session, symbol, '1h', proxy_manager, limit=25, is_futures=is_futures)
            if data and len(data) >= 25:
                closes = [float(candle[4]) for candle in data]
                returns = [(closes[i] - closes[i - 1]) / closes[i - 1] * 100 for i in range(1, len(closes)) if closes[i - 1] != 0]

                if len(returns) >= 24:
                    volatility_scores[symbol] = np.std(returns)  # Standard deviation

        except Exception as e:
            logging.warning(f"Failed to calculate volatility for {symbol}: {e}")

    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        await asyncio.gather(*(get_volatility(symbol, is_futures) for symbol, is_futures in batch))
        await asyncio.sleep(0.5)  # Small delay between batches

    top_volatile = {symbol for symbol, vol in sorted(volatility_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]}
    logging.info(f"Top {top_n} most volatile coins: {list(top_volatile)[:10]}...")  # Log first 10
    return top_volatile

def calculate_rsi_bb(closes: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates RSI and Bollinger Bands for a given list of close prices."""
    closes_np = np.array(closes, dtype=float)
    rsi = talib.RSI(closes_np, timeperiod=RSI_PERIOD)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(rsi, timeperiod=BB_LENGTH, nbdevup=BB_STDDEV, nbdevdn=BB_STDDEV, matype=0)
    return rsi, bb_upper, bb_middle, bb_lower

# === SCANNING FUNCTIONS ===

async def load_cache_async(cache_manager: CacheManager, timeframe: str) -> Optional[List[CacheEntry]]:
    """Loads scan results for a timeframe from cache."""
    candle_open = get_latest_candle_open(timeframe).isoformat()
    return await cache_manager.get_scan_results(timeframe, candle_open)

async def save_cache_async(cache_manager: CacheManager, timeframe: str, results: List[CacheEntry]) -> None:
    """Saves scan results for a timeframe to cache."""
    candle_open = get_latest_candle_open(timeframe).isoformat()
    ttl = CACHE_TTL_SECONDS.get(timeframe, 3600)
    await cache_manager.set_scan_results(timeframe, candle_open, results, ttl)

async def load_sent_state_async(cache_manager: CacheManager) -> Dict[str, str]:
    """Loads the bot's sent state from cache."""
    return await cache_manager.get_sent_state()

async def save_sent_state_async(cache_manager: CacheManager, state: Dict[str, str]) -> None:
    """Saves the bot's sent state to cache."""
    await cache_manager.set_sent_state(state)

async def scan_symbol_async(session: aiohttp.ClientSession, symbol: str, timeframes: List[str], proxy_manager: ProxyManager, is_futures: bool, hot_coins: Set[str] = None) -> List[CacheEntry]:
    """Scans a single symbol across multiple timeframes for BB touches."""
    results = []
    is_hot = (hot_coins is not None and symbol in hot_coins)

    for timeframe in timeframes:
        closes, timestamps = await fetch_klines_async(session, symbol, timeframe, proxy_manager, is_futures=is_futures)

        if closes is None or len(closes) < MIN_CANDLES_FOR_TALIB:
            logging.warning(f"Not enough data for {symbol} {timeframe}. Skipping.")
            continue

        rsi, bb_upper, bb_middle, bb_lower = calculate_rsi_bb(closes)

        idx = -2
        if np.isnan(rsi[idx]) or np.isnan(bb_upper[idx]) or np.isnan(bb_lower[idx]) or np.isnan(bb_middle[idx]):
            logging.warning(f"NaN values for {symbol} {timeframe}, skipping.")
            continue

        rsi_val, bb_upper_val, bb_middle_val, bb_lower_val = rsi[idx], bb_upper[idx], bb_middle[idx], bb_lower[idx]

        upper_touch = rsi_val >= bb_upper_val * (1 - UPPER_TOUCH_THRESHOLD)
        lower_touch = rsi_val <= bb_lower_val * (1 + LOWER_TOUCH_THRESHOLD)
        middle_touch = False
        direction = None

        if MIDDLE_TOUCH_THRESHOLD and not upper_touch and not lower_touch and abs(rsi_val - bb_middle_val) <= bb_middle_val * MIDDLE_TOUCH_THRESHOLD:
            middle_touch = True
            prev_rsi = rsi[idx - 1]
            prev_side = prev_rsi - bb_middle[idx - 1]
            curr_side = rsi_val - bb_middle_val
            direction = "from above" if prev_side > 0 and curr_side <= 0 else "from below" if prev_side < 0 and curr_side >= 0 else "from above" if curr_side > 0 else "from below"

        if upper_touch or lower_touch or middle_touch:
            touch_type = "UPPER" if upper_touch else "LOWER" if lower_touch else "MIDDLE"
            timestamp = datetime.utcfromtimestamp(timestamps[idx] / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')
            results.append(CacheEntry(
                symbol=symbol,
                timeframe=timeframe,
                rsi=rsi_val,
                bb_upper=bb_upper_val,
                bb_middle=bb_middle_val,
                bb_lower=bb_lower_val,
                touch_type=touch_type,
                timestamp=timestamp,
                hot=is_hot,
                direction=direction,
                market_type='FUTURES' if is_futures else 'SPOT'
            ))

    return results

async def scan_for_bb_touches_async(proxy_manager: ProxyManager, cache_manager: CacheManager) -> Tuple[List[CacheEntry], Set[str]]:
    """Orchestrates the scanning process for BB touches across all symbols and timeframes."""
    results = []
    active_timeframes = get_active_timeframes()
    cached_timeframes = [tf for tf in CACHED_TFS if tf in active_timeframes]
    uncached_timeframes = [tf for tf in active_timeframes if tf not in cached_timeframes]
    cached_results = {}
    cached_timeframes_used = set()

    async with aiohttp.ClientSession() as session:
        # STEP 1: Get all symbols
        futures_symbols, spot_symbols = await get_all_tradable_usdt_symbols_async(session, proxy_manager)
        all_symbols_with_type = [(s, True) for s in futures_symbols] + [(s, False) for s in spot_symbols]

        # STEP 2: Calculate volatility rankings for ALL symbols
        logging.info(f"[VOLATILITY] Calculating 24h volatility for {len(all_symbols_with_type)} symbols...")
        hot_coins = await calculate_volatility_rankings(session, all_symbols_with_type, proxy_manager, top_n=60)

        # STEP 3: Load cached results for relevant timeframes
        for timeframe in cached_timeframes:
            cached_data = await load_cache_async(cache_manager, timeframe)
            if cached_data is not None:
                cached_results[timeframe] = cached_data
                cached_timeframes_used.add(timeframe)
                logging.info(f"[CACHE] Loaded {len(cached_data)} results for {timeframe} from cache.")

        # STEP 4: Process cached timeframes that need a fresh scan
        for timeframe in cached_timeframes:
            if timeframe not in cached_results:
                logging.info(f"[SCAN] Performing fresh scan for cached timeframe: {timeframe}")
                for i in range(0, len(all_symbols_with_type), 70):
                    batch = all_symbols_with_type[i:i + 70]
                    batch_results = await asyncio.gather(*(scan_symbol_async(session, symbol, [timeframe], proxy_manager, is_futures, hot_coins) for symbol, is_futures in batch))
                    for res in batch_results:
                        results.extend(res)

                await save_cache_async(cache_manager, timeframe, results)
                cached_results[timeframe] = results
                logging.info(f"[CACHE] Saved fresh scan results for {timeframe} ({len(results)} items).")

        # STEP 5: Update cached results with fresh volatility data
        for timeframe in cached_timeframes:
            if timeframe in cached_results:
                for item in cached_results[timeframe]:
                    item.hot = item.symbol in hot_coins

        # STEP 6: Process uncached timeframes (always fresh scan)
        if uncached_timeframes:
            logging.info(f"[SCAN] Scanning uncached timeframes: {', '.join(uncached_timeframes)}")
            for i in range(0, len(all_symbols_with_type), 70):
                batch = all_symbols_with_type[i:i + 70]
                uncached_scan_results = await asyncio.gather(*(scan_symbol_async(session, symbol, uncached_timeframes, proxy_manager, is_futures, hot_coins) for symbol, is_futures in batch))
                results.extend(uncached_scan_results)

        logging.info(f"[RESULTS] Total BB touches found: {len(results)}")
        return results, cached_timeframes_used

# === FORMATTING & TELEGRAM SENDING ===

def format_results_by_timeframe(results: List[CacheEntry], cached_timeframes_used: Optional[Set[str]] = None) -> List[str]:
    """Formats the scan results into Telegram-ready messages, grouped by timeframe."""
    if not results:
        return ["*No BB touches detected at this time.*"]

    timeframe_order = {'1w': 7, '1d': 6, '4h': 5, '2h': 4, '1h': 3, '30m': 2, '15m': 1, '5m': 0, '3m': -1}
    grouped = {}

    for r in results:
        grouped.setdefault(r.timeframe, []).append(r)

    sorted_timeframes = sorted(grouped.keys(), key=lambda tf: timeframe_order.get(tf, -2), reverse=True)
    messages = []

    for timeframe in sorted_timeframes:
        items = grouped[timeframe]
        header = f"* BB Touches on {timeframe} Timeframe ({len(items)} symbols)*"
        if cached_timeframes_used and timeframe in cached_timeframes_used:
            header += " _(from cache)_"
        header += "\n"

        lines = []
        upper_touches = [i for i in items if i.touch_type == 'UPPER']
        middle_touches = [i for i in items if i.touch_type == 'MIDDLE']
        lower_touches = [i for i in items if i.touch_type == 'LOWER']

        def format_line(item: CacheEntry) -> str:
            market_tag = "[F]" if item.market_type == 'FUTURES' else "[S]"
            base = f"*{item.symbol}* {market_tag} - RSI: {item.rsi:.2f}"
            if item.touch_type == 'MIDDLE':
                arrow = "" if item.direction == "from above" else ""
                base += f" ({arrow})"
            return " " + base

        if upper_touches:
            lines.append("* UPPER BB Touches:*")
            lines.extend(format_line(item) for item in sorted(upper_touches, key=lambda x: x.symbol))

        if middle_touches:
            if upper_touches:
                lines.append("")
            lines.append("* MIDDLE BB Touches:*")
            lines.extend(format_line(item) for item in sorted(middle_touches, key=lambda x: x.symbol))

        if lower_touches:
            if upper_touches or middle_touches:
                lines.append("")
            lines.append("* LOWER BB Touches:*")
            lines.extend(format_line(item) for item in sorted(lower_touches, key=lambda x: x.symbol))

        messages.append(header + "\n" + "\n".join(lines))

    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    return [msg + f"\n\n_Report generated at {timestamp}_" for msg in messages]

def split_message(text: str, max_length: int = 4000) -> List[str]:
    """Splits a long message into chunks suitable for Telegram."""
    lines = text.split('\n')
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += "\n" + line if current_chunk else line
    
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> bool:
    """Sends a message to Telegram."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': True
    }
    try:
        for attempt in range(3):
            response = requests.post(url, data=payload, timeout=10)
            if response.status_code == 200:
                return True
            logging.warning(f"Telegram API returned {response.status_code}: {response.text}. Retrying...")
            time.sleep(1)
        logging.error(f"Telegram alert failed after multiple attempts: {response.text}")
        return False
    except Exception as e:
        logging.error(f"Exception sending Telegram alert: {e}")
        return False

# === MAIN ASYNC ENTRY POINT ===

async def main_async() -> None:
    """Main asynchronous function to run the bot."""
    start_time = time.time()
    BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

    if not BOT_TOKEN or not CHAT_ID:
        logging.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables are not set. Exiting.")
        return

    cache_manager = CacheManager(redis_url=REDIS_URL)
    await cache_manager.initialize()

    proxy_manager = ProxyManager(max_pool_size=30, min_working=7)
    await proxy_manager.update_proxies()

    try:
        logging.info("Starting BB touch scanner bot")

        # Load last sent state
        sent_state = await load_sent_state_async(cache_manager)

        active_timeframes = get_active_timeframes()
        now_opens = {tf: get_latest_candle_open(tf).isoformat() for tf in CACHED_TFS if tf in active_timeframes}
        to_send_cached_tfs = {tf for tf, iso in now_opens.items() if sent_state.get(tf) != iso}

        logging.info(f"Always-fresh TFs: {sorted(set(active_timeframes) - CACHED_TFS)}")
        logging.info(f"Cached TFs sending only on new candle open: {sorted(to_send_cached_tfs)}")

        # Scan for BB touches
        results, cached_timeframes_used = await scan_for_bb_touches_async(proxy_manager, cache_manager)

        # Format messages
        messages = format_results_by_timeframe(results, cached_timeframes_used=cached_timeframes_used)

        # Filter messages to send based on the logic:
        messages_to_dispatch = [
            msg for msg in messages if any(
                re.search(rf"BB Touches on {tf} Timeframe", msg) for tf in (set(active_timeframes) - CACHED_TFS) | to_send_cached_tfs
            )
        ]

        # Dispatch alerts
        if not messages_to_dispatch:
            logging.info("No BB-touch alerts to send this run.")
        else:
            for msg in messages_to_dispatch:
                logging.info(f"Sending message:\n{msg}")
                for chunk in split_message(msg):
                    send_telegram_alert(BOT_TOKEN, CHAT_ID, chunk)

        # Update sent state for cached TFs that were just sent
        for tf in to_send_cached_tfs:
            sent_state[tf] = now_opens[tf]
        await save_sent_state_async(cache_manager, sent_state)

        logging.info(f"Bot run completed in {time.time() - start_time:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"Fatal error after {elapsed:.1f}s: {e}", exc_info=True)
        if BOT_TOKEN and CHAT_ID:
            error_msg = f"* Scanner Error*\n`{e}` after `{elapsed:.1f}s`"
            send_telegram_alert(BOT_TOKEN, CHAT_ID, error_msg)

    finally:
        await proxy_manager.update_proxies()  # Ensure proxy pool is updated
        await cache_manager.close()
        logging.info("Bot resources cleaned up.")

if __name__ == "__main__":
    asyncio.run(main_async())
