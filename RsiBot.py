import aiohttp
import asyncio
import requests
import pandas as pd
import numpy as np
import talib
import logging
import time
import os
import threading
import random
from datetime import datetime, timedelta
import json
import itertools
import glob
from typing import List

# === CONFIG & CONSTANTS ===

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

CACHE_DIR = "bb_touch_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

BINANCE_FUTURES_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"
PROXY_TEST_URL = "https://api.binance.com/api/v3/time"

CANDLE_LIMIT = 60

UPPER_TOUCH_THRESHOLD = 0.02  # 2%
MIDDLE_TOUCH_THRESHOLD = 0.015  # 1.5%
LOWER_TOUCH_THRESHOLD = 0.02  # 2%

RSI_PERIOD = 14
BB_LENGTH = 34
BB_STDDEV = 2

PROXY_SOURCES = [
    "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"
]

TIMEFRAMES_TOGGLE = {
    '3m': False,
    '5m': True,
    '15m': True,
    '30m': True,
    '1h': True,
    '2h': True,
    '4h': True,
    '1d': True,
    '1w': True,
}

MIDDLE_BAND_TOGGLE = {
    '3m': False,
    '5m': False,
    '15m': False,
    '30m': False,
    '1h': False,
    '2h': False,
    '4h': True,
    '1d': True,
    '1w': True,
}

TIMEFRAME_MINUTES_MAP = {
    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
    '12h': 720, '1d': 1440, '1w': 10080
}

# === CACHE UTILITIES ===

def get_active_timeframes():
    return [tf for tf, enabled in TIMEFRAMES_TOGGLE.items() if enabled]

def get_latest_candle_open(timeframe: str, now=None):
    if now is None:
        now = datetime.utcnow()
    if timeframe not in TIMEFRAME_MINUTES_MAP:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    interval_minutes = TIMEFRAME_MINUTES_MAP[timeframe]
    total_minutes = now.hour * 60 + now.minute
    intervals_passed = total_minutes // interval_minutes
    candle_open_minutes = intervals_passed * interval_minutes
    candle_open = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=candle_open_minutes)
    return candle_open

def get_cache_file_name(timeframe):
    candle_open = get_latest_candle_open(timeframe)
    filename = f"bb_touch_cache_{timeframe}_{candle_open.strftime('%Y%m%dT%H%M')}.json"
    return os.path.join(CACHE_DIR, filename)

def load_cache(timeframe):
    cache_file = get_cache_file_name(timeframe)
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        cached_candle_open = datetime.fromisoformat(data.get('candle_open'))
        latest_candle_open = get_latest_candle_open(timeframe)
        if cached_candle_open == latest_candle_open:
            return data.get('results')
        else:
            logging.info(f"Cache outdated for {timeframe}, cache candle open: {cached_candle_open}, latest: {latest_candle_open}")
    except Exception as e:
        logging.warning(f"Failed to load {timeframe} cache: {e}")
    return None

def save_cache(timeframe, results):
    cache_file = get_cache_file_name(timeframe)
    candle_open = get_latest_candle_open(timeframe)
    data = {
        'candle_open': candle_open.isoformat(),
        'results': results
    }
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        logging.info(f"Cache saved successfully for {timeframe} at {cache_file}")
    except Exception as e:
        logging.warning(f"Failed to save {timeframe} cache: {e}")

def cleanup_old_caches(max_age_days=7):
    now = time.time()
    max_age_seconds = max_age_days * 86400
    pattern = os.path.join(CACHE_DIR, "bb_touch_cache_*.json")
    files = glob.glob(pattern)
    deleted_files = 0
    for file_path in files:
        try:
            file_age = now - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                os.remove(file_path)
                deleted_files += 1
        except Exception as e:
            logging.warning(f"Failed to delete old cache file {file_path}: {e}")
    if deleted_files > 0:
        logging.info(f"Cleaned up {deleted_files} old cache files older than {max_age_days} days.")

# === ASYNC PROXY POOL ===

async def fetch_proxies_from_url_async(url: str, default_scheme: str = "http") -> List[str]:
    proxies = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                text = await resp.text()
        for line in text.splitlines():
            p = line.strip()
            if not p:
                continue
            if "://" not in p:
                p = f"{default_scheme}://{p}"
            proxies.append(p)
        logging.info(f"Fetched {len(proxies)} proxies from {url}")
    except Exception as e:
        logging.error(f"Error fetching proxies from {url}: {e}")
    return proxies

async def test_proxy_async(session: aiohttp.ClientSession, proxy: str) -> bool:
    try:
        async with session.get(PROXY_TEST_URL, proxy=proxy, timeout=8) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False

async def test_proxy_speed_async(session: aiohttp.ClientSession, proxy: str) -> float:
    start = time.time()
    try:
        async with session.get(PROXY_TEST_URL, proxy=proxy, timeout=10) as resp:
            resp.raise_for_status()
            return time.time() - start
    except Exception:
        return float("inf")

async def test_proxies_concurrently_async(
    proxies: List[str],
    max_workers: int = 50,
    max_working: int = 25
) -> List[str]:
    working = []
    sem = asyncio.Semaphore(max_workers)
    async with aiohttp.ClientSession() as session:
        async def sem_test(p):
            async with sem:
                ok = await test_proxy_async(session, p)
                return p if ok else None

        tasks = [asyncio.create_task(sem_test(p)) for p in proxies]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                working.append(result)
                if len(working) >= max_working:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    break
    logging.info(f"test_proxies_concurrently_async → {len(working)}/{len(proxies)} worked")
    return working

async def rank_proxies_by_speed_async(
    proxies: List[str],
    max_workers: int = 20
) -> List[tuple]:
    sem = asyncio.Semaphore(max_workers)
    async with aiohttp.ClientSession() as session:
        async def sem_speed(p):
            async with sem:
                spd = await test_proxy_speed_async(session, p)
                return (p, spd)

        tasks = [asyncio.create_task(sem_speed(p)) for p in proxies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    valid = [r for r in results if isinstance(r, tuple)]
    valid.sort(key=lambda x: x[1])
    logging.info(f"rank_proxies_by_speed_async → ranked {len(valid)} proxies")
    return valid

class AsyncProxyPool:
    def __init__(
        self,
        sources: List[str] = PROXY_SOURCES,
        max_pool_size: int = 25,
        min_working: int = 5,
        check_interval: int = 600,
        max_failures: int = 3
    ):
        self.sources = sources
        self.max_pool_size = max_pool_size
        self.min_working = min_working
        self.check_interval = check_interval
        self.max_failures = max_failures

        self.proxies: List[str] = []
        self.failures = {}          # proxy -> count
        self.failed = set()         # proxies removed
        self.cycle = None           # itertools.cycle

        self._stop = False
        self._tasks = []

    async def initialize(self):
        await self.populate_to_max()
        self._tasks.append(asyncio.create_task(self._health_check_loop()))
        self._tasks.append(asyncio.create_task(self._fastest_proxy_loop()))

    async def _health_check_loop(self):
        while not self._stop:
            await asyncio.sleep(self.check_interval)
            await self.check_proxies()

    async def _fastest_proxy_loop(self):
        while not self._stop:
            await asyncio.sleep(3600)
            await self.update_fastest_proxy()

    async def populate_to_max(self):
        needed = self.max_pool_size - len(self.proxies)
        if needed <= 0:
            return
        new_list = []
        for src in self.sources:
            fetched = await fetch_proxies_from_url_async(src)
            new_list.extend(fetched)
            if len(new_list) >= needed * 2:
                break
        working = await test_proxies_concurrently_async(new_list, max_workers=50, max_working=needed)
        self.proxies.extend(working)
        self._rebuild_cycle()
        logging.info(f"Pool populated: {len(self.proxies)}/{self.max_pool_size}")

    def _rebuild_cycle(self):
        self.cycle = itertools.cycle(self.proxies)

    def get_next_proxy(self) -> str:
        if not self.proxies:
            return None
        for _ in range(len(self.proxies)):
            p = next(self.cycle)
            if p not in self.failed:
                return p
        return None

    def mark_proxy_failure(self, proxy: str):
        c = self.failures.get(proxy, 0) + 1
        self.failures[proxy] = c
        logging.warning(f"Proxy {proxy} failure {c}/{self.max_failures}")
        if c >= self.max_failures:
            self.failed.add(proxy)
            if proxy in self.proxies:
                self.proxies.remove(proxy)
                logging.warning(f"Removed {proxy} from pool")
            self._rebuild_cycle()

    def reset_proxy_failures(self, proxy: str):
        if proxy in self.failures:
            self.failures[proxy] = 0
        if proxy in self.failed:
            self.failed.remove(proxy)
            if proxy not in self.proxies:
                self.proxies.append(proxy)
                self._rebuild_cycle()

    async def check_proxies(self):
        if not self.proxies:
            return
        async with aiohttp.ClientSession() as session:
            sem = asyncio.Semaphore(50)
            async def sem_test(p):
                async with sem:
                    ok = await test_proxy_async(session, p)
                    return (p, ok)
            tasks = [asyncio.create_task(sem_test(p)) for p in self.proxies]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        alive = [p for p, ok in results if ok]
        removed = len(self.proxies) - len(alive)
        self.proxies = alive
        self.failed.clear()
        self.failures = {}
        self._rebuild_cycle()
        logging.info(f"Health check removed {removed} proxies; pool now {len(self.proxies)}")
        if len(self.proxies) < self.min_working:
            await self.populate_to_max()

    async def update_fastest_proxy(self):
        ranked = await rank_proxies_by_speed_async(self.proxies, max_workers=20)
        if ranked:
            fastest = ranked[0][0]
            logging.info(f"Fastest proxy is now {fastest}")
        else:
            logging.warning("Could not determine fastest proxy")

    async def shutdown(self):
        self._stop = True
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

# === ASYNC REQUESTS & SCANNING ===

async def make_request_async(session, url, params=None, proxy_manager=None, max_attempts=4):
    if proxy_manager is None:
        raise ValueError("proxy_manager argument is required")
    attempt = 0
    while attempt < max_attempts:
        proxy = proxy_manager.get_next_proxy()
        if proxy is None:
            logging.warning("No working proxies available, waiting before retry...")
            await asyncio.sleep(5)
            continue

        proxy_url = proxy if proxy.startswith(("http://", "https://")) else f"http://{proxy}"

        endpoint = url.split('/')[-1] if '/' in url else url
        logging.debug(f"Request to {endpoint}: using proxy {proxy} (attempt {attempt+1}/{max_attempts})")

        try:
            async with session.get(url, params=params, proxy=proxy_url, timeout=15, ssl=True) as resp:
                if resp.status == 451:
                    logging.warning(f"Proxy {proxy} returned HTTP 451, blacklisting and retrying with different proxy")
                    proxy_manager.mark_proxy_failure(proxy)
                    continue  # retry immediately without incrementing attempt

                resp.raise_for_status()
                proxy_manager.reset_proxy_failures(proxy)
                return await resp.json()

        except Exception as e:
            logging.error(f"Request failed with proxy {proxy}: {e}")
            proxy_manager.mark_proxy_failure(proxy)
            attempt += 1
            wait_time = 2 * attempt
            logging.info(f"Retrying in {wait_time} seconds (attempt {attempt}/{max_attempts})...")
            await asyncio.sleep(wait_time)

    raise RuntimeError(f"Request failed after {max_attempts} attempts")

async def get_perpetual_usdt_symbols_async(proxy_manager, max_attempts=5):
    for attempt in range(1, max_attempts + 1):
        try:
            async with aiohttp.ClientSession() as session:
                data = await make_request_async(session, BINANCE_FUTURES_EXCHANGE_INFO, proxy_manager=proxy_manager)
                symbols = [
                    s['symbol'] for s in data.get('symbols', [])
                    if s.get('contractType') == 'PERPETUAL' and s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'
                ]
                if len(symbols) >= 10:
                    return symbols
                logging.warning(f"Too few symbols ({len(symbols)}) via proxy, retrying...")
        except Exception as e:
            logging.warning(f"Proxy attempt {attempt} failed: {e}")
        await asyncio.sleep(3)
    raise RuntimeError("Failed to fetch USDT perpetual symbols via proxies after multiple attempts")

async def fetch_klines_async(session, symbol, interval, proxy_manager, limit=CANDLE_LIMIT):
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        data = await make_request_async(session, BINANCE_FUTURES_KLINES, params=params, proxy_manager=proxy_manager)
        closes = [float(k[4]) for k in data]
        timestamps = [k[0] for k in data]
        return closes, timestamps
    except Exception as e:
        logging.error(f"Error fetching klines for {symbol} {interval}: {e}")
        return None, None

async def get_daily_change_percent_async(session, symbol, proxy_manager):
    params = {'symbol': symbol, 'interval': '1d', 'limit': 1}
    try:
        data = await make_request_async(session, BINANCE_FUTURES_KLINES, params=params, proxy_manager=proxy_manager)
        if not data or len(data) < 1:
            return None
        k = data[-1]
        daily_open = float(k[1])
        current_price = float(k[4])
        if daily_open == 0:
            return None
        return (current_price - daily_open) / daily_open * 100
    except Exception as e:
        logging.warning(f"Could not fetch daily change for {symbol}: {e}")
        return None

def calculate_rsi_bb(closes):
    closes_np = np.array(closes)
    rsi = talib.RSI(closes_np, timeperiod=RSI_PERIOD)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        rsi,
        timeperiod=BB_LENGTH,
        nbdevup=BB_STDDEV,
        nbdevdn=BB_STDDEV,
        matype=0
    )
    return rsi, bb_upper, bb_middle, bb_lower

# === SCANNING ===

async def scan_symbol_async(symbol, timeframes, proxy_manager):
    results = []
    async with aiohttp.ClientSession() as session:
        daily_change = await get_daily_change_percent_async(session, symbol, proxy_manager)

        for timeframe in timeframes:
            closes, timestamps = await fetch_klines_async(session, symbol, timeframe, proxy_manager)
            if closes is None or len(closes) < CANDLE_LIMIT:
                logging.warning(f"Not enough data for {symbol} {timeframe}. Skipping.")
                continue
            idx = -2
            if idx < -len(closes):
                logging.warning(f"Not enough candles for {symbol} {timeframe} to skip open candle. Skipping.")
                continue

            rsi, bb_upper, bb_middle, bb_lower = calculate_rsi_bb(closes)
            if np.isnan(rsi[idx]) or np.isnan(bb_upper[idx]) or np.isnan(bb_lower[idx]) or np.isnan(bb_middle[idx]):
                logging.warning(f"NaN values for {symbol} {timeframe}, skipping.")
                continue

            rsi_val = rsi[idx]
            bb_upper_val = bb_upper[idx]
            bb_middle_val = bb_middle[idx]
            bb_lower_val = bb_lower[idx]

            upper_touch = rsi_val >= bb_upper_val * (1 - UPPER_TOUCH_THRESHOLD)
            lower_touch = rsi_val <= bb_lower_val * (1 + LOWER_TOUCH_THRESHOLD)
            middle_touch = False

            if MIDDLE_BAND_TOGGLE.get(timeframe, False):
                if not upper_touch and not lower_touch:
                    if abs(rsi_val - bb_middle_val) <= bb_middle_val * MIDDLE_TOUCH_THRESHOLD:
                        middle_touch = True

            if upper_touch or lower_touch or middle_touch:
                if upper_touch:
                    touch_type = "UPPER"
                elif lower_touch:
                    touch_type = "LOWER"
                else:
                    touch_type = "MIDDLE"

                timestamp = datetime.utcfromtimestamp(timestamps[idx] / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')
                hot = False
                if daily_change is not None and daily_change > 5:
                    hot = True

                item = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'rsi': rsi_val,
                    'bb_upper': bb_upper_val,
                    'bb_middle': bb_middle_val,
                    'bb_lower': bb_lower_val,
                    'touch_type': touch_type,
                    'timestamp': timestamp,
                    'hot': hot,
                    'daily_change': daily_change
                }

                results.append(item)

                logging.info(f"Alert: {symbol} on {timeframe} timeframe touching {touch_type} BB line at {timestamp} {'🔥' if hot else ''}")

    return results

async def scan_for_bb_touches_async(proxy_manager):
    symbols = await get_perpetual_usdt_symbols_async(proxy_manager)
    results = []
    batch_size = 30
    active_timeframes = get_active_timeframes()

    # Cache handling for these timeframes
    cached_timeframes = ['1w', '1d', '4h']
    cached_results = {}
    uncached_timeframes = [tf for tf in active_timeframes if tf not in cached_timeframes]

    # Load cache for cached timeframes
    for timeframe in cached_timeframes:
        if timeframe in active_timeframes:
            cached_results[timeframe] = load_cache(timeframe)
            if cached_results[timeframe] is not None:
                logging.info(f"[CACHE] Loaded {timeframe} timeframe results from cache")
            else:
                logging.info(f"[CACHE] No valid cache found for {timeframe}, will scan fresh")

    total_symbols = len(symbols)

    # Scan cached timeframes that have no valid cache
    for timeframe in cached_timeframes:
        if timeframe in active_timeframes and cached_results.get(timeframe) is None:
            logging.info(f"[SCAN] Scanning {timeframe} timeframe fresh data...")
            timeframe_results = []
            for i in range(0, total_symbols, batch_size):
                batch = symbols[i:i+batch_size]
                tasks = [scan_symbol_async(symbol, [timeframe], proxy_manager) for symbol in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in batch_results:
                    if isinstance(res, Exception):
                        logging.error(f"Error scanning {timeframe} timeframe: {res}")
                    else:
                        timeframe_results.extend(res)
            save_cache(timeframe, timeframe_results)
            cached_results[timeframe] = timeframe_results
            logging.info(f"[CACHE] Saved fresh scan results for {timeframe}")

    # Scan uncached timeframes all at once
    uncached_results = []
    if uncached_timeframes:
        logging.info(f"[SCAN] Scanning uncached timeframes: {uncached_timeframes}")
        for i in range(0, total_symbols, batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [scan_symbol_async(symbol, uncached_timeframes, proxy_manager) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in batch_results:
                if isinstance(res, Exception):
                    logging.error(f"Error scanning uncached timeframes: {res}")
                else:
                    uncached_results.extend(res)

    # Combine results
    for timeframe in cached_timeframes:
        if timeframe in cached_results:
            results.extend(cached_results[timeframe])
    results.extend(uncached_results)

    logging.info(f"[RESULTS] Total BB touches found: {len(results)}")
    return results

# === Formatting and Telegram sending ===

def format_results_by_timeframe(results, cached_timeframes_used=None):
    if not results:
        return ["*No BB touches detected at this time.*"]

    timeframe_order = {'1w': 7, '1d': 6, '4h': 5, '2h': 4, '1h': 3, '30m': 2, '15m': 1, '5m': 0, '3m': -1}

    grouped = {}
    for r in results:
        grouped.setdefault(r['timeframe'], []).append(r)

    sorted_timeframes = sorted(grouped.keys(), key=lambda tf: timeframe_order.get(tf, -2), reverse=True)

    messages = []
    for timeframe, items in [(tf, grouped[tf]) for tf in sorted_timeframes]:
        header = f"*🔍 BB Touches on {timeframe} Timeframe ({len(items)} symbols)*"
        if cached_timeframes_used and timeframe in cached_timeframes_used:
            header += " _(from cache)_"
        header += "\n"

        upper_touches = [i for i in items if i['touch_type'] == 'UPPER']
        middle_touches = [i for i in items if i['touch_type'] == 'MIDDLE']
        lower_touches = [i for i in items if i['touch_type'] == 'LOWER']

        lines = []

        def format_line(item):
            base = f"*{item['symbol']}* - RSI: {item['rsi']:.2f}"
            if item.get('hot'):
                base += " 🔥"
            return "• " + base

        if upper_touches:
            lines.append("*⬆️ UPPER BB Touches:*")
            for item in sorted(upper_touches, key=lambda x: x['symbol']):
                lines.append(format_line(item))

        if middle_touches:
            if upper_touches:
                lines.append("")  # blank line separator
            lines.append("*➖ MIDDLE BB Touches:*")
            for item in sorted(middle_touches, key=lambda x: x['symbol']):
                lines.append(format_line(item))

        if lower_touches:
            if upper_touches or middle_touches:
                lines.append("")  # blank line separator
            lines.append("*⬇️ LOWER BB Touches:*")
            for item in sorted(lower_touches, key=lambda x: x['symbol']):
                lines.append(format_line(item))

        messages.append(header + "\n" + "\n".join(lines))

    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    messages = [m + f"\n\n_Report generated at {timestamp}_" for m in messages]
    return messages

def split_message(text, max_length=4000):
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def send_telegram_alert(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        for attempt in range(3):
            response = requests.post(url, data=payload, timeout=10)
            if response.status_code == 200:
                return True
            time.sleep(1)
        logging.error(f"Telegram alert failed: {response.text}")
        return False
    except Exception as e:
        logging.error(f"Exception sending Telegram alert: {e}")
        return False

# === Async main entry point ===

async def main_async():
    start_time = time.time()
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram bot token or chat ID not set in environment variables.")

    logging.info("Starting BB touch scanner bot...")

    cleanup_old_caches(max_age_days=7)

    try:
        proxy_pool = AsyncProxyPool(
            sources=PROXY_SOURCES,
            max_pool_size=25,
            min_working=5,
            check_interval=600,
            max_failures=3
        )
        await proxy_pool.initialize()

        results = await scan_for_bb_touches_async(proxy_pool)

        cached_timeframes_used = [
            tf for tf in ['1w', '1d', '4h']
            if tf in get_active_timeframes() and load_cache(tf) is not None
        ]

        messages = format_results_by_timeframe(results, cached_timeframes_used=cached_timeframes_used)
        fresh_messages = [msg for msg in messages if "(from cache)" not in msg]

        if not fresh_messages:
            logging.info("No fresh BB touch alerts to send (all messages from cache).")

        for i, msg in enumerate(fresh_messages, 1):
            logging.info(f"Sending fresh message {i}/{len(fresh_messages)}")
            chunks = split_message(msg)
            for idx, chunk in enumerate(chunks, 1):
                if len(chunks) > 1:
                    chunk = f"{chunk}\n\n_Part {idx} of {len(chunks)}_"
                success = send_telegram_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, chunk)
                if not success:
                    logging.error(f"Failed to send part {idx} of message {i}")

        elapsed = time.time() - start_time
        logging.info(f"Bot run completed successfully in {elapsed:.2f} seconds")

    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"Fatal error after {elapsed:.2f}s: {e}")
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            error_msg = (
                f"*⚠️ Scanner Error*\n\n"
                f"The bot encountered an error after running for {elapsed:.2f}s:\n`{str(e)}`"
            )
            send_telegram_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, error_msg)

if __name__ == "__main__":
    asyncio.run(main_async())
