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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import itertools
import glob

# === CONFIG & CONSTANTS ===

logging.basicConfig(
    level=logging.DEBUG,
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

# === PROXY MANAGER ===

class ProxyManager:
    def __init__(self, proxy_sources, min_working_proxies=3):
        self.proxy_sources = proxy_sources
        self.min_working_proxies = min_working_proxies
        self.proxies = []
        self.blacklisted = set()
        self.lock = threading.RLock()
        self.refresh_in_progress = False
        self.proxy_cycle = None

    def _initialize_proxies(self):
        logging.info("Starting proxy initialization...")
        asyncio.run(self._refresh_proxies_async())
        logging.info(f"Proxy initialization complete. Found {len(self.proxies)} working proxies")

    async def _refresh_proxies_async(self):
        with self.lock:
            if self.refresh_in_progress:
                logging.debug("Proxy refresh already in progress, skipping...")
                return
            self.refresh_in_progress = True
        try:
            logging.info("Refreshing proxy pool asynchronously...")
            new_proxies = await self._fetch_and_test_proxies_async()
            with self.lock:
                good_existing = [p for p in self.proxies if p['failures'] < 2]
                new_proxy_addrs = [p['proxy'] for p in good_existing]
                for proxy, speed in new_proxies:
                    if proxy not in new_proxy_addrs:
                        good_existing.append({'proxy': proxy, 'failures': 0, 'speed': speed})
                self.proxies = sorted(good_existing, key=lambda x: (x['failures'], x['speed']))
                self.proxy_cycle = None
                logging.info(f"Proxy pool refreshed. Now have {len(self.proxies)} working proxies")
        except Exception as e:
            logging.error(f"Error refreshing proxies: {str(e)}")
        finally:
            with self.lock:
                self.refresh_in_progress = False

    async def _fetch_and_test_proxies_async(self):
        all_proxies = set()
        async with aiohttp.ClientSession() as session:
            for url in self.proxy_sources:
                try:
                    logging.info(f"Fetching proxies from {url} ...")
                    async with session.get(url, timeout=10) as resp:
                        text = await resp.text()
                        proxies = [line.strip() for line in text.splitlines()
                                   if line.strip() and line.strip() not in self.blacklisted]
                        all_proxies.update(proxies)
                        logging.info(f"Fetched {len(proxies)} proxies from {url}")
                except Exception as e:
                    logging.error(f"Failed to fetch proxies from {url}: {e}")

            logging.info(f"Total unique proxies fetched: {len(all_proxies)}")

            test_proxies = list(all_proxies)
            random.shuffle(test_proxies)
            test_proxies = test_proxies[:300]  # Limit to 300 for testing

            logging.info(f"Testing {len(test_proxies)} proxies against Binance asynchronously...")

            working = []
            http_451_count = 0
            other_fail_count = 0

            tasks = [self._test_proxy_async(session, proxy) for proxy in test_proxies]
            for future in asyncio.as_completed(tasks):
                proxy, speed = await future
                if proxy:
                    working.append((proxy, speed))
                    logging.info(f"Proxy {proxy} works, response time: {speed:.2f}s")
                    fast_proxies = [p for p, s in working if s < 5]
                    if len(fast_proxies) >= self.min_working_proxies:
                        logging.info(f"Found minimum required fast proxies ({len(fast_proxies)}), stopping test early.")
                        break
                else:
                    # Count failures for diagnostics
                    # We can enhance _test_proxy_async to return failure reason if needed
                    pass

            logging.info(f"Proxy testing complete. Working proxies: {len(working)}")

            if not working:
                logging.warning("No working proxies found!")
                return []

            # Filter fast proxies (<10s)
            fast_proxies = [(p, s) for p, s in working if s < 10]
            if len(fast_proxies) < self.min_working_proxies:
class ProxyManager:
    def __init__(self, proxy_sources, min_working_proxies=3):
        self.proxy_sources = proxy_sources
        self.min_working_proxies = min_working_proxies
        self.proxies = []
        self.blacklisted = set()
        self.lock = threading.RLock()
        self.refresh_in_progress = False
        self.proxy_cycle = None

    def _initialize_proxies(self):
        logging.info("Starting proxy initialization...")
        asyncio.run(self._refresh_proxies_async())
        logging.info(f"Proxy initialization complete. Found {len(self.proxies)} working proxies")

    async def _refresh_proxies_async(self):
        with self.lock:
            if self.refresh_in_progress:
                logging.debug("Proxy refresh already in progress, skipping...")
                return
            self.refresh_in_progress = True
        try:
            logging.info("Refreshing proxy pool asynchronously...")
            new_proxies = await self._fetch_and_test_proxies_async()
            logging.info(f"Number of new working proxies found: {len(new_proxies)}")
            with self.lock:
                good_existing = [p for p in self.proxies if p['failures'] < 2]
                new_proxy_addrs = [p['proxy'] for p in good_existing]
                for proxy, speed in new_proxies:
                    if proxy not in new_proxy_addrs:
                        good_existing.append({'proxy': proxy, 'failures': 0, 'speed': speed})
                self.proxies = sorted(good_existing, key=lambda x: (x['failures'], x['speed']))
                self.proxy_cycle = None
                logging.info(f"Proxy pool refreshed. Now have {len(self.proxies)} working proxies")
        except Exception as e:
            logging.error(f"Error refreshing proxies: {str(e)}")
        finally:
            with self.lock:
                self.refresh_in_progress = False

    async def _fetch_and_test_proxies_async(self):
        all_proxies = set()
        async with aiohttp.ClientSession() as session:
            for url in self.proxy_sources:
                try:
                    logging.info(f"Fetching proxies from {url} ...")
                    async with session.get(url, timeout=10) as resp:
                        text = await resp.text()
                        proxies = [line.strip() for line in text.splitlines()
                                   if line.strip() and line.strip() not in self.blacklisted]
                        all_proxies.update(proxies)
                        logging.info(f"Fetched {len(proxies)} proxies from {url}")
                except Exception as e:
                    logging.error(f"Failed to fetch proxies from {url}: {e}")

            logging.info(f"Total unique proxies fetched: {len(all_proxies)}")

            test_proxies = list(all_proxies)
            random.shuffle(test_proxies)
            test_proxies = test_proxies[:300]

            logging.info(f"Testing {len(test_proxies)} proxies against Binance asynchronously...")

            working = []
            tasks = [self._test_proxy_async(session, proxy) for proxy in test_proxies]

            for future in asyncio.as_completed(tasks):
                proxy, speed = await future
                if proxy:
                    working.append((proxy, speed))
                    logging.info(f"Proxy {proxy} works, response time: {speed:.2f}s")
                    fast_proxies = [p for p, s in working if s < 5]
                    if len(fast_proxies) >= self.min_working_proxies:
                        logging.info(f"Found minimum required fast proxies ({len(fast_proxies)}), stopping test early.")
                        break

            logging.info(f"Proxy testing complete. Working proxies: {len(working)}")

            if not working:
                logging.warning("No working proxies found!")
                return []

            fast_proxies = [(p, s) for p, s in working if s < 10]
            if len(fast_proxies) < self.min_working_proxies:
                logging.warning(f"Found only {len(fast_proxies)} fast proxies, accepting some slower ones...")
                slower = [(p, s) for p, s in working if 10 <= s <= 20]
                slower.sort(key=lambda x: x[1])
                fast_proxies.extend(slower[:self.min_working_proxies - len(fast_proxies)])

            return sorted(fast_proxies, key=lambda x: x[1])

    async def _test_proxy_async(self, session, proxy):
        if proxy in self.blacklisted:
            logging.debug(f"Proxy {proxy} is blacklisted, skipping test.")
            return None, None

        proxy_url = proxy if proxy.startswith("http://") or proxy.startswith("https://") else f"http://{proxy}"

        try:
            start = time.time()
            async with session.get(PROXY_TEST_URL, proxy=proxy_url, timeout=8, ssl=True) as resp:
                if resp.status == 451:
                    logging.warning(f"Proxy {proxy} blocked with HTTP 451, blacklisting.")
                    self.blacklisted.add(proxy)
                    return None, None
                if resp.status != 200:
                    logging.warning(f"Proxy {proxy} returned status {resp.status}, blacklisting.")
                    self.blacklisted.add(proxy)
                    return None, None
                elapsed = time.time() - start
                return proxy, elapsed
        except Exception as e:
            logging.warning(f"Proxy {proxy} test failed with exception: {e}")
            self.blacklisted.add(proxy)
            return None, None

    def _update_proxy_cycle(self):
        with self.lock:
            good_proxies = [p for p in self.proxies if p['failures'] < 3]
            if not good_proxies:
                raise RuntimeError("No working proxies available.")
            random.shuffle(good_proxies)
            self.proxy_cycle = itertools.cycle(good_proxies)
            logging.info(f"Proxy cycle updated with {len(good_proxies)} proxies")

    def get_proxy(self):
        with self.lock:
            if self.proxy_cycle is None:
                self._update_proxy_cycle()
            try:
                proxy_info = next(self.proxy_cycle)
                logging.debug(f"Selected proxy {proxy_info['proxy']} from cycle")
                return proxy_info
            except StopIteration:
                self._update_proxy_cycle()
                proxy_info = next(self.proxy_cycle)
                logging.debug(f"Selected proxy {proxy_info['proxy']} from cycle after reset")
                return proxy_info

    def mark_success(self, proxy_info):
        with self.lock:
            for p in self.proxies:
                if p['proxy'] == proxy_info['proxy']:
                    if p['failures'] > 0:
                        p['failures'] = 0
                        logging.info(f"Proxy {p['proxy']} marked success, failures reset")
                    break

    def mark_failure(self, proxy_info):
        with self.lock:
            for p in self.proxies:
                if p['proxy'] == proxy_info['proxy']:
                    p['failures'] += 1
                    logging.warning(f"Proxy {p['proxy']} failure count now {p['failures']}")
                    if p['failures'] >= 3:
                        logging.warning(f"Removing failed proxy {p['proxy']}")
                        self.proxies.remove(p)
                        self.blacklisted.add(p['proxy'])
                        self.proxy_cycle = None
                        if len(self.proxies) < self.min_working_proxies and not self.refresh_in_progress:
                            threading.Thread(target=self._refresh_proxies_async, daemon=True).start()
                    break


# === ASYNC REQUESTS & SCANNING ===

async def make_request_async(session, url, params=None, proxy_manager=None, max_attempts=4):
    if proxy_manager is None:
        raise ValueError("proxy_manager argument is required")
    attempt = 0
    while attempt < max_attempts:
        proxy_info = None
        try:
            proxy_info = proxy_manager.get_proxy()
        except RuntimeError:
            logging.warning("No working proxies available, waiting before retry...")
            await asyncio.sleep(5)
            continue

        proxy_str = proxy_info['proxy']
        proxy_url = proxy_str if proxy_str.startswith("http://") or proxy_str.startswith("https://") else f"http://{proxy_str}"

        try:
            async with session.get(url, params=params, proxy=proxy_url, timeout=15, ssl=True) as resp:
                if resp.status == 451:
                    logging.warning(f"Proxy {proxy_str} returned HTTP 451, blacklisting and retrying with different proxy")
                    proxy_manager.blacklisted.add(proxy_str)
                    proxy_manager.mark_failure(proxy_info)
                    continue  # retry immediately without incrementing attempt

                resp.raise_for_status()
                proxy_manager.mark_success(proxy_info)
                return await resp.json()

        except Exception as e:
            logging.error(f"Request failed with proxy {proxy_str}: {e}")
            if proxy_info:
                proxy_manager.mark_failure(proxy_info)
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
        proxy_info = proxy_manager.get_proxy()
        proxy_str = proxy_info['proxy']
        proxy_url = proxy_str if proxy_str.startswith("http://") or proxy_str.startswith("https://") else f"http://{proxy_str}"
        data = await make_request_async(session, BINANCE_FUTURES_KLINES, params=params, proxy_manager=proxy_manager)
        proxy_manager.mark_success(proxy_info)
        closes = [float(k[4]) for k in data]
        timestamps = [k[0] for k in data]
        return closes, timestamps
    except Exception as e:
        logging.error(f"Error fetching klines for {symbol} {interval}: {e}")
        proxy_manager.mark_failure(proxy_info)
        return None, None

async def get_daily_change_percent_async(session, symbol, proxy_manager):
    params = {'symbol': symbol, 'interval': '1d', 'limit': 1}
    try:
        proxy_info = proxy_manager.get_proxy()
        proxy_str = proxy_info['proxy']
        proxy_url = proxy_str if proxy_str.startswith("http://") or proxy_str.startswith("https://") else f"http://{proxy_str}"
        data = await make_request_async(session, BINANCE_FUTURES_KLINES, params=params, proxy_manager=proxy_manager)
        proxy_manager.mark_success(proxy_info)
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
        proxy_manager.mark_failure(proxy_info)
        return None

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

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        tasks = [scan_symbol_async(symbol, active_timeframes, proxy_manager) for symbol in batch]
        batch_results = await asyncio.gather(*tasks)
        for res in batch_results:
            results.extend(res)

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
        proxy_manager = ProxyManager(PROXY_SOURCES, min_working_proxies=3)
        logging.info("Initializing proxies...")
        await proxy_manager._refresh_proxies_async()  # Explicitly refresh proxies before scanning
        logging.info("Proxies initialized.")
        results = await scan_for_bb_touches_async(proxy_manager)
        cached_timeframes_used = [tf for tf in ['1w', '1d', '4h'] if tf in get_active_timeframes() and load_cache(tf) is not None]
        messages = format_results_by_timeframe(results, cached_timeframes_used=cached_timeframes_used)

        for i, msg in enumerate(messages, 1):
            logging.info(f"Sending message {i}/{len(messages)}")
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
            error_msg = f"*⚠️ Scanner Error*\n\nThe bot encountered an error after running for {elapsed:.2f}s:\n`{str(e)}`"
            send_telegram_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, error_msg)

if __name__ == "__main__":
    asyncio.run(main_async())
