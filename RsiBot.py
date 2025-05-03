import requests
import pandas as pd
import numpy as np
import talib
import logging
import time
import os
import threading
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import itertools

# Configure logging with more detail, keep SSL warnings visible but avoid disabling verify in requests calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

CACHE_DIR = "bb_touch_cache"  # Directory to store cache files
os.makedirs(CACHE_DIR, exist_ok=True)

# Constants
BINANCE_FUTURES_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"
PROXY_TEST_URL = "https://api.binance.com/api/v3/time"  # Lightweight proxy test URL
CANDLE_LIMIT = 55
UPPER_TOUCH_THRESHOLD = 0.02  # 2%
MIDDLE_TOUCH_THRESHOLD = 0.015  # 1.5%, between upper and lower thresholds
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

# Which timeframes should include/display the middle BB line?
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

def get_cache_file_name(timeframe):
    return os.path.join(CACHE_DIR, f"bb_touch_cache_{timeframe}.json")

def load_cache(timeframe):
    cache_file = get_cache_file_name(timeframe)
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        cache_date = data.get('date')
        if cache_date == datetime.utcnow().strftime('%Y-%m-%d'):
            return data.get('results')
    except Exception as e:
        logging.warning(f"Failed to load {timeframe} cache: {e}")
    return None

def save_cache(timeframe, results):
    cache_file = get_cache_file_name(timeframe)
    data = {
        'date': datetime.utcnow().strftime('%Y-%m-%d'),
        'results': results
    }
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        logging.info(f"Cache saved successfully for {timeframe} at {cache_file}")
    except Exception as e:
        logging.warning(f"Failed to save {timeframe} cache: {e}")

def get_active_timeframes():
    return [tf for tf, enabled in TIMEFRAMES_TOGGLE.items() if enabled]

def test_proxy(proxy: str) -> bool:
    try:
        proxies = {"http": proxy, "https": proxy}
        response = requests.get(PROXY_TEST_URL, proxies=proxies, timeout=10, verify=True)
        return response.status_code in range(200, 300)
    except Exception as e:
        if "Connection reset" in str(e):
            logging.debug("Proxy %s failed with connection reset", proxy)
        else:
            logging.debug("Proxy %s failed: %s", proxy, e)
        return False

class ProxyManager:
    def __init__(self, proxy_sources, min_working_proxies=3):
        self.proxy_sources = proxy_sources
        self.min_working_proxies = min_working_proxies
        self.proxies = []  # [{'proxy': proxy_str, 'failures': 0, 'speed': response_time}]
        self.blacklisted = set()
        self.lock = threading.RLock()  # Changed to RLock to avoid deadlock
        self.refresh_in_progress = False
        self.proxy_cycle = None
        self._initialize_proxies()

    def _initialize_proxies(self):
        logging.info("Starting proxy initialization...")
        self._refresh_proxies(blocking=True)
        logging.info(f"Proxy initialization complete. Found {len(self.proxies)} working proxies")

    def _refresh_proxies(self, blocking=False):
        with self.lock:
            if self.refresh_in_progress:
                logging.debug("Proxy refresh already in progress, skipping...")
                return
            self.refresh_in_progress = True
        try:
            logging.info("Refreshing proxy pool...")
            new_proxies = self._fetch_and_test_proxies()
            with self.lock:
                good_existing = [p for p in self.proxies if p['failures'] < 2]
                new_proxy_addrs = [p['proxy'] for p in good_existing]
                for proxy, speed in new_proxies:
                    if proxy not in new_proxy_addrs:
                        good_existing.append({'proxy': proxy, 'failures': 0, 'speed': speed})
                self.proxies = sorted(good_existing, key=lambda x: (x['failures'], x['speed']))
                self.proxy_cycle = None  # Reset cycle to refresh proxies
                logging.info(f"Proxy pool refreshed. Now have {len(self.proxies)} working proxies")
        except Exception as e:
            logging.error(f"Error refreshing proxies: {str(e)}")
        finally:
            with self.lock:
                self.refresh_in_progress = False

    def _fetch_and_test_proxies(self):
        all_proxies = set()
        for url in self.proxy_sources:
            try:
                logging.info(f"Fetching proxies from {url}")
                response = requests.get(url, timeout=10)
                proxies = [line.strip() for line in response.text.splitlines()
                           if line.strip() and line.strip() not in self.blacklisted]
                all_proxies.update(proxies)
                logging.info(f"Found {len(proxies)} candidate proxies from {url}")
            except Exception as e:
                logging.error(f"Failed to fetch proxies from {url}: {e}")

        test_proxies = list(all_proxies)
        random.shuffle(test_proxies)
        test_proxies = test_proxies[:200]

        logging.info(f"Testing {len(test_proxies)} proxies against Binance...")
        working = []

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = {executor.submit(self._test_proxy, proxy): proxy for proxy in test_proxies}
            for future in as_completed(futures):
                result = future.result()
                if result[0]:
                    working.append(result)
                    logging.info(f"Proxy {result[0]} works, response time: {result[1]:.2f}s")
                    fast_proxies = [p for p, s in working if s < 5]
                    if len(fast_proxies) >= self.min_working_proxies:
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break

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

    def _test_proxy(self, proxy):
        if proxy in self.blacklisted:
            return None, None

        proxy_url = proxy if proxy.startswith("http://") or proxy.startswith("https://") else f"http://{proxy}"
        proxies = {"http": proxy_url, "https": proxy_url}

        try:
            start = time.time()
            response = requests.get("https://api.binance.com/api/v3/time", proxies=proxies, timeout=(3, 5), verify=True)
            if response.status_code != 200:
                self.blacklisted.add(proxy)
                return None, None
            elapsed = time.time() - start
            return proxy, elapsed
        except Exception:
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
                return proxy_info
            except StopIteration:
                self._update_proxy_cycle()
                return next(self.proxy_cycle)

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
                        self.proxy_cycle = None  # reset cycle to refresh proxies
                        if len(self.proxies) < self.min_working_proxies and not self.refresh_in_progress:
                            threading.Thread(target=self._refresh_proxies, daemon=True).start()
                    break

def make_request(url, params=None, proxy_manager=None, max_attempts=4):
    for attempt in range(max_attempts):
        try:
            proxy_info = proxy_manager.get_proxy()
        except RuntimeError as e:
            logging.warning("No working proxies available, refreshing proxy pool and retrying...")
            if not proxy_manager.refresh_in_progress:
                threading.Thread(target=proxy_manager._refresh_proxies, daemon=True).start()
            time.sleep(5)
            continue

        proxy_str = proxy_info['proxy']
        proxy_url = proxy_str if proxy_str.startswith("http://") or proxy_str.startswith("https://") else f"http://{proxy_str}"
        proxies = {"http": proxy_url, "https": proxy_url}

        endpoint = url.split('/')[-1] if '/' in url else url
        logging.debug(f"Request to {endpoint}: using proxy {proxy_str} (attempt {attempt+1}/{max_attempts})")

        try:
            connect_timeout = 5
            read_timeout = max(10, int(proxy_info['speed'] * 2))
            resp = requests.get(url, params=params, proxies=proxies, timeout=(connect_timeout, read_timeout), verify=True)
            resp.raise_for_status()
            proxy_manager.mark_success(proxy_info)
            logging.debug(f"Request successful: {endpoint}")
            return resp.json()
        except Exception as e:
            logging.error(f"Request failed: {str(e)}")
            proxy_manager.mark_failure(proxy_info)
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Request failed after {max_attempts} attempts")
            wait_time = 2 * (attempt + 1)
            logging.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

def get_perpetual_usdt_symbols(proxy_manager, max_attempts=5, per_attempt_timeout=10):
    logging.info("Starting to fetch USDT perpetual symbols list via proxies only...")

    for attempt in range(1, max_attempts + 1):
        try:
            logging.info(f"Fetching symbols via proxy (attempt {attempt}/{max_attempts})...")
            data = make_request(
                BINANCE_FUTURES_EXCHANGE_INFO,
                proxy_manager=proxy_manager,
                max_attempts=2
            )
            symbols = [
                s['symbol'] for s in data.get('symbols', [])
                if s.get('contractType') == 'PERPETUAL'
                   and s.get('quoteAsset') == 'USDT'
                   and s.get('status') == 'TRADING'
            ]
            logging.info(f"Fetched {len(symbols)} symbols via proxy")
            if len(symbols) >= 10:
                return symbols
            logging.warning(f"Too few symbols ({len(symbols)}) via proxy, retrying...")
        except Exception as e:
            logging.warning(f"Proxy attempt {attempt} failed: {e}")
        time.sleep(3)

    raise RuntimeError("Failed to fetch USDT perpetual symbols via proxies after multiple attempts")

def fetch_klines(symbol, interval, proxy_manager, limit=CANDLE_LIMIT):
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        logging.debug(f"Fetching {symbol} {interval} klines...")
        data = make_request(BINANCE_FUTURES_KLINES, params=params, proxy_manager=proxy_manager)
        closes = [float(k[4]) for k in data]
        timestamps = [k[0] for k in data]
        return closes, timestamps
    except Exception as e:
        logging.error(f"Error fetching klines for {symbol} {interval}: {e}")
        return None, None

def get_daily_change_percent(symbol, proxy_manager):
    params = {'symbol': symbol, 'interval': '1d', 'limit': 1}
    try:
        data = make_request(BINANCE_FUTURES_KLINES, params=params, proxy_manager=proxy_manager)
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

def scan_symbol(symbol, timeframes, proxy_manager):
    results = []

    daily_change = get_daily_change_percent(symbol, proxy_manager)
    if daily_change is None:
        logging.warning(f"Could not get daily change for {symbol}")

    for timeframe in timeframes:
        closes, timestamps = fetch_klines(symbol, timeframe, proxy_manager)
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

        if MIDDLE_BAND_DETECTION.get(timeframe, False):
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

def scan_for_bb_touches(proxy_manager):
    symbols = get_perpetual_usdt_symbols(proxy_manager)
    results = []
    total_symbols = len(symbols)
    batch_size = 30
    active_timeframes = get_active_timeframes()

    cached_timeframes = ['1w', '1d', '4h']
    uncached_timeframes = [tf for tf in active_timeframes if tf not in cached_timeframes]

    cached_results = {}
    for timeframe in cached_timeframes:
        if timeframe in active_timeframes:
            cached_results[timeframe] = load_cache(timeframe)
            if cached_results[timeframe] is not None:
                logging.info(f"[CACHE] Loaded {timeframe} timeframe results from cache")
            else:
                logging.info(f"[CACHE] No valid cache found for {timeframe}, will scan fresh")

    for timeframe in cached_timeframes:
        if timeframe in active_timeframes and cached_results.get(timeframe) is None:
            logging.info(f"[SCAN] Scanning {timeframe} timeframe fresh data...")
            timeframe_results = []
            for i in range(0, total_symbols, batch_size):
                batch = symbols[i:i+batch_size]
                with ThreadPoolExecutor(max_workers=20) as executor:
                    futures = {executor.submit(scan_symbol, symbol, [timeframe], proxy_manager): symbol for symbol in batch}
                    for future in as_completed(futures):
                        try:
                            timeframe_results.extend(future.result())
                        except Exception as e:
                            logging.error(f"Error scanning {timeframe} timeframe: {e}")
            save_cache(timeframe, timeframe_results)
            cached_results[timeframe] = timeframe_results
            logging.info(f"[CACHE] Saved fresh scan results for {timeframe}")

    uncached_results = []
    if uncached_timeframes:
        logging.info(f"[SCAN] Scanning uncached timeframes: {uncached_timeframes}")
        for i in range(0, total_symbols, batch_size):
            batch = symbols[i:i+batch_size]
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(scan_symbol, symbol, uncached_timeframes, proxy_manager): symbol for symbol in batch}
                for future in as_completed(futures):
                    try:
                        uncached_results.extend(future.result())
                    except Exception as e:
                        logging.error(f"Error scanning uncached timeframes: {e}")

    results = []
    for timeframe in cached_timeframes:
        if timeframe in cached_results:
            results.extend(cached_results[timeframe])
    results.extend(uncached_results)

    logging.info(f"[RESULTS] Total BB touches found: {len(results)}")
    return results

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
            parts = [f"*{item['symbol']}*", f"RSI: {item['rsi']:.2f}"]
            # We don't include MB in the line format as per original style
            if item.get('hot'):
                parts.append("🔥")
            return "• " + " - ".join(parts)

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

def main():
    start_time = time.time()
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram bot token or chat ID not set in environment variables.")

    logging.info("Starting BB touch scanner bot...")

    try:
        logging.info("Initializing proxy manager...")
        proxy_manager = ProxyManager(PROXY_SOURCES, min_working_proxies=3)
        logging.info("Starting scan process...")
        results = scan_for_bb_touches(proxy_manager)
        logging.info(f"Scan complete, formatting {len(results)} results...")
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
    main()
