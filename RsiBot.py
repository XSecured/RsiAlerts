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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# Constants
BINANCE_FUTURES_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"
CANDLE_LIMIT = 55
UPPER_TOUCH_THRESHOLD = 0.02  # 2%
LOWER_TOUCH_THRESHOLD = 0.02  # 2%
RSI_PERIOD = 14
BB_LENGTH = 34
BB_STDDEV = 2

PROXY_SOURCES = [
    "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/https.txt",
    # Add more trusted sources as needed
]

# Timeframe toggles
TIMEFRAMES_TOGGLE = {
    '3m': True,
    '5m': True,
    '15m': True,
    '30m': True,
    '1h': True,
    '2h': True,
    '4h': True,
    '1d': True,
    '1w': True,
}

def get_active_timeframes():
    return [tf for tf, enabled in TIMEFRAMES_TOGGLE.items() if enabled]

class ProxyManager:
    def __init__(self, proxy_sources, min_working_proxies=10, max_failures=3, refresh_interval=180):
        self.proxy_sources = proxy_sources
        self.min_working_proxies = min_working_proxies
        self.max_failures = max_failures
        self.proxies = []  # [{'proxy': str, 'failures': int, 'speed': float}]
        self.blacklisted = set()
        self.lock = threading.Lock()
        self.refresh_interval = refresh_interval  # seconds
        self.last_refresh = 0
        self.refresh_in_progress = False
        
        self._initialize_proxies()
        # Start background proxy refresher thread
        threading.Thread(target=self._background_refresh_loop, daemon=True).start()

    def _initialize_proxies(self):
        logging.info("Initializing proxy pool...")
        self._refresh_proxies(blocking=True)

    def _background_refresh_loop(self):
        while True:
            time.sleep(self.refresh_interval)
            with self.lock:
                if not self.refresh_in_progress:
                    logging.info("Background proxy refresh triggered.")
                    self._refresh_proxies(blocking=False)

    def _refresh_proxies(self, blocking=False):
        with self.lock:
            if self.refresh_in_progress:
                return
            self.refresh_in_progress = True
        
        def refresh():
            try:
                logging.info("Refreshing proxies...")
                new_proxies = self._fetch_and_test_proxies()
                with self.lock:
                    good_existing = [p for p in self.proxies if p['failures'] < self.max_failures]
                    existing_proxies = {p['proxy'] for p in good_existing}
                    for proxy, speed in new_proxies:
                        if proxy not in existing_proxies and proxy not in self.blacklisted:
                        good_existing.append({'proxy': proxy, 'failures': 0, 'speed': speed})
                    self.proxies = sorted(good_existing, key=lambda x: (x['failures'], x['speed']))
                    self.last_refresh = time.time()
                    logging.info(f"Proxy pool refreshed: {len(self.proxies)} proxies available.")
            except Exception as e:
                logging.error(f"Error refreshing proxies: {e}")
            finally:
                with self.lock:
                    self.refresh_in_progress = False
        
        if blocking:
            refresh()
        else:
            threading.Thread(target=refresh, daemon=True).start()

    def _fetch_and_test_proxies(self):
        all_proxies = set()
        for url in self.proxy_sources:
            try:
                resp = requests.get(url, timeout=10)
                proxies = [line.strip() for line in resp.text.splitlines() if line.strip() and line.strip() not in self.blacklisted]
                all_proxies.update(proxies)
                logging.info(f"Fetched {len(proxies)} proxies from {url}")
            except Exception as e:
                logging.error(f"Failed to fetch proxies from {url}: {e}")

        test_proxies = list(all_proxies)
        random.shuffle(test_proxies)
        test_proxies = test_proxies[:200]

        working = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(self._test_proxy, proxy): proxy for proxy in test_proxies}
            for future in as_completed(futures):
                proxy, speed = future.result()
                if proxy:
                    working.append((proxy, speed))
                    if len(working) >= self.min_working_proxies * 3:
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
        if not working:
            logging.warning("No working proxies found!")
            return []
        if len(working) < self.min_working_proxies:
            logging.warning(f"Only {len(working)} working proxies found, less than min_working_proxies={self.min_working_proxies}")
        return sorted(working, key=lambda x: x[1])

    def _test_proxy(self, proxy):
        proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
        try:
            start = time.time()
            resp = requests.get(BINANCE_FUTURES_EXCHANGE_INFO, proxies=proxies, timeout=5)
            if resp.status_code != 200:
                self.blacklisted.add(proxy)
                return None, None
            speed = time.time() - start
            if speed > 5:  # Only keep proxies faster than 5 seconds
                self.blacklisted.add(proxy)
                return None, None
            return proxy, speed
        except Exception:
            self.blacklisted.add(proxy)
            return None, None

    def get_proxy(self):
        with self.lock:
            if not self.proxies:
                logging.error("No working proxies available.")
                raise RuntimeError("No working proxies available.")
            # Select the fastest proxy with failures < max_failures
            available = [p for p in self.proxies if p['failures'] < self.max_failures]
            if not available:
                logging.error("All proxies have reached failure limit.")
                raise RuntimeError("All proxies have reached failure limit.")
            fastest = min(available, key=lambda x: x['speed'])
            return fastest

    def mark_failure(self, proxy_info):
        with self.lock:
            for p in self.proxies:
                if p['proxy'] == proxy_info['proxy']:
                    p['failures'] += 1
                    logging.warning(f"Proxy {p['proxy']} failure count: {p['failures']}")
                    if p['failures'] >= self.max_failures:
                        logging.warning(f"Removing proxy {p['proxy']} due to failures")
                        self.proxies.remove(p)
                        self.blacklisted.add(p['proxy'])
                    break

def make_request(url, params=None, proxy_manager=None, max_attempts=5):
    last_exc = None
    for attempt in range(max_attempts):
        proxy_info = None
        try:
            proxy_info = proxy_manager.get_proxy()
            proxies = {"http": f"http://{proxy_info['proxy']}", "https": f"http://{proxy_info['proxy']}"}
            logging.info(f"Request to {url.split('/')[-1]} using proxy {proxy_info['proxy']} (attempt {attempt+1}/{max_attempts})")
            resp = requests.get(url, params=params, proxies=proxies, timeout=10, verify=False)  # Fixed timeout
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logging.error(f"Request failed: {e}")
            if proxy_info:
                proxy_manager.mark_failure(proxy_info)
            last_exc = e
            logging.info("Retrying in 3 seconds...")
            time.sleep(3)
    raise RuntimeError(f"Request failed after {max_attempts} attempts: {last_exc}")

def get_perpetual_usdt_symbols(proxy_manager):
    logging.info("Fetching USDT perpetual symbols...")
    for attempt in range(5):
        try:
            data = make_request(BINANCE_FUTURES_EXCHANGE_INFO, proxy_manager=proxy_manager)
            symbols = [
                s['symbol'] for s in data['symbols']
                if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
            ]
            logging.info(f"Fetched {len(symbols)} symbols")
            if len(symbols) < 10:
                logging.warning("Too few symbols, retrying...")
                time.sleep(3)
                continue
            return symbols
        except Exception as e:
            logging.error(f"Error fetching symbols: {e}")
            time.sleep(5)
    raise RuntimeError("Failed to fetch symbols after multiple attempts")

def fetch_klines(symbol, interval, proxy_manager, limit=CANDLE_LIMIT):
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        data = make_request(BINANCE_FUTURES_KLINES, params=params, proxy_manager=proxy_manager)
        closes = [float(k[4]) for k in data]
        timestamps = [k[0] for k in data]
        if len(closes) < limit:
            logging.warning(f"Fewer than {limit} klines for {symbol} {interval}")
        return closes, timestamps
    except Exception as e:
        logging.error(f"Error fetching klines for {symbol} {interval}: {e}")
        return [], []

def calculate_rsi_bb(closes):
    closes_np = np.array(closes)
    rsi = talib.RSI(closes_np, timeperiod=RSI_PERIOD)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        rsi, timeperiod=BB_LENGTH, nbdevup=BB_STDDEV, nbdevdn=BB_STDDEV, matype=0
    )
    return rsi, bb_upper, bb_middle, bb_lower

def scan_symbol(symbol, timeframes, proxy_manager):
    results = []
    for timeframe in timeframes:
        closes, timestamps = fetch_klines(symbol, timeframe, proxy_manager)
        if len(closes) < CANDLE_LIMIT:
            logging.warning(f"Not enough klines data for {symbol} {timeframe}, skipping.")
            continue
        rsi, bb_upper, bb_middle, bb_lower = calculate_rsi_bb(closes)
        idx = -2  # Use the previous candle to avoid open candle
        if idx < -len(closes):
            logging.warning(f"Not enough candles for {symbol} {timeframe} to skip open candle. Skipping.")
            continue
        if np.isnan(rsi[idx]) or np.isnan(bb_upper[idx]) or np.isnan(bb_lower[idx]):
            logging.warning(f"NaN values for {symbol} {timeframe}, skipping.")
            continue
        rsi_val = rsi[idx]
        bb_upper_val = bb_upper[idx]
        bb_lower_val = bb_lower[idx]
        upper_touch = rsi_val >= bb_upper_val * (1 - UPPER_TOUCH_THRESHOLD)
        lower_touch = rsi_val <= bb_lower_val * (1 + LOWER_TOUCH_THRESHOLD)
        if upper_touch or lower_touch:
            touch_type = "UPPER" if upper_touch else "LOWER"
            timestamp = datetime.utcfromtimestamp(timestamps[idx] / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')
            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'rsi': rsi_val,
                'bb_upper': bb_upper_val,
                'bb_lower': bb_lower_val,
                'touch_type': touch_type,
                'timestamp': timestamp
            })
            logging.info(f"Alert: {symbol} on {timeframe} touching {touch_type} BB line at {timestamp}")
    return results

def scan_for_bb_touches(proxy_manager):
    symbols = get_perpetual_usdt_symbols(proxy_manager)
    results = []
    total_symbols = len(symbols)
    completed = 0
    batch_size = 20
    active_timeframes = get_active_timeframes()
    logging.info(f"Active timeframes: {active_timeframes}")
    for i in range(0, total_symbols, batch_size):
        batch = symbols[i:i+batch_size]
        batch_results = []
        logging.info(f"Processing batch {i//batch_size + 1}/{(total_symbols + batch_size - 1)//batch_size}")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(scan_symbol, symbol, active_timeframes, proxy_manager): symbol for symbol in batch}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    symbol_results = future.result()
                    batch_results.extend(symbol_results)
                except Exception as e:
                    logging.error(f"Error scanning {symbol}: {e}")
                completed += 1
                if completed % 10 == 0 or completed == total_symbols:
                    logging.info(f"Completed {completed}/{total_symbols} symbols")
        results.extend(batch_results)
        if i + batch_size < total_symbols:
            time.sleep(1)
    logging.info(f"Scan completed for all {total_symbols} symbols")
    return results

def format_results_by_timeframe(results):
    if not results:
        return ["*No BB touches detected at this time.*"]
    grouped = {}
    for r in results:
        grouped.setdefault(r['timeframe'], []).append(r)
    messages = []
    for timeframe, items in sorted(grouped.items()):
        header = f"*🔍 BB Touches on {timeframe} Timeframe ({len(items)} symbols)*\n"
        upper_touches = [i for i in items if i['touch_type'] == 'UPPER']
        lower_touches = [i for i in items if i['touch_type'] == 'LOWER']
        lines = []
        if upper_touches:
            lines.append("*⬆️ UPPER BB Touches:*")
            for item in sorted(upper_touches, key=lambda x: x['symbol']):
                lines.append(f"• *{item['symbol']}* - RSI: {item['rsi']:.2f}")
        if lower_touches:
            if upper_touches:
                lines.append("")
            lines.append("*⬇️ LOWER BB Touches:*")
            for item in sorted(lower_touches, key=lambda x: x['symbol']):
                lines.append(f"• *{item['symbol']}* - RSI: {item['rsi']:.2f}")
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
        proxy_manager = ProxyManager(PROXY_SOURCES, min_working_proxies=10, refresh_interval=180)
        results = scan_for_bb_touches(proxy_manager)
        messages = format_results_by_timeframe(results)

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
