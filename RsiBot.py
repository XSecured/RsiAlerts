import requests
import pandas as pd
import numpy as np
import talib
import logging
import time
import os
import threading
import random
import json
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

# Proxy sources
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/https.txt"
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

# -----------------------------
# Proxy Pool System
# -----------------------------
def fetch_proxies_from_url(url: str, default_scheme: str = "http") -> list:
    proxies = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        lines = response.text.strip().splitlines()
        for line in lines:
            proxy = line.strip()
            if not proxy:
                continue
            if "://" in proxy:
                proxies.append(proxy)
            else:
                proxies.append(f"{default_scheme}://{proxy}")
        logging.info("Fetched %d proxies from %s", len(proxies), url)
    except Exception as e:
        logging.error("Error fetching proxies from URL %s: %s", url, e)
    return proxies

def test_proxy(proxy: str) -> bool:
    try:
        test_url = "https://api.binance.com/api/v3/time"
        response = requests.get(test_url, proxies={"http": proxy, "https": proxy}, timeout=10, verify=True)
        return response.status_code in range(200, 300)
    except Exception as e:
        if "Connection reset" in str(e):
            logging.debug("Proxy %s failed with connection reset", proxy)
        else:
            logging.debug("Proxy %s failed: %s", proxy, e)
        return False

def test_proxy_speed(proxy: str) -> float:
    test_url = "https://api.binance.com/api/v3/time"  # lightweight endpoint
    try:
        start_time = time.time()
        response = requests.get(test_url, proxies={"http": proxy, "https": proxy}, timeout=10, verify=True)
        response.raise_for_status()
        end_time = time.time()
        return end_time - start_time
    except Exception:
        return float("inf")

def rank_proxies_by_speed(proxies: list) -> list:
    ranked = []
    for proxy in proxies:
        speed = test_proxy_speed(proxy)
        if speed < float("inf"):  # Only include working proxies
            ranked.append((proxy, speed))
    ranked.sort(key=lambda x: x[1])  # Sort by speed
    return ranked

def test_proxies_concurrently(proxies: list, max_workers: int = 50, max_working: int = 10) -> list:
    working = []
    tested = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(test_proxy, proxy): proxy for proxy in proxies}
        try:
            for future in as_completed(futures):
                tested += 1
                proxy = futures[future]
                if future.result():
                    working.append(proxy)
                    if tested % 10 == 0:
                        logging.info("Proxy check: Tested %d | Working: %d", tested, len(working))
                if len(working) >= max_working:
                    break
        finally:
            if len(working) >= max_working:
                for f in futures:
                    if not f.done():
                        f.cancel()
    logging.info("Found %d working proxies (tested %d)", len(working), tested)
    return working[:max_working]

class ProxyPool:
    def __init__(self, proxy_sources, min_working_proxies=10, max_failures=3, refresh_interval=180):
        self.proxy_sources = proxy_sources
        self.min_working_proxies = min_working_proxies
        self.proxies = []
        self.fastest_proxy = None
        self.blacklisted = set()
        self.lock = threading.Lock()
        self.refresh_interval = refresh_interval
        self.proxy_failures = {}  # Track failures per proxy
        self.max_failures = max_failures
        self.proxy_cache_file = "working_proxies.json"
        
        self.load_cached_proxies()
        self.populate_proxy_pool()
        self.start_proxy_checker()
        self.start_fastest_proxy_checker()
        
    def load_cached_proxies(self):
        try:
            if os.path.exists(self.proxy_cache_file):
                with open(self.proxy_cache_file, 'r') as f:
                    cached_proxies = json.load(f)
                    if isinstance(cached_proxies, list) and cached_proxies:
                        self.proxies = cached_proxies
                        logging.info(f"Loaded {len(self.proxies)} proxies from cache")
        except Exception as e:
            logging.error(f"Error loading cached proxies: {e}")
            
    def save_cached_proxies(self):
        try:
            if self.proxies:
                with open(self.proxy_cache_file, 'w') as f:
                    json.dump(self.proxies, f)
                logging.info(f"Saved {len(self.proxies)} proxies to cache")
        except Exception as e:
            logging.error(f"Error saving cached proxies: {e}")
            
    def populate_proxy_pool(self):
        logging.info("Initializing proxy pool...")
        for url in self.proxy_sources:
            raw_proxies = fetch_proxies_from_url(url)
            if not raw_proxies:
                continue
                
            random.shuffle(raw_proxies)
            test_proxies = raw_proxies[:200]
            working_proxies = test_proxies_concurrently(test_proxies, max_working=self.min_working_proxies)
            
            with self.lock:
                self.proxies.extend(working_proxies)
                if len(self.proxies) >= self.min_working_proxies:
                    break
                    
        self.update_fastest_proxy()
        logging.info(f"Proxy pool initialized with {len(self.proxies)} proxies. Fastest: {self.fastest_proxy}")
        self.save_cached_proxies()
            
    def update_fastest_proxy(self, exclude=None):
        with self.lock:
            proxies_to_test = [p for p in self.proxies if p != exclude]
            if not proxies_to_test:
                logging.warning("No proxies available to find fastest")
                return
                
            logging.info("Finding fastest proxy...")
            ranked = rank_proxies_by_speed(proxies_to_test)
            if ranked:
                self.fastest_proxy = ranked[0][0]
                logging.info(f"Fastest proxy is now: {self.fastest_proxy} with speed {ranked[0][1]:.2f}s")
            else:
                logging.warning("No working proxies found during speed test")
                
    def start_fastest_proxy_checker(self):
        def checker_loop():
            while True:
                time.sleep(3600)
                self.update_fastest_proxy()
                
        threading.Thread(target=checker_loop, daemon=True).start()
        
    def check_proxies(self):
        with self.lock:
            initial_count = len(self.proxies)
            working = [p for p in self.proxies if test_proxy(p)]
            self.proxies = working
            
            removed = initial_count - len(self.proxies)
            if removed > 0:
                logging.info(f"Removed {removed} dead proxies, {len(self.proxies)} remaining")
                
            if len(self.proxies) < self.min_working_proxies:
                logging.info(f"Low on proxies ({len(self.proxies)}), getting more...")
                self.populate_proxy_pool()
                
    def start_proxy_checker(self):
        def checker_loop():
            while True:
                time.sleep(self.refresh_interval)
                logging.info("Running periodic proxy check...")
                self.check_proxies()
                
        threading.Thread(target=checker_loop, daemon=True).start()
                
    def mark_proxy_failure(self, proxy):
        with self.lock:
            if proxy not in self.proxy_failures:
                self.proxy_failures[proxy] = 1
            else:
                self.proxy_failures[proxy] += 1
                
            if proxy == self.fastest_proxy and self.proxy_failures[proxy] >= self.max_failures:
                logging.warning(f"Fastest proxy {proxy} failed {self.proxy_failures[proxy]} times, finding new fastest")
                self.update_fastest_proxy(exclude=proxy)
                self.proxies = [p for p in self.proxies if p != proxy]
                
    def reset_proxy_failures(self, proxy):
        with self.lock:
            if proxy in self.proxy_failures:
                self.proxy_failures[proxy] = 0

def fetch_with_retry(url, params=None, proxy_pool=None, max_retries=5, backoff_factor=1.0):
    session = requests.Session()
    retries = 0
    
    while retries < max_retries:
        if proxy_pool and proxy_pool.fastest_proxy:
            proxy_url = proxy_pool.fastest_proxy
            session.proxies = {"http": proxy_url, "https": proxy_url}
            logging.debug(f"Using fastest proxy: {proxy_url}")
        
        try:
            response = session.get(url, params=params, timeout=15, verify=True)
            response.raise_for_status()
            
            if proxy_pool and proxy_pool.fastest_proxy:
                proxy_pool.reset_proxy_failures(proxy_pool.fastest_proxy)
            
            return response.json()
        except requests.exceptions.RequestException as e:
            retries += 1
            logging.warning(f"Request failed (attempt {retries}/{max_retries}): {e}")
            if proxy_pool and proxy_pool.fastest_proxy:
                proxy_pool.mark_proxy_failure(proxy_pool.fastest_proxy)
            if retries >= max_retries:
                logging.error(f"Failed after {max_retries} retries")
                raise
            wait_time = backoff_factor * (2 ** (retries - 1))
            logging.info(f"Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
    return None

def get_perpetual_usdt_symbols(proxy_pool):
    logging.info("Fetching USDT perpetual symbols...")
    for attempt in range(3):
        try:
            data = fetch_with_retry(BINANCE_FUTURES_EXCHANGE_INFO, proxy_pool=proxy_pool)
            symbols = [
                s['symbol'] for s in data['symbols']
                if s['contractType'] == 'PERPETUAL' and 
                s['quoteAsset'] == 'USDT' and 
                s['status'] == 'TRADING'
            ]
            logging.info(f"Successfully fetched {len(symbols)} USDT perpetual symbols")
            return symbols
        except Exception as e:
            logging.error(f"Error fetching symbols (attempt {attempt+1}): {e}")
            time.sleep(5)
    raise RuntimeError("Failed to fetch symbols after multiple attempts")

def fetch_klines(symbol, interval, proxy_pool, limit=CANDLE_LIMIT, max_retries=3):
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    for attempt in range(max_retries):
        try:
            data = fetch_with_retry(BINANCE_FUTURES_KLINES, params=params, proxy_pool=proxy_pool)
            if not data or len(data) < limit:
                logging.warning(f"Received only {len(data) if data else 0}/{limit} klines for {symbol} {interval}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            closes = [float(k[4]) for k in data]
            timestamps = [k[0] for k in data]
            return closes, timestamps
        except Exception as e:
            logging.error(f"Error fetching klines for {symbol} {interval}: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    logging.error(f"Failed to fetch klines for {symbol} {interval} after {max_retries} attempts")
    return [], []

def calculate_rsi_bb(closes):
    closes_np = np.array(closes)
    rsi = talib.RSI(closes_np, timeperiod=RSI_PERIOD)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        rsi, timeperiod=BB_LENGTH, nbdevup=BB_STDDEV, nbdevdn=BB_STDDEV, matype=0
    )
    return rsi, bb_upper, bb_middle, bb_lower

def scan_symbol(symbol, timeframes, proxy_pool):
    results = []
    for timeframe in timeframes:
        for retry in range(3):
            closes, timestamps = fetch_klines(symbol, timeframe, proxy_pool)
            if len(closes) < CANDLE_LIMIT:
                if retry < 2:
                    logging.warning(f"Insufficient data for {symbol} {timeframe}, retrying...")
                    time.sleep(2)
                    continue
                else:
                    logging.warning(f"Not enough klines data for {symbol} {timeframe}, skipping after retries")
                    break
            rsi, bb_upper, bb_middle, bb_lower = calculate_rsi_bb(closes)
            idx = -2
            if idx < -len(closes):
                logging.warning(f"Not enough candles for {symbol} {timeframe} to skip open candle")
                break
            if np.isnan(rsi[idx]) or np.isnan(bb_upper[idx]) or np.isnan(bb_lower[idx]):
                logging.warning(f"NaN values for {symbol} {timeframe}, skipping")
                break
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
            break
    return results

def scan_for_bb_touches(proxy_pool):
    symbols = get_perpetual_usdt_symbols(proxy_pool)
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
            futures = {executor.submit(scan_symbol, symbol, active_timeframes, proxy_pool): symbol for symbol in batch}
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
        proxy_pool = ProxyPool(PROXY_SOURCES, min_working_proxies=10, refresh_interval=180)
        time.sleep(5)  # Wait for proxies to initialize

        if not proxy_pool.fastest_proxy:
            logging.warning("No fastest proxy found yet, proceeding anyway...")

        logging.info("Starting scan process...")
        results = scan_for_bb_touches(proxy_pool)

        logging.info(f"Scan complete, formatting {len(results)} results...")
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
