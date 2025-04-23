import requests
import pandas as pd
import numpy as np
import talib
import logging
from datetime import datetime
import os
import threading
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

BINANCE_FUTURES_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"

TIMEFRAMES = ['4h', '1d', '1w']
CANDLE_LIMIT = 50

UPPER_TOUCH_THRESHOLD = 0.01  # 1%
LOWER_TOUCH_THRESHOLD = 0.01  # 1%

RSI_PERIOD = 14
BB_LENGTH = 34
BB_STDDEV = 2

PROXY_LIST_URL = "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"

class ProxyManager:
    def __init__(self, proxy_url, max_proxies=20, timeout=5, max_failures=10):
        self.proxy_url = proxy_url
        self.max_proxies = max_proxies
        self.timeout = timeout
        self.max_failures = max_failures
        self.lock = threading.Lock()
        self.failure_counts = {}
        self.proxies = []
        self.index = 0
        self.refresh_proxies()

    def test_single_proxy(self, proxy):
        """Test proxy against BOTH Binance endpoints we'll be using"""
        proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
        try:
            # Test 1: Exchange Info endpoint
            start = time.time()
            r1 = requests.get(BINANCE_FUTURES_EXCHANGE_INFO, proxies=proxies, timeout=self.timeout)
            if r1.status_code != 200:
                return None, None
                
            # Test 2: Klines endpoint with BTC/USDT
            params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1}
            r2 = requests.get(BINANCE_FUTURES_KLINES, params=params, proxies=proxies, timeout=self.timeout)
            if r2.status_code != 200:
                return None, None
                
            elapsed = time.time() - start
            logging.info(f"Proxy {proxy} works with ALL endpoints, response time: {elapsed:.2f}s")
            return proxy, elapsed
        except Exception:
            return None, None

    def fetch_and_test_proxies(self):
        logging.info("Fetching proxy list...")
        try:
            resp = requests.get(self.proxy_url, timeout=10)
            resp.raise_for_status()
            raw_proxies = [line.strip() for line in resp.text.split('\n') if line.strip()]
        except Exception as e:
            logging.error(f"Failed to fetch proxy list: {e}")
            return []

        random.shuffle(raw_proxies)
        valid = []

        logging.info(f"Testing proxies against Binance endpoints (multithreaded)...")

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {executor.submit(self.test_single_proxy, proxy): proxy for proxy in raw_proxies[:200]}
            for future in as_completed(futures):
                proxy, speed = future.result()
                if proxy:
                    valid.append((proxy, speed))
                    if len(valid) >= self.max_proxies * 3:
                        break

        # Sort by speed ascending and keep top max_proxies
        valid.sort(key=lambda x: x[1])
        fastest = [p for p, s in valid[:self.max_proxies]]
        logging.info(f"Selected top {len(fastest)} fastest proxies that work with ALL Binance endpoints.")
        return fastest

    def refresh_proxies(self):
        with self.lock:
            self.proxies = self.fetch_and_test_proxies()
            self.failure_counts = {p: 0 for p in self.proxies}
            self.index = 0
            if not self.proxies:
                logging.error("No working proxies found after refresh!")

    def get_proxy(self):
        with self.lock:
            if not self.proxies:
                logging.error("Proxy list empty, refreshing proxies...")
                self.refresh_proxies()
                if not self.proxies:
                    raise RuntimeError("No working proxies available after refresh.")
            proxy = self.proxies[self.index % len(self.proxies)]
            self.index += 1
            return {"http": f"http://{proxy}", "https": f"http://{proxy}"}

    def mark_bad(self, proxy):
        if not proxy:
            return
        with self.lock:
            p = proxy["http"].replace("http://", "")
            if p in self.failure_counts:
                self.failure_counts[p] += 1
                if self.failure_counts[p] >= self.max_failures:
                    if p in self.proxies:
                        self.proxies.remove(p)
                        logging.warning(f"Removed bad proxy {p} after {self.max_failures} failures")
                    del self.failure_counts[p]
            else:
                logging.warning(f"Marking unknown proxy {p} as bad")
            if not self.proxies:
                logging.error("All proxies removed! Refreshing proxy list...")
                self.refresh_proxies()

def make_request(url, params=None, proxy_manager=None, timeout=8, retries=3):
    for attempt in range(retries):
        proxy = proxy_manager.get_proxy() if proxy_manager else None
        try:
            resp = requests.get(url, params=params, proxies=proxy, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if proxy_manager and proxy:
                logging.error(f"Error with proxy {proxy.get('https')}: {e}")
                proxy_manager.mark_bad(proxy)
            else:
                logging.error(f"Direct connection error: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(1)
    return None

def get_perpetual_usdt_symbols(proxy_manager):
    """Keep trying until we successfully fetch symbols - never give up!"""
    backoff_time = 1
    max_backoff = 30
    attempt = 1
    
    while True:  # Infinite loop - we'll only exit when successful
        try:
            logging.info(f"Fetching symbols attempt #{attempt}...")
            data = make_request(BINANCE_FUTURES_EXCHANGE_INFO, proxy_manager=proxy_manager)
            symbols = [
                s['symbol'] for s in data['symbols']
                if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
            ]
            symbol_count = len(symbols)
            logging.info(f"Found {symbol_count} USDT perpetual symbols.")
            
            # Extra validation - make sure we got a reasonable amount of symbols
            if symbol_count < 10:
                logging.warning(f"Only found {symbol_count} symbols, which seems too few. Retrying...")
                attempt += 1
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff)
                continue
                
            return symbols
        except Exception as e:
            logging.error(f"Failed to fetch symbols (attempt #{attempt}): {e}")
            # If all proxies are bad, this will trigger a refresh
            attempt += 1
            time.sleep(backoff_time)
            backoff_time = min(backoff_time * 2, max_backoff)

def fetch_klines(symbol, interval, proxy_manager, limit=CANDLE_LIMIT):
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        data = make_request(BINANCE_FUTURES_KLINES, params=params, proxy_manager=proxy_manager)
        closes = [float(k[4]) for k in data]
        timestamps = [k[0] for k in data]
        return closes, timestamps
    except Exception as e:
        logging.error(f"Error fetching klines for {symbol} {interval}: {e}")
        return None, None

def calculate_rsi_bb(closes):
    closes_np = np.array(closes)
    rsi = talib.RSI(closes_np, timeperiod=RSI_PERIOD)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        rsi,
        timeperiod=BB_LENGTH,
        nbdevup=BB_STDDEV,
        nbdevdn=BB_STDDEV,
        matype=0  # SMA
    )
    return rsi, bb_upper, bb_middle, bb_lower

def scan_symbol(symbol, timeframes, proxy_manager):
    results = []
    for timeframe in timeframes:
        closes, timestamps = fetch_klines(symbol, timeframe, proxy_manager)
        if closes is None or len(closes) < CANDLE_LIMIT:
            logging.warning(f"Not enough data for {symbol} {timeframe}. Skipping.")
            continue

        idx = -2  # Use second last candle to avoid current open candle
        if idx < -len(closes):
            logging.warning(f"Not enough candles for {symbol} {timeframe} to skip open candle. Skipping.")
            continue

        rsi, bb_upper, bb_middle, bb_lower = calculate_rsi_bb(closes)
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
            logging.info(f"Alert: {symbol} on {timeframe} timeframe touching {touch_type} BB line at {timestamp}")
    return results

def scan_for_bb_touches_multithreaded(proxy_manager):
    symbols = get_perpetual_usdt_symbols(proxy_manager)
    results = []
    total_symbols = len(symbols)
    completed = 0

    # Use 8 threads for scanning
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(scan_symbol, symbol, TIMEFRAMES, proxy_manager): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                symbol_results = future.result()
                results.extend(symbol_results)
            except Exception as e:
                logging.error(f"Error in thread scanning {symbol}: {e}")
            completed += 1
            logging.info(f"Completed {completed}/{total_symbols} symbols")

    logging.info(f"Scan completed for all {total_symbols} symbols")
    return results

def send_telegram_alert(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        for attempt in range(3):
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                return
            time.sleep(1)
        logging.error(f"Telegram alert failed: {response.text}")
    except Exception as e:
        logging.error(f"Exception sending Telegram alert: {e}")

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
                lines.append("")  # spacing
            lines.append("*⬇️ LOWER BB Touches:*")
            for item in sorted(lower_touches, key=lambda x: x['symbol']):
                lines.append(f"• *{item['symbol']}* - RSI: {item['rsi']:.2f}")

        messages.append(header + "\n" + "\n".join(lines))

    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    messages = [m + f"\n\n_Report generated at {timestamp}_" for m in messages]

    return messages

def main():
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram bot token or chat ID not set in environment variables.")

    logging.info("Starting BB touch scanner bot...")

    proxy_manager = ProxyManager(
        proxy_url=PROXY_LIST_URL,
        max_proxies=20,
        timeout=5,
        max_failures=10
    )

    results = scan_for_bb_touches_multithreaded(proxy_manager)

    messages = format_results_by_timeframe(results)

    for msg in messages:
        send_telegram_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)

if __name__ == "__main__":
    main()
