import requests
import pandas as pd
import numpy as np
import talib
import logging
from datetime import datetime
import os
import threading
import random
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
CANDLE_LIMIT = 52

UPPER_TOUCH_THRESHOLD = 0.01  # 1%
LOWER_TOUCH_THRESHOLD = 0.01  # 1%

RSI_PERIOD = 14
BB_LENGTH = 34
BB_STDDEV = 2

PROXY_LIST_URL = "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"

class ProxyManager:
    def __init__(self, proxy_url, test_url, test_params, max_proxies=20, timeout=5):
        self.proxy_url = proxy_url
        self.test_url = test_url
        self.test_params = test_params
        self.max_proxies = max_proxies
        self.timeout = timeout
        self.lock = threading.Lock()
        self.proxies = self.fetch_and_test_proxies()
        self.index = 0
        if not self.proxies:
            logging.error("No working proxies found. Exiting.")
            raise RuntimeError("No working proxies available.")

    def test_single_proxy(self, proxy):
        proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
        try:
            import time
            start = time.time()
            r = requests.get(self.test_url, params=self.test_params, proxies=proxies, timeout=self.timeout)
            elapsed = time.time() - start
            if r.status_code == 200:
                logging.info(f"Proxy {proxy} works, response time: {elapsed:.2f}s")
                return proxy, elapsed
        except Exception:
            pass
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

        logging.info(f"Testing proxies to find up to {self.max_proxies} fastest working ones (multithreaded)...")

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {executor.submit(self.test_single_proxy, proxy): proxy for proxy in raw_proxies}
            for future in as_completed(futures):
                proxy, speed = future.result()
                if proxy:
                    valid.append((proxy, speed))
                    if len(valid) >= self.max_proxies * 5:  # test more to pick fastest 5 later
                        break

        # Sort by speed ascending and keep only top 5
        valid.sort(key=lambda x: x[1])
        fastest = [p for p, s in valid[:5]]
        logging.info(f"Selected top 5 fastest proxies: {fastest}")
        return fastest

    def get_proxy(self):
        with self.lock:
            if not self.proxies:
                raise RuntimeError("No working proxies available.")
            proxy = self.proxies[self.index % len(self.proxies)]
            self.index += 1
            return {"http": f"http://{proxy}", "https": f"http://{proxy}"}

    def mark_bad(self, proxy):
        with self.lock:
            p = proxy["http"].replace("http://", "")
            if p in self.proxies:
                self.proxies.remove(p)
                logging.warning(f"Removed bad proxy: {p}")
            if not self.proxies:
                logging.error("All proxies removed! No working proxies left.")
                raise RuntimeError("No working proxies left.")

def get_perpetual_usdt_symbols(proxy_manager):
    for attempt in range(5):
        proxy = proxy_manager.get_proxy()
        try:
            logging.info(f"Fetching symbols using proxy {proxy['https']}")
            resp = requests.get(BINANCE_FUTURES_EXCHANGE_INFO, proxies=proxy, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            symbols = [
                s['symbol'] for s in data['symbols']
                if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
            ]
            logging.info(f"Found {len(symbols)} USDT perpetual symbols.")
            return symbols
        except Exception as e:
            logging.error(f"Error fetching symbols with proxy {proxy['https']}: {e}")
            proxy_manager.mark_bad(proxy)
    logging.error("Failed to fetch symbols after multiple attempts.")
    return []

def fetch_klines(symbol, interval, proxy_manager, limit=CANDLE_LIMIT):
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    for attempt in range(3):
        proxy = proxy_manager.get_proxy()
        try:
            resp = requests.get(BINANCE_FUTURES_KLINES, params=params, proxies=proxy, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            closes = [float(k[4]) for k in data]
            timestamps = [k[0] for k in data]
            return closes, timestamps
        except Exception as e:
            logging.error(f"Error fetching klines for {symbol} {interval} with proxy {proxy['https']}: {e}")
            proxy_manager.mark_bad(proxy)
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

        # Use second last candle to avoid current open candle
        idx = -2
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
    if not symbols:
        logging.error("No symbols to scan.")
        return []
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(scan_symbol, symbol, TIMEFRAMES, proxy_manager): symbol for symbol in symbols}
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as e:
                logging.error(f"Error in thread scanning symbol: {e}")
    return results

def send_telegram_alert(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            logging.error(f"Telegram alert failed: {response.text}")
    except Exception as e:
        logging.error(f"Exception sending Telegram alert: {e}")

def format_results_by_timeframe(results):
    # Group results by timeframe
    grouped = {}
    for r in results:
        grouped.setdefault(r['timeframe'], []).append(r)

    messages = []
    for timeframe, items in grouped.items():
        header = f"*BB Touches on {timeframe} timeframe*\n"
        lines = []
        for item in items:
            line = (
                f"- *{item['symbol']}* touching *{item['touch_type']}* BB\n"
                f"  RSI: {item['rsi']:.2f}, BB Upper: {item['bb_upper']:.2f}, BB Lower: {item['bb_lower']:.2f}\n"
                f"  Time: {item['timestamp']}"
            )
            lines.append(line)
        messages.append(header + "\n".join(lines))
    return messages

def main():
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram bot token or chat ID not set in environment variables.")

    logging.info("Starting BB touch scanner bot...")

    proxy_manager = ProxyManager(
        proxy_url=PROXY_LIST_URL,
        test_url=BINANCE_FUTURES_KLINES,
        test_params={"symbol": "BTCUSDT", "interval": "1d", "limit": 1},
        max_proxies=20,
        timeout=5
    )

    results = scan_for_bb_touches_multithreaded(proxy_manager)

    if not results:
        logging.info("No BB touches detected at this time.")
        return

    messages = format_results_by_timeframe(results)

    for msg in messages:
        send_telegram_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)

if __name__ == "__main__":
    main()
