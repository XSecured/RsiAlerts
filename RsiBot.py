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

# Configure logging with more detail
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

# Multiple proxy sources for redundancy
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"
]

# Timeframe toggles
TIMEFRAMES_TOGGLE = {
    '3m': True,
    '5m': True
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
    def __init__(self, proxy_sources, min_working_proxies=3):
        self.proxy_sources = proxy_sources
        self.min_working_proxies = min_working_proxies
        self.proxies = []  # Format: [{'proxy': proxy_str, 'failures': 0, 'speed': response_time}]
        self.blacklisted = set()
        self.lock = threading.Lock()
        self.refresh_in_progress = False
        self.selected_proxy = None  # Selected proxy for entire run
        
        # Initialize proxy pool
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
        
        with ThreadPoolExecutor(max_workers=20) as executor:
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
        
        proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
        
        try:
            start = time.time()
            response = requests.get(BINANCE_FUTURES_EXCHANGE_INFO, proxies=proxies, timeout=15)
            if response.status_code != 200:
                self.blacklisted.add(proxy)
                return None, None
            elapsed = time.time() - start
            return proxy, elapsed
        except Exception:
            self.blacklisted.add(proxy)
            return None, None
    
    def get_proxy(self):
        with self.lock:
            if self.selected_proxy:
                logging.debug(f"Using selected proxy: {self.selected_proxy['proxy']}")
                return self.selected_proxy
            
            if not self.proxies:
                raise RuntimeError("No working proxies available. Cannot proceed.")
            
            self.proxies.sort(key=lambda x: (x['failures'], x['speed']))
            return self.proxies[0]
    
    def mark_success(self, proxy_info):
        with self.lock:
            if not self.selected_proxy:
                self.selected_proxy = proxy_info
                logging.info(f"Selected proxy for entire run: {proxy_info['proxy']}")
    
    def mark_failure(self, proxy_info):
        with self.lock:
            if self.selected_proxy and proxy_info['proxy'] == self.selected_proxy['proxy']:
                logging.warning(f"Selected proxy {proxy_info['proxy']} failed, clearing selection")
                self.selected_proxy = None
            
            for p in self.proxies:
                if p['proxy'] == proxy_info['proxy']:
                    p['failures'] += 1
                    logging.warning(f"Proxy {p['proxy']} failure count now {p['failures']}")
                    if p['failures'] >= 3:
                        logging.warning(f"Removing failed proxy {p['proxy']}")
                        self.proxies.remove(p)
                        self.blacklisted.add(p['proxy'])
                        if len(self.proxies) < self.min_working_proxies and not self.refresh_in_progress:
                            threading.Thread(target=self._refresh_proxies, daemon=True).start()
                    break

def make_request(url, params=None, proxy_manager=None, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            proxy_info = proxy_manager.get_proxy()
            proxy_str = proxy_info['proxy']
            proxies = {"http": f"http://{proxy_str}", "https": f"http://{proxy_str}"}
            
            endpoint = url.split('/')[-1] if '/' in url else url
            logging.info(f"Request to {endpoint}: using proxy {proxy_str} (attempt {attempt+1}/{max_attempts})")
            
            timeout = min(15, max(5, proxy_info['speed'] * 2))
            resp = requests.get(url, params=params, proxies=proxies, timeout=timeout, verify=False)
            resp.raise_for_status()
            
            proxy_manager.mark_success(proxy_info)
            
            logging.info(f"Request successful: {endpoint}")
            return resp.json()
        except Exception as e:
            logging.error(f"Request failed: {str(e)}")
            if 'proxy_info' in locals():
                proxy_manager.mark_failure(proxy_info)
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Request failed after {max_attempts} attempts")
            wait_time = 2 * (attempt + 1)
            logging.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

def get_perpetual_usdt_symbols(proxy_manager):
    logging.info("Starting to fetch symbols list...")
    for attempt in range(5):
        try:
            logging.info(f"Fetching symbols (attempt {attempt+1}/5)...")
            data = make_request(BINANCE_FUTURES_EXCHANGE_INFO, proxy_manager=proxy_manager)
            symbols = [
                s['symbol'] for s in data['symbols']
                if s['contractType'] == 'PERPETUAL' and 
                s['quoteAsset'] == 'USDT' and 
                s['status'] == 'TRADING'
            ]
            logging.info(f"Successfully fetched {len(symbols)} USDT perpetual symbols")
            if len(symbols) < 10:
                logging.warning(f"Too few symbols found ({len(symbols)}), retrying...")
                time.sleep(3)
                continue
            return symbols
        except Exception as e:
            logging.error(f"Failed to fetch symbols: {str(e)}")
            time.sleep(5)
    raise RuntimeError("Failed to fetch symbols after multiple attempts")

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
        logging.info("Initializing proxy manager...")
        proxy_manager = ProxyManager(PROXY_SOURCES, min_working_proxies=3)
        logging.info("Starting scan process...")
        results = scan_for_bb_touches(proxy_manager)
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
