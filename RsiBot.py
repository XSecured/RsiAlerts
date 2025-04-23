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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
ALTERNATE_PROXY_URLS = [
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/https.txt",
    "https://raw.githubusercontent.com/mertguvencli/http-proxy-list/main/proxy-list/data.txt"
]

# Create a session for direct connections with retry logic
def create_retry_session(retries=3, backoff_factor=0.3):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Global direct session for fallback
DIRECT_SESSION = create_retry_session()

class ProxyManager:
    def __init__(self, proxy_urls, min_pool_size=5, timeout=5, max_failures=3):
        self.proxy_urls = [proxy_urls] if isinstance(proxy_urls, str) else proxy_urls
        self.min_pool_size = min_pool_size
        self.timeout = timeout
        self.max_failures = max_failures
        self.lock = threading.Lock()
        self.proxies = []  # list of dicts: {'proxy': proxy_str, 'failures': 0, 'last_used': timestamp}
        self.proxy_blacklist = set()  # Store failed proxies to avoid retesting them
        self.refresh_needed = False
        self.last_refresh = 0
        self.init_proxies()

    def init_proxies(self):
        """Initialize proxy pool"""
        logging.info("Initializing proxy pool...")
        self.refresh_proxies(blocking=True)
        
    def test_single_proxy(self, proxy):
        """Test a single proxy with Binance API endpoints"""
        if proxy in self.proxy_blacklist:
            return None, None
            
        proxies = {"http": f"http://{proxy}", "https": f"http://{proxy}"}
        try:
            start = time.time()
            r1 = requests.get(
                BINANCE_FUTURES_EXCHANGE_INFO, 
                proxies=proxies, 
                timeout=self.timeout
            )
            if r1.status_code != 200:
                self.proxy_blacklist.add(proxy)
                return None, None
                
            params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1}
            r2 = requests.get(
                BINANCE_FUTURES_KLINES, 
                params=params, 
                proxies=proxies, 
                timeout=self.timeout
            )
            if r2.status_code != 200:
                self.proxy_blacklist.add(proxy)
                return None, None
                
            elapsed = time.time() - start
            logging.info(f"Proxy {proxy} works, response time: {elapsed:.2f}s")
            return proxy, elapsed
        except Exception:
            self.proxy_blacklist.add(proxy)
            return None, None

    def fetch_and_test_proxies(self):
        """Fetch proxies from multiple sources and test them"""
        raw_proxies = []
        
        # Try multiple proxy sources
        for url in self.proxy_urls:
            try:
                logging.info(f"Fetching proxies from {url}...")
                resp = DIRECT_SESSION.get(url, timeout=10)
                resp.raise_for_status()
                new_proxies = [line.strip() for line in resp.text.split('\n') if line.strip()]
                logging.info(f"Found {len(new_proxies)} raw proxies from {url}")
                raw_proxies.extend(new_proxies)
                if len(raw_proxies) > 200:  # We have enough raw proxies to test
                    break
            except Exception as e:
                logging.error(f"Failed to fetch proxy list from {url}: {e}")
        
        # Remove duplicates and blacklisted proxies
        raw_proxies = [p for p in set(raw_proxies) if p not in self.proxy_blacklist]
        random.shuffle(raw_proxies)
        valid = []

        logging.info(f"Testing proxies against Binance endpoints...")
        
        # Test proxies in parallel
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(self.test_single_proxy, proxy): proxy 
                      for proxy in raw_proxies[:100]}  # Test up to 100 at once
            
            for future in as_completed(futures):
                proxy, speed = future.result()
                if proxy:
                    valid.append((proxy, speed))
                    if len(valid) >= self.min_pool_size * 2:
                        # Cancel pending futures once we have enough
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break

        valid.sort(key=lambda x: x[1])  # Sort by speed
        fastest = [p for p, s in valid[:self.min_pool_size * 2]]
        logging.info(f"Selected top {len(fastest)} fastest proxies.")
        return fastest

    def refresh_proxies(self, blocking=False):
        """Refresh proxy pool"""
        with self.lock:
            # Don't refresh too frequently
            if time.time() - self.last_refresh < 30:
                return
            self.last_refresh = time.time()
            
        if blocking:
            self._do_refresh()
        else:
            # Background refresh
            refresh_thread = threading.Thread(target=self._do_refresh)
            refresh_thread.daemon = True
            refresh_thread.start()
    
    def _do_refresh(self):
        """Actual proxy refresh logic"""
        try:
            new_proxies = self.fetch_and_test_proxies()
            with self.lock:
                # Keep existing proxies with low failure count
                existing_good = [p for p in self.proxies if p['failures'] < self.max_failures/2]
                
                # Add new proxies
                new_formatted = [{'proxy': p, 'failures': 0, 'last_used': 0} for p in new_proxies 
                                if p not in [x['proxy'] for x in existing_good]]
                
                # Combine and sort by reliability
                self.proxies = sorted(existing_good + new_formatted, key=lambda x: x['failures'])
                self.refresh_needed = False
                
                logging.info(f"Proxy pool refreshed. Now have {len(self.proxies)} proxies.")
        except Exception as e:
            logging.error(f"Error refreshing proxies: {e}")

    def get_proxy(self):
        """Get a proxy using least-recently-used strategy"""
        with self.lock:
            # Refresh if needed
            if len(self.proxies) < self.min_pool_size:
                if not self.refresh_needed:
                    self.refresh_needed = True
                    self.refresh_proxies(blocking=False)
                    
                # Return None if no proxies available (triggers direct connection)
                if not self.proxies:
                    return None
            
            # Sort by failures and last used time
            self.proxies.sort(key=lambda x: (x['failures'], x['last_used']))
            proxy_info = self.proxies[0]
            
            # Mark as recently used
            proxy_info['last_used'] = time.time()
            
            return proxy_info

    def mark_bad(self, proxy_info):
        """Mark a proxy as having failed"""
        if not proxy_info:
            return
            
        with self.lock:
            if proxy_info not in self.proxies:
                return
                
            proxy_info['failures'] += 1
            
            # Remove if too many failures
            if proxy_info['failures'] >= self.max_failures:
                proxy_addr = proxy_info['proxy']
                logging.warning(f"Removing proxy {proxy_addr} after {self.max_failures} failures")
                self.proxies.remove(proxy_info)
                self.proxy_blacklist.add(proxy_addr)
                
                # Trigger refresh if needed
                if len(self.proxies) < self.min_pool_size and not self.refresh_needed:
                    self.refresh_needed = True
                    self.refresh_proxies(blocking=False)

def make_request(url, params=None, proxy_manager=None, max_retries=5, timeout=10):
    retries = 0
    while retries < max_retries:
        proxy_info = proxy_manager.get_proxy() if proxy_manager else None
        if not proxy_info:
            raise RuntimeError("No working proxies available.")
        proxies = {"http": f"http://{proxy_info['proxy']}", "https": f"http://{proxy_info['proxy']}"}
        try:
            resp = requests.get(url, params=params, proxies=proxies, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logging.error(f"Error with proxy {proxy_info['proxy']}: {e}")
            proxy_manager.mark_bad(proxy_info)
            retries += 1
            time.sleep(0.5 * retries)
    raise RuntimeError(f"Failed to make request after {max_retries} attempts")


def get_perpetual_usdt_symbols(proxy_manager):
    """Fetch USDT perpetual futures from Binance with robust proxy handling"""
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            # Cycle through proxies
            proxy_info = proxy_manager.get_proxy()
            if not proxy_info:
                logging.warning("No available proxies to fetch symbols, waiting and retrying...")
                time.sleep(5)  # Wait before retrying
                continue
            
            logging.info(f"Fetching symbols attempt #{attempt} using proxy: {proxy_info['proxy']}")
            data = make_request(BINANCE_FUTURES_EXCHANGE_INFO, proxy_manager=proxy_manager)
            symbols = [
                s['symbol'] for s in data['symbols']
                if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
            ]
            logging.info(f"Found {len(symbols)} USDT perpetual symbols.")
            if len(symbols) < 10:
                logging.warning(f"Too few symbols found ({len(symbols)}), retrying...")
                time.sleep(2)
                continue
            return symbols
        except Exception as e:
            logging.error(f"Failed to fetch symbols: {e}")
            time.sleep(2)
    raise RuntimeError("Failed to fetch symbols after multiple attempts")


def fetch_klines(symbol, interval, proxy_manager, limit=CANDLE_LIMIT):
    """Fetch klines/candlestick data from Binance"""
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
    """Calculate RSI and Bollinger Bands"""
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
    """Scan a symbol across timeframes for BB touches"""
    results = []
    for timeframe in timeframes:
        closes, timestamps = fetch_klines(symbol, timeframe, proxy_manager)
        if closes is None or len(closes) < CANDLE_LIMIT:
            logging.warning(f"Not enough data for {symbol} {timeframe}. Skipping.")
            continue

        idx = -2  # second last candle to avoid current open candle
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
    """Scan for BB touches with improved batching"""
    symbols = get_perpetual_usdt_symbols(proxy_manager)
    results = []
    total_symbols = len(symbols)
    completed = 0
    
    # Process in batches of 20 symbols
    batch_size = 20
    for i in range(0, total_symbols, batch_size):
        batch = symbols[i:i+batch_size]
        batch_results = []
        
        logging.info(f"Processing batch {i//batch_size + 1}/{(total_symbols+batch_size-1)//batch_size} ({len(batch)} symbols)")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(scan_symbol, symbol, TIMEFRAMES, proxy_manager): symbol for symbol in batch}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    symbol_results = future.result()
                    batch_results.extend(symbol_results)
                except Exception as e:
                    logging.error(f"Error scanning {symbol}: {e}")
                completed += 1
                logging.info(f"Completed {completed}/{total_symbols} symbols")
                
        results.extend(batch_results)
        
        # Small delay between batches
        if i + batch_size < total_symbols:
            time.sleep(1)

    logging.info(f"Scan completed for all {total_symbols} symbols")
    return results

def send_telegram_alert(bot_token, chat_id, message):
    """Send a message to Telegram"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        for attempt in range(3):
            response = DIRECT_SESSION.post(url, data=payload)
            if response.status_code == 200:
                return
            time.sleep(1)
        logging.error(f"Telegram alert failed: {response.text}")
    except Exception as e:
        logging.error(f"Exception sending Telegram alert: {e}")

def format_results_by_timeframe(results):
    """Format results into messages grouped by timeframe"""
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
    """Main entry point for the scanner"""
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram bot token or chat ID not set in environment variables.")

    logging.info("Starting BB touch scanner bot...")

    # Initialize proxy manager with multiple sources
    proxy_manager = ProxyManager(
        proxy_urls=[PROXY_LIST_URL] + ALTERNATE_PROXY_URLS,
        min_pool_size=10,
        timeout=15,
        max_failures=3
    )

    start_time = time.time()
    try:
        results = scan_for_bb_touches_multithreaded(proxy_manager)
        messages = format_results_by_timeframe(results)

        for msg in messages:
            send_telegram_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
            
        elapsed = time.time() - start_time
        logging.info(f"Scan completed in {elapsed:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Fatal error in scanner: {e}")
        # Send error notification to Telegram
        error_msg = f"*⚠️ Scanner Error*\n\nThe scanner encountered an error:\n`{str(e)}`"
        send_telegram_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, error_msg)

if __name__ == "__main__":
    main()
