import requests
import pandas as pd
import numpy as np
import talib
import logging
from datetime import datetime
import os

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

def get_perpetual_usdt_symbols():
    """Fetch all Binance USDT perpetual futures symbols."""
    try:
        response = requests.get(BINANCE_FUTURES_EXCHANGE_INFO)
        response.raise_for_status()
        data = response.json()
        symbols = [
            s['symbol'] for s in data['symbols']
            if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
        ]
        logging.info(f"Found {len(symbols)} USDT perpetual symbols.")
        return symbols
    except Exception as e:
        logging.error(f"Error fetching symbols: {e}")
        return []

def fetch_klines(symbol, interval, limit=CANDLE_LIMIT):
    """Fetch klines (candles) from Binance Futures public API."""
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    try:
        response = requests.get(BINANCE_FUTURES_KLINES, params=params)
        response.raise_for_status()
        data = response.json()
        closes = [float(k[4]) for k in data]  # Close prices
        timestamps = [k[0] for k in data]      # Open time in ms
        return closes, timestamps
    except Exception as e:
        logging.error(f"Error fetching klines for {symbol} {interval}: {e}")
        return None, None

def calculate_rsi_bb(closes):
    """Calculate RSI and Bollinger Bands on RSI."""
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

def scan_for_bb_touches():
    """Scan all symbols and timeframes for RSI BB touches or near-touches."""
    symbols = get_perpetual_usdt_symbols()
    results = []

    for symbol in symbols:
        for timeframe in TIMEFRAMES:
            closes, timestamps = fetch_klines(symbol, timeframe)
            if closes is None or len(closes) < CANDLE_LIMIT:
                logging.warning(f"Not enough data for {symbol} {timeframe}. Skipping.")
                continue

            rsi, bb_upper, bb_middle, bb_lower = calculate_rsi_bb(closes)

            # Use the latest candle index
            idx = -1

            # Validate we have no NaNs for latest values
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
                result = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'rsi': rsi_val,
                    'bb_upper': bb_upper_val,
                    'bb_lower': bb_lower_val,
                    'touch_type': touch_type,
                    'timestamp': timestamp
                }
                results.append(result)
                logging.info(f"Alert: {symbol} on {timeframe} timeframe is touching {touch_type} BB line at {timestamp}")

    return results

def send_telegram_alert(bot_token, chat_id, message):
    """Send alert message to Telegram."""
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

def main():
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram bot token or chat ID not set in environment variables.")


    logging.info("Starting BB touch scanner bot...")

    results = scan_for_bb_touches()

    if not results:
        logging.info("No BB touches detected at this time.")
        return

    for res in results:
        msg = (
            f"*{res['symbol']}* on *{res['timeframe']}* timeframe\n"
            f"RSI: {res['rsi']:.2f}\n"
            f"BB Upper: {res['bb_upper']:.2f}\n"
            f"BB Lower: {res['bb_lower']:.2f}\n"
            f"Touch Type: {res['touch_type']}\n"
            f"Timestamp: {res['timestamp']}"
        )
        send_telegram_alert(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)

if __name__ == "__main__":
    main()
