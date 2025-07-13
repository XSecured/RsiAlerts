import aiohttp
import asyncio
import requests
import pandas as pd
import numpy as np
import talib
import logging
import time
import os
import random
from datetime import datetime, timedelta
import json
import itertools
import glob
import re
from typing import List, Optional
import redis.asyncio as aioredis

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
    '30m': True,
    '1h': True,
    '2h': True,
    '4h': True,
    '1d': True,
    '1w': True,
}

TIMEFRAME_MINUTES_MAP = {
    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
    '12h': 720, '1d': 1440, '1w': 10080
}

# TTL for cached timeframes in seconds (candle duration + buffer)
CACHE_TTL_SECONDS = {
    '4h': 4 * 3600 + 600,    # 4h + 10min buffer
    '1d': 24 * 3600 + 1800,  # 1d + 30min buffer
    '1w': 7 * 24 * 3600 + 3600,  # 1w + 1h buffer
}

CACHED_TFS = {'4h', '1d', '1w'}

# === Cache Manager using Redis ===

class CacheManager:
    def __init__(self, redis_url=None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis = None

    async def initialize(self):
        if self.redis is None:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logging.info(f"Connected to Redis at {self.redis_url}")

    async def close(self):
        if self.redis:
            # Try the new method first, fall back to old method
            if hasattr(self.redis, 'aclose'):
                await self.redis.aclose()
            else:
                await self.redis.close()
            self.redis = None
            logging.info("Redis connection closed")

    def _scan_key(self, timeframe: str, candle_open_iso: str) -> str:
        return f"bb_touch:scan:{timeframe}:{candle_open_iso}"

    def _sent_state_key(self) -> str:
        return "bb_touch:sent_state"

    async def get_scan_results(self, timeframe: str, candle_open_iso: str) -> Optional[list]:
        key = self._scan_key(timeframe, candle_open_iso)
        data = await self.redis.get(key)
        if data:
            try:
                return json.loads(data)
            except Exception as e:
                logging.warning(f"Failed to decode cache data for {key}: {e}")
        return None

    async def set_scan_results(self, timeframe: str, candle_open_iso: str, results: list, ttl_seconds: int):
        key = self._scan_key(timeframe, candle_open_iso)
        try:
            await self.redis.set(key, json.dumps(results), ex=ttl_seconds)
            logging.info(f"Cache set for {key} with TTL {ttl_seconds}s")
        except Exception as e:
            logging.warning(f"Failed to set cache for {key}: {e}")

    async def get_sent_state(self) -> dict:
        key = self._sent_state_key()
        data = await self.redis.get(key)
        if data:
            try:
                return json.loads(data)
            except Exception as e:
                logging.warning(f"Failed to decode sent state: {e}")
        return {}

    async def set_sent_state(self, state: dict):
        key = self._sent_state_key()
        try:
            await self.redis.set(key, json.dumps(state))
            logging.info("Sent state updated")
        except Exception as e:
            logging.warning(f"Failed to update sent state: {e}")

# === UTILITY FUNCTIONS ===

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

# === ASYNC PROXY POOL & REQUESTS ===

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
        max_pool_size: int = 20,
        min_working: int = 5,
        check_interval: int = 600,
        max_failures: int = 3
    ):
        self.sources = sources
        self.max_pool_size = max_pool_size
        self.min_working = min_working
        self.check_interval = check_interval
        self.max_failures = max_failures
        self.blacklisted = set()           # Add immediate blacklist
        self._proxy_lock = asyncio.Lock()  # Add lock for thread safety

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
        """Faster, more aggressive proxy replenishment"""
        async with self._proxy_lock:
            needed = self.max_pool_size - len(self.proxies)
            if needed <= 0:
                return
            
            # Clear blacklist if we're running low on proxies
            if len(self.proxies) < self.min_working:
                logging.warning("Running low on proxies, clearing blacklist")
                self.blacklisted.clear()
                self.failed.clear()
                self.failures.clear()
        
            new_list = []
            for src in self.sources:
                fetched = await fetch_proxies_from_url_async(src)
                # Filter out already blacklisted proxies
                fetched = [p for p in fetched if p not in self.blacklisted]
                new_list.extend(fetched)
                if len(new_list) >= needed * 2:
                    break
                
            working = await test_proxies_concurrently_async(
                new_list, 
                max_workers=100,  # Increase workers for faster testing
                max_working=needed
            )
        
            self.proxies.extend(working)
            self._rebuild_cycle()
            logging.info(f"Pool populated: {len(self.proxies)}/{self.max_pool_size}")

    def _rebuild_cycle(self):
        self.cycle = itertools.cycle(self.proxies)

    async def get_next_proxy(self) -> str:
        """Thread-safe proxy selection with immediate blacklisting"""
        async with self._proxy_lock:
            if not self.proxies:
                return None
                
            # Remove blacklisted proxies from active pool immediately
            self.proxies = [p for p in self.proxies if p not in self.blacklisted]
            if not self.proxies:
                return None
                
            self._rebuild_cycle()
            
            for _ in range(len(self.proxies)):
                proxy = next(self.cycle)
                if proxy not in self.blacklisted and proxy not in self.failed:
                    return proxy
            return None

    async def mark_proxy_failure(self, proxy: str):
        """Immediately blacklist problematic proxies"""
        async with self._proxy_lock:
            c = self.failures.get(proxy, 0) + 1
            self.failures[proxy] = c
            
            logging.warning(f"Proxy {proxy} failure {c}/{self.max_failures}")
            
            if c >= self.max_failures:
                # Immediate blacklisting
                self.blacklisted.add(proxy)
                self.failed.add(proxy)
                
                if proxy in self.proxies:
                    self.proxies.remove(proxy)
                    logging.warning(f"Removed {proxy} from pool")
                    self._rebuild_cycle()

    async def reset_proxy_failures(self, proxy: str):
        """Reset failures and remove from blacklist"""
        async with self._proxy_lock:
            if proxy in self.failures:
                self.failures[proxy] = 0
            if proxy in self.failed:
                self.failed.remove(proxy)
            if proxy in self.blacklisted:
                self.blacklisted.remove(proxy)
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
    """Fixed request function with proper proxy handling"""
    
    attempt = 0
    used_proxies = set()  # Track proxies used in this request
    
    while attempt < max_attempts:
        proxy = await proxy_manager.get_next_proxy()  # Make it async
        
        if proxy is None:
            logging.warning("No working proxies available → refreshing pool…")
            await proxy_manager.populate_to_max()
            proxy = await proxy_manager.get_next_proxy()
            
            if proxy is None:
                logging.error("Still no proxies after repopulation")
                break
                
        # Skip if we already tried this proxy in this request
        if proxy in used_proxies:
            attempt += 1
            continue
            
        used_proxies.add(proxy)
        proxy_url = proxy if proxy.startswith(("http://","https://")) else f"http://{proxy}"
        
        try:
            async with session.get(url, params=params, proxy=proxy_url, timeout=15, ssl=True) as resp:
                if resp.status == 451:
                    logging.warning(f"Proxy {proxy} returned HTTP 451 → marking failure")
                    await proxy_manager.mark_proxy_failure(proxy)  # Make it async
                    attempt += 1
                    continue

                resp.raise_for_status()
                await proxy_manager.reset_proxy_failures(proxy)  # Make it async
                return await resp.json()

        except Exception as e:
            logging.error(f"Request failed with proxy {proxy}: {e}")
            await proxy_manager.mark_proxy_failure(proxy)  # Make it async
            attempt += 1
            
            # Only sleep if we're going to retry
            if attempt < max_attempts:
                await asyncio.sleep(min(2 * attempt, 10))  # Cap the sleep time

    raise RuntimeError(f"Request to {url} failed after {max_attempts} attempts")

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

# === SCANNING FUNCTIONS ===

async def load_cache_async(cache_manager: CacheManager, timeframe: str):
    candle_open = get_latest_candle_open(timeframe).isoformat()
    return await cache_manager.get_scan_results(timeframe, candle_open)

async def save_cache_async(cache_manager: CacheManager, timeframe: str, results: list):
    candle_open = get_latest_candle_open(timeframe).isoformat()
    ttl = CACHE_TTL_SECONDS.get(timeframe, 3600)
    await cache_manager.set_scan_results(timeframe, candle_open, results, ttl)

async def load_sent_state_async(cache_manager: CacheManager):
    return await cache_manager.get_sent_state()

async def save_sent_state_async(cache_manager: CacheManager, state: dict):
    await cache_manager.set_sent_state(state)

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
            direction = None
              
            if MIDDLE_BAND_TOGGLE.get(timeframe, False):
                if not upper_touch and not lower_touch:
                    if abs(rsi_val - bb_middle_val) <= bb_middle_val * MIDDLE_TOUCH_THRESHOLD:
                        middle_touch = True
            
                        # 1) Look for an actual cross
                        prev_rsi       = rsi[idx-1]
                        prev_bb_middle = bb_middle[idx-1]
                        curr_side      = rsi_val  - bb_middle_val  
                        prev_side      = prev_rsi - prev_bb_middle
            
                        if prev_side > 0 and curr_side <= 0:
                            direction = "from above"   
                        elif prev_side < 0 and curr_side >= 0:
                            direction = "from below"   
                        else:
                            # 2) No cross – derive from current side
                            direction = "from above" if curr_side > 0 else "from below"

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
                    'daily_change': daily_change,
                    'direction': direction
                }

                results.append(item)

    return results

async def scan_for_bb_touches_async(proxy_manager, cache_manager):
    symbols = await get_perpetual_usdt_symbols_async(proxy_manager)
    results = []
    batch_size = 70
    active_timeframes = get_active_timeframes()

    cached_timeframes = [tf for tf in CACHED_TFS if tf in active_timeframes]
    uncached_timeframes = [tf for tf in active_timeframes if tf not in cached_timeframes]

    cached_results = {}

    # Load cached results
    for timeframe in cached_timeframes:
        cached_results[timeframe] = await load_cache_async(cache_manager, timeframe)
        if cached_results[timeframe] is not None:
            logging.info(f"[CACHE] Loaded {timeframe} timeframe results from cache")
        else:
            logging.info(f"[CACHE] No valid cache found for {timeframe}, will scan fresh")

    total_symbols = len(symbols)
    fresh_timeframes = set()

    # Scan cached timeframes missing cache
    for timeframe in cached_timeframes:
        if cached_results.get(timeframe) is None:
            logging.info(f"[SCAN] Scanning {timeframe} timeframe fresh data...")
            timeframe_results = []
            for i in range(0, total_symbols, batch_size):
                batch = symbols[i:i + batch_size]
                tasks = [scan_symbol_async(symbol, [timeframe], proxy_manager) for symbol in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in batch_results:
                    if isinstance(res, Exception):
                        logging.error(f"Error scanning {timeframe} timeframe: {res}")
                    else:
                        timeframe_results.extend(res)
            await save_cache_async(cache_manager, timeframe, timeframe_results)
            cached_results[timeframe] = timeframe_results
            fresh_timeframes.add(timeframe)
            logging.info(f"[CACHE] Saved fresh scan results for {timeframe}")

    # Scan uncached timeframes fresh every run
    uncached_results = []
    if uncached_timeframes:
        logging.info(f"[SCAN] Scanning uncached timeframes: {uncached_timeframes}")
        for i in range(0, total_symbols, batch_size):
            batch = symbols[i:i + batch_size]
            tasks = [scan_symbol_async(symbol, uncached_timeframes, proxy_manager) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in batch_results:
                if isinstance(res, Exception):
                    logging.error(f"Error scanning uncached timeframes: {res}")
                else:
                    uncached_results.extend(res)
        fresh_timeframes.update(uncached_timeframes)

    # Combine all results
    for timeframe in cached_timeframes:
        if timeframe in cached_results:
            results.extend(cached_results[timeframe])
    results.extend(uncached_results)

    logging.info(f"[RESULTS] Total BB touches found: {len(results)}")

    return results, fresh_timeframes

# === FORMATTING & TELEGRAM SENDING ===

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
            if item['touch_type'] == 'MIDDLE':
                arrow = "↓" if item['direction'] == "from above" else "↑"
                base += f" ({arrow})"
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

# === MAIN ASYNC ENTRY POINT ===

async def main_async():
    start_time = time.time()
    BOT = os.environ.get("TELEGRAM_BOT_TOKEN")
    CHAT = os.environ.get("TELEGRAM_CHAT_ID")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

    cache_manager = CacheManager(redis_url=REDIS_URL)
    await cache_manager.initialize()

    try:
        logging.info("Starting BB touch scanner bot…")

        # Load last sent state and compute candle opens
        sent_state = await load_sent_state_async(cache_manager)

        active = set(get_active_timeframes())
        now_opens = {tf: get_latest_candle_open(tf).isoformat() for tf in CACHED_TFS if tf in active}
        to_send_cached = {tf for tf, iso in now_opens.items() if sent_state.get(tf) != iso}

        logging.info(f"Always-fresh TFs (every run): {sorted(active - CACHED_TFS)}")
        logging.info(f"Cached TFs sending only now: {sorted(to_send_cached)}")

        # Initialize proxy pool (your existing code)
        proxy_pool = AsyncProxyPool(
            sources=PROXY_SOURCES,
            max_pool_size=25,
            min_working=5,
            check_interval=600,
            max_failures=3
        )
        await proxy_pool.initialize()

        # Scan for BB touches
        results, fresh_timeframes = await scan_for_bb_touches_async(proxy_pool, cache_manager)

        # Format messages
        initial_cached = [tf for tf in CACHED_TFS if tf in active and await load_cache_async(cache_manager, tf) is not None]
        messages = format_results_by_timeframe(results, cached_timeframes_used=initial_cached)

        # Filter messages to send
        allowed_tfs = (active - CACHED_TFS).union(to_send_cached)

        def msg_timeframe(msg: str) -> str:
            m = re.search(r"BB Touches on (\S+) Timeframe", msg)
            return m.group(1) if m else ""

        to_send = [m for m in messages if msg_timeframe(m) in allowed_tfs]

        # Dispatch alerts
        if not to_send:
            logging.info("No BB-touch alerts to send this run.")
        else:
            for i, msg in enumerate(to_send, 1):
                logging.info(f"Sending message {i}/{len(to_send)}")
                for chunk in split_message(msg):
                    send_telegram_alert(BOT, CHAT, chunk)

        # Update sent state for cached TFs sent
        for tf in to_send_cached:
            sent_state[tf] = now_opens[tf]
        await save_sent_state_async(cache_manager, sent_state)

        logging.info(f"Bot run completed in {time.time()-start_time:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"Fatal error after {elapsed:.1f}s: {e}")
        if BOT and CHAT:
            error_msg = f"*⚠️ Scanner Error*\n`{e}` after {elapsed:.1f}s"
            send_telegram_alert(BOT, CHAT, error_msg)

    finally:
        await cache_manager.close()

if __name__ == "__main__":
    asyncio.run(main_async())
