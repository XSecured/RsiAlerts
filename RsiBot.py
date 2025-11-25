import asyncio
import json
import logging
import os
import random
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Set, Optional, Tuple, Any
from itertools import cycle

import aiohttp
import numpy as np
import talib
import redis.asyncio as aioredis

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

@dataclass
class Config:
    # System
    MAX_CONCURRENCY: int = 50
    REQUEST_TIMEOUT: int = 5
    MAX_RETRIES: int = 3
    
    # Redis & Cache
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL_MAP: Dict[str, int] = field(default_factory=lambda: {
        '4h': 14400 + 600, '1d': 86400 + 1800, '1w': 604800 + 3600
    })
    
    # Telegram
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    PROXY_URL: str = "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"

    # Trading Params
    RSI_PERIOD: int = 14
    BB_LENGTH: int = 34
    BB_STDDEV: float = 2.0
    CANDLE_LIMIT: int = 60
    MIN_CANDLES: int = 36  # Safe buffer for Talib

    UPPER_TOUCH_THRESHOLD: float = 0.02
    LOWER_TOUCH_THRESHOLD: float = 0.02
    MIDDLE_TOUCH_THRESHOLD: float = 0.035

    # Ignore List
    IGNORED_SYMBOLS: Set[str] = field(default_factory=lambda: {
        "USDPUSDT", "USD1USDT", "TUSDUSDT", "AEURUSDT", "USDCUSDT", "EURUSDT"
    })

CONFIG = Config()

# Timeframe Mapping
TIMEFRAME_MINUTES = {
    '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, 
    '1d': 1440, '1w': 10080
}

# Which TFs to scan
ACTIVE_TFS = ['15m', '30m', '1h', '2h', '4h', '1d', '1w']
# Which TFs support Middle Band checks
MIDDLE_BAND_TFS = ['1h', '2h', '4h', '1d', '1w']
# Which TFs are "High Timeframe" (Cached/Once-per-candle)
CACHED_TFS = {'4h', '1d', '1w'}

# ==========================================
# DATA MODELS
# ==========================================

@dataclass
class TouchHit:
    symbol: str
    exchange: str 
    market: str   
    timeframe: str
    rsi: float
    touch_type: str # UPPER, LOWER, MIDDLE
    direction: str = "" # "from above", "from below"
    hot: bool = False

# ==========================================
# ROBUST PROXY POOL (IDENTICAL TO MARKET SCAN)
# ==========================================

class AsyncProxyPool:
    def __init__(self, max_pool_size=25):
        self.proxies: List[str] = []
        self.max_pool_size = max_pool_size
        self.iterator = None
        self._lock = asyncio.Lock()

    async def populate(self, url: str, session: aiohttp.ClientSession):
        if not url: return
        raw = []
        try:
            logging.info(f"📥 Fetching proxies from {url}...")
            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    for line in text.splitlines():
                        p = line.strip()
                        if p: raw.append(p if "://" in p else f"http://{p}")
        except Exception as e:
            logging.error(f"❌ Proxy fetch failed: {e}")
            return

        logging.info(f"🔎 Validating {len(raw)} proxies...")
        self.proxies = []
        random.shuffle(raw)
        
        sem = asyncio.Semaphore(200)
        async def protected_test(p):
            async with sem: return await self._test_proxy(p, session)

        tasks = [asyncio.create_task(protected_test(p)) for p in raw]
        for future in asyncio.as_completed(tasks):
            try:
                proxy, is_good = await future
                if is_good:
                    self.proxies.append(proxy)
                    if len(self.proxies) >= self.max_pool_size: break
            except: pass
        
        for t in tasks:
            if not t.done(): t.cancel()
        await asyncio.sleep(0.1)
        
        if self.proxies:
            self.iterator = cycle(self.proxies)
            logging.info(f"✅ Proxy Pool Ready: {len(self.proxies)} proxies.")
        else:
            logging.error("❌ NO WORKING PROXIES FOUND!")

    async def _test_proxy(self, proxy: str, session: aiohttp.ClientSession) -> Tuple[str, bool]:
        try:
            async with session.get("https://api.binance.com/api/v3/time", proxy=proxy, timeout=3) as resp:
                return proxy, resp.status == 200
        except: return proxy, False

    async def get_proxy(self) -> Optional[str]:
        if not self.proxies: return None
        async with self._lock: return next(self.iterator)

# ==========================================
# REDIS CACHE MANAGER
# ==========================================

class CacheManager:
    def __init__(self):
        self.redis = None

    async def init(self):
        try:
            self.redis = await aioredis.from_url(CONFIG.REDIS_URL, decode_responses=True)
            await self.redis.ping()
            logging.info("✅ Redis Connected")
        except:
            logging.warning("⚠️ Redis failed. Caching disabled.")
            self.redis = None

    async def close(self):
        if self.redis: await self.redis.close()

    async def get_sent_state(self) -> Dict[str, str]:
        if not self.redis: return {}
        try:
            val = await self.redis.get("bb_bot:sent_state")
            return json.loads(val) if val else {}
        except: return {}

    async def save_sent_state(self, state: Dict[str, str]):
        if not self.redis: return
        try:
            await self.redis.set("bb_bot:sent_state", json.dumps(state))
        except: pass

# ==========================================
# EXCHANGE CLIENTS & LOGIC
# ==========================================

class ExchangeClient:
    def __init__(self, session: aiohttp.ClientSession, proxy_pool: AsyncProxyPool):
        self.session = session
        self.proxies = proxy_pool
        limit = CONFIG.MAX_CONCURRENCY if proxy_pool.proxies else 5
        self.sem = asyncio.Semaphore(limit)

    async def _request(self, url: str, params: dict = None) -> Any:
        for attempt in range(CONFIG.MAX_RETRIES):
            proxy = await self.proxies.get_proxy()
            try:
                async with self.sem:
                    async with self.session.get(url, params=params, proxy=proxy, timeout=CONFIG.REQUEST_TIMEOUT) as resp:
                        if resp.status == 200: return await resp.json()
                        elif resp.status == 429: await asyncio.sleep(5)
            except: pass
            await asyncio.sleep(0.5 * attempt)
        return None

class BinanceClient(ExchangeClient):
    async def get_perp_symbols(self) -> List[str]:
        data = await self._request('https://fapi.binance.com/fapi/v1/exchangeInfo')
        if not data: return []
        return [s['symbol'] for s in data['symbols'] if s.get('contractType') == 'PERPETUAL' and s['status'] == 'TRADING' and s.get('quoteAsset') == 'USDT']
    
    async def get_spot_symbols(self) -> List[str]:
        data = await self._request('https://api.binance.com/api/v3/exchangeInfo')
        if not data: return []
        return [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING' and s.get('quoteAsset') == 'USDT']

    async def fetch_closes_volatility(self, symbol: str, market: str) -> List[float]:
        # Fetch 1h candles for volatility (25 limit)
        base = 'https://api.binance.com/api/v3/klines' if market == "spot" else 'https://fapi.binance.com/fapi/v1/klines'
        data = await self._request(base, {'symbol': symbol, 'interval': '1h', 'limit': 25})
        if not data: return []
        try: return [float(c[4]) for c in data]
        except: return []

    async def fetch_closes(self, symbol: str, interval: str, market: str) -> List[float]:
        base = 'https://api.binance.com/api/v3/klines' if market == "spot" else 'https://fapi.binance.com/fapi/v1/klines'
        data = await self._request(base, {'symbol': symbol, 'interval': interval, 'limit': CONFIG.CANDLE_LIMIT})
        if not data: return []
        try: return [float(c[4]) for c in data]
        except: return []

class BybitClient(ExchangeClient):
    async def get_perp_symbols(self) -> List[str]:
        data = await self._request('https://api.bybit.com/v5/market/instruments-info', {'category': 'linear'})
        if not data: return []
        return [s['symbol'] for s in data['result']['list'] if s['status'] == 'Trading' and s['quoteCoin'] == 'USDT']

    async def get_spot_symbols(self) -> List[str]:
        data = await self._request('https://api.bybit.com/v5/market/instruments-info', {'category': 'spot'})
        if not data: return []
        return [s['symbol'] for s in data['result']['list'] if s['status'] == 'Trading' and s['quoteCoin'] == 'USDT']

    async def fetch_closes(self, symbol: str, interval: str, market: str) -> List[float]:
        url = 'https://api.bybit.com/v5/market/kline'
        cat = 'linear' if market == 'perp' else 'spot'
        # Bybit interval mapping
        b_int = {"15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240", "1d": "D", "1w": "W"}.get(interval, "D")
        data = await self._request(url, {'category': cat, 'symbol': symbol, 'interval': b_int, 'limit': CONFIG.CANDLE_LIMIT})
        if not data: return []
        raw = data.get('result', {}).get('list', [])
        if not raw: return []
        closes = [float(c[4]) for c in raw]
        return closes[::-1] # Oldest -> Newest

# ==========================================
# CORE LOGIC
# ==========================================

def get_candle_open_iso(tf: str) -> str:
    now = datetime.now(timezone.utc)
    mins = TIMEFRAME_MINUTES.get(tf, 60)
    total = now.hour * 60 + now.minute
    floored = (total // mins) * mins
    open_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=floored)
    return open_time.isoformat()

def calculate_volatility(closes: List[float]) -> float:
    if len(closes) < 24: return 0.0
    returns = []
    for i in range(1, len(closes)):
        if closes[i-1] != 0:
            returns.append((closes[i] - closes[i-1]) / closes[i-1] * 100)
    return np.std(returns) if returns else 0.0

def check_bb_rsi(closes: List[float], tf: str) -> Tuple[Optional[str], Optional[str], float]:
    # Returns: (touch_type, direction, rsi_val)
    if len(closes) < CONFIG.MIN_CANDLES: return None, None, 0.0
    
    np_c = np.array(closes, dtype=float)
    rsi = talib.RSI(np_c, timeperiod=CONFIG.RSI_PERIOD)
    upper, mid, lower = talib.BBANDS(rsi, timeperiod=CONFIG.BB_LENGTH, nbdevup=CONFIG.BB_STDDEV, nbdevdn=CONFIG.BB_STDDEV, matype=0)
    
    idx = -2 # Closed candle
    if np.isnan(rsi[idx]) or np.isnan(upper[idx]): return None, None, 0.0
    
    curr_rsi = rsi[idx]
    curr_up, curr_mid, curr_low = upper[idx], mid[idx], lower[idx]
    
    # Upper
    if curr_rsi >= curr_up * (1 - CONFIG.UPPER_TOUCH_THRESHOLD):
        return "UPPER", None, curr_rsi
    
    # Lower
    if curr_rsi <= curr_low * (1 + CONFIG.LOWER_TOUCH_THRESHOLD):
        return "LOWER", None, curr_rsi
        
    # Middle
    if tf in MIDDLE_BAND_TFS:
        if abs(curr_rsi - curr_mid) <= (curr_mid * CONFIG.MIDDLE_TOUCH_THRESHOLD):
            # Direction logic
            prev_rsi = rsi[idx-1]
            prev_mid = mid[idx-1]
            prev_diff = prev_rsi - prev_mid
            curr_diff = curr_rsi - curr_mid
            
            if prev_diff > 0 and curr_diff <= 0: direction = "from above"
            elif prev_diff < 0 and curr_diff >= 0: direction = "from below"
            else: direction = "from above" if curr_diff > 0 else "from below"
            
            return "MIDDLE", direction, curr_rsi
            
    return None, None, 0.0

# ==========================================
# MAIN BOT
# ==========================================

class RsiBot:
    def __init__(self):
        self.cache = CacheManager()
        self.proxies = AsyncProxyPool()
        
    async def send_report(self, hits: List[TouchHit]):
        if not hits: 
            logging.info("No hits to report.")
            return
        
        # Grouping
        grouped = {}
        for h in hits:
            grouped.setdefault(h.timeframe, {}).setdefault(h.touch_type, []).append(h)
            
        # Sort TFs
        tf_order = ["1w", "1d", "4h", "2h", "1h", "30m", "15m"]
        
        messages = []
        
        for tf in tf_order:
            if tf not in grouped: continue
            
            # Build Message
            lines = [
                f"🔍 *BB Touches on {tf} Timeframe*",
                ""
            ]
            
            types = ["UPPER", "MIDDLE", "LOWER"]
            headers = {"UPPER": "⬆️ UPPER BB Touches:", "MIDDLE": "🔶 MIDDLE BB Touches:", "LOWER": "⬇️ LOWER BB Touches:"}
            
            has_content = False
            
            for t_type in types:
                items = grouped[tf].get(t_type, [])
                if not items: continue
                has_content = True
                
                lines.append(f"*{headers[t_type]}*")
                
                # Sort by symbol
                items.sort(key=lambda x: x.symbol)
                
                for item in items:
                    # Format: • BTCUSDT [FUTURES] - RSI: 72.5 🔥
                    market_tag = "[FUTURES]" if item.market == 'perp' else "[SPOT]"
                    clean_sym = item.symbol.replace("USDT", "")
                    hot_icon = " 🔥" if item.hot else ""
                    
                    extra = ""
                    if t_type == "MIDDLE":
                        arrow = "🔻" if item.direction == "from above" else "🔹"
                        extra = f" ({arrow})"
                    
                    line = f"• *{clean_sym}* {market_tag} - RSI: {item.rsi:.2f}{extra}{hot_icon}"
                    lines.append(line)
                lines.append("")
            
            if has_content:
                messages.append("\n".join(lines))
        
        # Send
        async with aiohttp.ClientSession() as session:
            for msg in messages:
                chunks = [msg[i:i+4000] for i in range(0, len(msg), 4000)]
                for chunk in chunks:
                    try:
                        url = f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage"
                        await session.post(url, json={"chat_id": CONFIG.CHAT_ID, "text": chunk, "parse_mode": "Markdown"})
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        logging.error(f"TG Error: {e}")

    async def run(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        await self.cache.init()
        
        async with aiohttp.ClientSession() as session:
            # 1. Init Proxies
            await self.proxies.populate(CONFIG.PROXY_URL, session)
            
            # 2. Clients
            binance = BinanceClient(session, self.proxies)
            bybit = BybitClient(session, self.proxies)
            
            # 3. Fetch Symbols
            logging.info("Fetching symbols...")
            bp, bs = await asyncio.gather(binance.get_perp_symbols(), binance.get_spot_symbols())
            yp, ys = await asyncio.gather(bybit.get_perp_symbols(), bybit.get_spot_symbols())
            
            # 4. Deduplication (Pattern Filter + Priority)
            seen = set()
            def filter_u(syms):
                res = []
                for s in syms:
                    if s in CONFIG.IGNORED_SYMBOLS: continue
                    if not s.endswith("USDT"): continue
                    norm = s.upper().replace("USDT", "")
                    if norm not in seen:
                        res.append(s)
                        seen.add(norm)
                return res
                
            f_bp, f_bs, f_yp, f_ys = filter_u(bp), filter_u(bs), filter_u(yp), filter_u(ys)
            all_pairs = [] # (client, sym, mkt, ex)
            for s in f_bp: all_pairs.append((binance, s, 'perp', 'Binance'))
            for s in f_bs: all_pairs.append((binance, s, 'spot', 'Binance'))
            for s in f_yp: all_pairs.append((bybit, s, 'perp', 'Bybit'))
            for s in f_ys: all_pairs.append((bybit, s, 'spot', 'Bybit'))
            
            logging.info(f"Total symbols to scan: {len(all_pairs)}")
            
            # 5. Volatility Ranking (Top 60)
            # We use Binance Perp for volatility calculation as reference
            logging.info("Calculating Volatility Rankings...")
            vol_scores = {}
            async def check_vol(s):
                closes = await binance.fetch_closes_volatility(s, 'perp')
                v = calculate_volatility(closes)
                if v > 0: vol_scores[s] = v
            
            # Check only Binance Perps for volatility to save time/calls
            vol_tasks = [check_vol(s) for s in f_bp]
            # Batch them
            for i in range(0, len(vol_tasks), 100):
                await asyncio.gather(*vol_tasks[i:i+100])
                
            hot_coins = set(sorted(vol_scores, key=vol_scores.get, reverse=True)[:60])
            logging.info(f"Identified {len(hot_coins)} HOT coins 🔥")

            # 6. Check Cache State for High TFs
            sent_state = await self.cache.get_sent_state()
            
            # We determine which TFs to actually scan
            # Low TFs: Always scan
            # High TFs: Only scan if current candle time != last sent time
            
            tfs_map = {} # TF -> should_scan
            for tf in ACTIVE_TFS:
                if tf in CACHED_TFS:
                    iso = get_candle_open_iso(tf)
                    # We check if we ALREADY sent this specific candle
                    last_sent = sent_state.get(tf, "")
                    if last_sent != iso:
                        tfs_map[tf] = True # New candle! Scan it.
                        # Update state immediately so we don't rescan next run? 
                        # No, update state only after successful send.
                    else:
                        tfs_map[tf] = False # Already handled this candle
                else:
                    tfs_map[tf] = True # Always scan low TFs
            
            logging.info(f"Scan Plan: {tfs_map}")
            
            # 7. Scanning
            scan_tasks = []
            async def scan_one(client, sym, mkt, ex, tfs_to_check):
                hits = []
                for tf in tfs_to_check:
                    closes = await client.fetch_closes(sym, tf, mkt)
                    t_type, direction, rsi_val = check_bb_rsi(closes, tf)
                    if t_type:
                        hits.append(TouchHit(
                            symbol=sym, exchange=ex, market=mkt, timeframe=tf,
                            rsi=rsi_val, touch_type=t_type, direction=direction,
                            hot=(sym in hot_coins)
                        ))
                return hits

            # Only pass TFs that are True in tfs_map
            active_scan_tfs = [tf for tf, active in tfs_map.items() if active]
            
            if not active_scan_tfs:
                logging.info("All high TFs cached and up-to-date. Low TFs disabled? Nothing to scan.")
                return

            for client, sym, mkt, ex in all_pairs:
                scan_tasks.append(scan_one(client, sym, mkt, ex, active_scan_tfs))
                
            results = []
            for f in asyncio.as_completed(scan_tasks):
                results.extend(await f)
                
            # 8. Reporting & State Update
            if results:
                await self.send_report(results)
                
                # Update state for High TFs that had results or were scanned successfully
                new_state = sent_state.copy()
                for tf in CACHED_TFS:
                    if tfs_map[tf]: # If we decided to scan it
                        new_state[tf] = get_candle_open_iso(tf)
                
                await self.cache.save_sent_state(new_state)

        await self.cache.close()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    bot = RsiBot()
    asyncio.run(bot.run())
