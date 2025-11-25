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
    FETCH_TIMEOUT_TOTAL: int = 15
    REQUEST_TIMEOUT: int = 5
    MAX_RETRIES: int = 3
    MAX_CONCURRENCY: int = 50
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_DIR: str = "bb_touch_cache"

    # Telegram
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    PROXY_URL: str = "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"

    # Indicators
    RSI_PERIOD: int = 14
    BB_LENGTH: int = 34
    BB_STDDEV: float = 2.0
    CANDLE_LIMIT: int = 60
    MIN_CANDLES: int = 40  # Safety buffer for Talib

    # Thresholds
    UPPER_TOUCH_THRESHOLD: float = 0.02 # 2%
    LOWER_TOUCH_THRESHOLD: float = 0.02 # 2%
    MIDDLE_TOUCH_THRESHOLD: float = 0.035 # 3.5%

    IGNORED_SYMBOLS: Set[str] = field(default_factory=lambda: {
        "USDPUSDT", "USD1USDT", "TUSDUSDT", "AEURUSDT", "USDCUSDT", "EURUSDT"
    })

CONFIG = Config()

# Timeframe Mapping
TIMEFRAME_MAP = {
    '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, 
    '1d': 1440, '1w': 10080
}

# Toggles
ACTIVE_TFS = ['15m', '30m', '1h', '2h', '4h', '1d', '1w']
MIDDLE_BAND_TFS = ['1h', '2h', '4h', '1d', '1w']

# ==========================================
# DATA MODELS
# ==========================================

@dataclass
class TouchHit:
    symbol: str
    exchange: str # Binance / Bybit
    market: str   # PERP / SPOT
    timeframe: str
    rsi: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    touch_type: str # UPPER, LOWER, MIDDLE
    direction: str = ""
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
        """Fetches AND validates proxies with high-speed early exit."""
        if not url:
            logging.warning("⚠️ No Proxy URL provided! Running without proxies.")
            return

        raw_proxies = []
        try:
            logging.info(f"📥 Fetching proxies from {url}...")
            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    for line in text.splitlines():
                        p = line.strip()
                        if p:
                            raw_proxies.append(p if "://" in p else f"http://{p}")
        except Exception as e:
            logging.error(f"❌ Failed to fetch proxy list: {e}")
            return

        logging.info(f"🔎 Validating {len(raw_proxies)} proxies (Target: {self.max_pool_size})...")
        
        self.proxies = []
        random.shuffle(raw_proxies)
        
        # Throttled check
        sem = asyncio.Semaphore(200)

        async def protected_test(p):
            async with sem:
                return await self._test_proxy(p, session)

        tasks = [asyncio.create_task(protected_test(p)) for p in raw_proxies]

        for future in asyncio.as_completed(tasks):
            try:
                proxy, is_good = await future
                if is_good:
                    self.proxies.append(proxy)
                    if len(self.proxies) >= self.max_pool_size:
                        break
            except: pass
        
        # Cancel remaining
        for t in tasks:
            if not t.done(): t.cancel()
        
        await asyncio.sleep(0.1)
        
        if self.proxies:
            self.iterator = cycle(self.proxies)
            logging.info(f"✅ Proxy Pool Ready: {len(self.proxies)} working proxies.")
        else:
            logging.error("❌ NO WORKING PROXIES FOUND! Calls will likely fail.")

    async def _test_proxy(self, proxy: str, session: aiohttp.ClientSession) -> Tuple[str, bool]:
        try:
            # Fast timeout 3s
            test_url = "https://api.binance.com/api/v3/time"
            async with session.get(test_url, proxy=proxy, timeout=3) as resp:
                return proxy, resp.status == 200
        except:
            return proxy, False

    async def get_proxy(self) -> Optional[str]:
        if not self.proxies: return None
        async with self._lock:
            return next(self.iterator)

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
        except Exception as e:
            logging.warning(f"⚠️ Redis connection failed: {e}")
            self.redis = None

    async def close(self):
        if self.redis: await self.redis.close()

    def _key(self, tf: str, iso_time: str) -> str:
        return f"bb_bot:scan:{tf}:{iso_time}"

    async def get_cached_results(self, tf: str, iso_time: str):
        if not self.redis: return None
        try:
            data = await self.redis.get(self._key(tf, iso_time))
            return json.loads(data) if data else None
        except: return None

    async def save_results(self, tf: str, iso_time: str, data: list):
        if not self.redis: return
        # Cache TTL varies by timeframe
        ttl = 3600 if tf == '1h' else 14400
        if tf == '1d': ttl = 86400
        try:
            await self.redis.set(self._key(tf, iso_time), json.dumps(data), ex=ttl)
        except: pass

    async def get_sent_state(self):
        if not self.redis: return {}
        try:
            val = await self.redis.get("bb_bot:sent_state")
            return json.loads(val) if val else {}
        except: return {}

    async def save_sent_state(self, state):
        if not self.redis: return
        try:
            await self.redis.set("bb_bot:sent_state", json.dumps(state))
        except: pass

# ==========================================
# EXCHANGE CLIENTS
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

    async def fetch_closes(self, symbol: str, interval: str, market: str) -> List[float]:
        base = 'https://api.binance.com/api/v3/klines' if market == "spot" else 'https://fapi.binance.com/fapi/v1/klines'
        data = await self._request(base, {'symbol': symbol, 'interval': interval, 'limit': CONFIG.CANDLE_LIMIT})
        if not data: return []
        # Return closes only (index 4)
        try:
            return [float(c[4]) for c in data]
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
        bybit_int = {"1M": "M", "1w": "W", "1d": "D"}.get(interval, interval[-1].upper() if interval[:-1].isdigit() else "D")
        if interval == '15m': bybit_int = '15'
        if interval == '30m': bybit_int = '30'
        if interval == '1h': bybit_int = '60'
        if interval == '2h': bybit_int = '120'
        if interval == '4h': bybit_int = '240'
        
        data = await self._request(url, {'category': cat, 'symbol': symbol, 'interval': bybit_int, 'limit': CONFIG.CANDLE_LIMIT})
        if not data: return []
        raw = data.get('result', {}).get('list', [])
        if not raw: return []
        # Bybit V5 is Newest->Oldest. We need Oldest->Newest for Talib
        closes = [float(c[4]) for c in raw]
        return closes[::-1]

# ==========================================
# CORE LOGIC
# ==========================================

def calculate_indicators(closes: List[float]) -> Optional[TouchHit]:
    if len(closes) < CONFIG.MIN_CANDLES: return None
    
    np_closes = np.array(closes, dtype=float)
    
    # RSI
    rsi = talib.RSI(np_closes, timeperiod=CONFIG.RSI_PERIOD)
    
    # BB
    upper, middle, lower = talib.BBANDS(rsi, timeperiod=CONFIG.BB_LENGTH, nbdevup=CONFIG.BB_STDDEV, nbdevdn=CONFIG.BB_STDDEV, matype=0)
    
    # Check last closed candle (index -2, since -1 is forming)
    idx = -2
    if np.isnan(rsi[idx]) or np.isnan(upper[idx]): return None
    
    curr_rsi = rsi[idx]
    bb_up = upper[idx]
    bb_mid = middle[idx]
    bb_low = lower[idx]
    
    # Logic
    touch_type = ""
    direction = ""
    
    # Upper Touch
    if curr_rsi >= bb_up * (1 - CONFIG.UPPER_TOUCH_THRESHOLD):
        touch_type = "UPPER"
    
    # Lower Touch
    elif curr_rsi <= bb_low * (1 + CONFIG.LOWER_TOUCH_THRESHOLD):
        touch_type = "LOWER"
        
    # Middle Touch (Optional Logic)
    elif abs(curr_rsi - bb_mid) <= (bb_mid * CONFIG.MIDDLE_TOUCH_THRESHOLD):
        touch_type = "MIDDLE"
        # Direction
        prev_rsi = rsi[idx-1]
        prev_mid = middle[idx-1]
        if prev_rsi > prev_mid and curr_rsi <= bb_mid: direction = "from above"
        elif prev_rsi < prev_mid and curr_rsi >= bb_mid: direction = "from below"
        else: direction = "near"

    if not touch_type: return None
    
    return TouchHit(
        symbol="", exchange="", market="", timeframe="",
        rsi=curr_rsi, bb_upper=bb_up, bb_middle=bb_mid, bb_lower=bb_low,
        touch_type=touch_type, direction=direction
    )

def get_candle_open_time(tf_str: str) -> str:
    """Get ISO timestamp of the current candle open for cache keys."""
    now = datetime.now(timezone.utc)
    minutes = TIMEFRAME_MAP.get(tf_str, 60)
    
    # Simple floor to nearest interval
    total_min = now.hour * 60 + now.minute
    floored = (total_min // minutes) * minutes
    
    open_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=floored)
    return open_time.isoformat()

# ==========================================
# MAIN BOT
# ==========================================

class RsiBot:
    def __init__(self):
        self.cache = CacheManager()
        self.proxies = AsyncProxyPool()
        
    async def send_report(self, hits: List[TouchHit], cached_tfs: Set[str]):
        if not hits:
            logging.info("No hits to report.")
            return
            
        # Helper
        def _clean_sym(s): return s.replace("USDT", "")
        
        lines = [
            "🚨 *RSI & BB SCANNER*",
            "═════════════════════\n"
        ]
        
        # Group: Timeframe -> Exchange -> Type -> List
        grouped = {}
        for h in hits:
            grouped.setdefault(h.timeframe, {}).setdefault(h.exchange, {}).setdefault(h.touch_type, []).append(h)
            
        # Sort TFs by duration
        tf_order = ["15m", "30m", "1h", "2h", "4h", "1d", "1w"]
        
        for tf in tf_order:
            if tf not in grouped: continue
            
            is_cached = tf in cached_tfs
            header = f"📅 *{tf.upper()}*" + (" _(Cached)_" if is_cached else "")
            lines.append(header + "\n")
            
            exchanges = [("Binance", "🟡"), ("Bybit", "⚫")]
            
            for ex_name, ex_icon in exchanges:
                ex_data = grouped[tf].get(ex_name, {})
                
                upper = sorted(ex_data.get("UPPER", []), key=lambda x: x.symbol)
                lower = sorted(ex_data.get("LOWER", []), key=lambda x: x.symbol)
                middle = sorted(ex_data.get("MIDDLE", []), key=lambda x: x.symbol)
                
                if not (upper or lower or middle): continue
                
                lines.append(f"┌ {ex_icon} *{ex_name.upper()}*")
                
                # Formatter
                def fmt(items, label, icon):
                    res = []
                    for i in items:
                        hot = "🔥" if i.hot else ""
                        res.append(f"{_clean_sym(i.symbol)} {hot}({i.rsi:.1f})")
                    return f"{icon} *{label}*: {', '.join(res)}"
                
                # Bracket Logic
                sections = []
                if upper: sections.append(fmt(upper, "Overbought", "📈"))
                if lower: sections.append(fmt(lower, "Oversold", "📉"))
                if middle: sections.append(fmt(middle, "Middle", "🔹"))
                
                for i, s in enumerate(sections):
                    prefix = "└" if i == len(sections)-1 else "│"
                    lines.append(f"{prefix} {s}")
                
                lines.append("")
            
            lines.append("═════════════════════\n")
            
        text = "\n".join(lines)
        
        # Send Logic
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage"
            # Chunking
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for chunk in chunks:
                try:
                    await session.post(url, json={"chat_id": CONFIG.CHAT_ID, "text": chunk, "parse_mode": "Markdown"})
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logging.error(f"TG Send Error: {e}")

    async def run(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        
        await self.cache.init()
        
        async with aiohttp.ClientSession() as session:
            # 1. Proxies
            await self.proxies.populate(CONFIG.PROXY_URL, session)
            
            # 2. Clients
            binance = BinanceClient(session, self.proxies)
            bybit = BybitClient(session, self.proxies)
            
            # 3. Fetch Symbols
            logging.info("🌍 Fetching symbols...")
            b_perp, b_spot = await asyncio.gather(binance.get_perp_symbols(), binance.get_spot_symbols())
            y_perp, y_spot = await asyncio.gather(bybit.get_perp_symbols(), bybit.get_spot_symbols())
            
            # 4. Deduplication (Same logic as Market Scan)
            seen_norm = set()
            def filter_unique(syms):
                res = []
                for s in syms:
                    if s in CONFIG.IGNORED_SYMBOLS: continue
                    if not s.endswith("USDT"): continue # Ban dated futures
                    
                    norm = s.upper().replace("USDT", "") # Simple norm
                    if norm not in seen_norm:
                        res.append(s)
                        seen_norm.add(norm)
                return res

            final_bp = filter_unique(b_perp)
            final_bs = filter_unique(b_spot)
            final_yp = filter_unique(y_perp)
            final_ys = filter_unique(y_spot)
            
            logging.info(f"Scanning: B-Perp:{len(final_bp)} B-Spot:{len(final_bs)} Y-Perp:{len(final_yp)} Y-Spot:{len(final_ys)}")
            
            # 5. Check Cached Timeframes
            sent_state = await self.cache.get_sent_state()
            tfs_to_scan = []
            cached_tfs_found = set()
            
            # Logic: Some TFs are cached. If cache is valid (same candle), skip scanning unless force refresh needed.
            # Here, we simplify: Scan everything, but only ALERT if new candle or uncached.
            
            # Actually, for speed, we should skip scanning if cache exists and is valid.
            # But for this specific bot logic ("BB Touch"), price moves inside the candle, so we usually want to scan 
            # lower timeframes FRESH every time. Higher TFs (4h, 1d) can be cached.
            
            # Let's scan everything fresh for maximum accuracy, but use cache to prevent double-alerting 
            # if the candle hasn't changed.
            
            # 6. Build Tasks
            scan_tasks = []
            
            async def scan_one(client, sym, mkt, ex, tfs):
                hits = []
                for tf in tfs:
                    # Optimization: Check Middle Band toggle
                    if "MIDDLE" not in hits and tf not in MIDDLE_BAND_TFS:
                        pass # Logic specific to middle band reqs
                        
                    closes = await client.fetch_closes(sym, tf, mkt)
                    hit = calculate_indicators(closes)
                    if hit:
                        hit.symbol = sym
                        hit.exchange = ex
                        hit.market = mkt
                        hit.timeframe = tf
                        hits.append(hit)
                return hits

            # Add tasks
            for s in final_bp: scan_tasks.append(scan_one(binance, s, 'perp', 'Binance', ACTIVE_TFS))
            for s in final_bs: scan_tasks.append(scan_one(binance, s, 'spot', 'Binance', ACTIVE_TFS))
            for s in final_yp: scan_tasks.append(scan_one(bybit, s, 'perp', 'Bybit', ACTIVE_TFS))
            for s in final_ys: scan_tasks.append(scan_one(bybit, s, 'spot', 'Bybit', ACTIVE_TFS))
            
            # 7. Execute
            all_hits = []
            for f in asyncio.as_completed(scan_tasks):
                res = await f
                all_hits.extend(res)
                
            # 8. Filter Alerts (State Management)
            # We only want to send alerts if:
            # A) It's a low timeframe (always send)
            # B) It's a high timeframe AND it's a NEW candle since last alert
            
            hits_to_send = []
            new_state = sent_state.copy()
            
            for h in all_hits:
                tf = h.timeframe
                
                if tf in ['15m', '30m', '1h']: # Always alert low TF
                    hits_to_send.append(h)
                else:
                    # High TF: Check state
                    iso = get_candle_open_time(tf)
                    last_sent = sent_state.get(f"{h.exchange}_{h.symbol}_{tf}_{h.touch_type}", "")
                    
                    if last_sent != iso:
                        hits_to_send.append(h)
                        new_state[f"{h.exchange}_{h.symbol}_{tf}_{h.touch_type}"] = iso
            
            await self.cache.save_sent_state(new_state)
            
            # 9. Send
            await self.send_report(hits_to_send, set())
            
        await self.cache.close()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    bot = RsiBot()
    asyncio.run(bot.run())
