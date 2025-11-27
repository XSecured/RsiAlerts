import asyncio
import json
import logging
import os
import random
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Set, Optional, Tuple, Any
from itertools import cycle

import aiohttp
import numpy as np
import talib
import redis.asyncio as aioredis

# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class Config:
    MAX_CONCURRENCY: int = 100 # Tuned for stability
    REQUEST_TIMEOUT: int = 7
    MAX_RETRIES: int = 5
    
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    PROXY_URL: str = "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"

    RSI_PERIOD: int = 14
    BB_LENGTH: int = 34
    BB_STDDEV: float = 2.0
    CANDLE_LIMIT: int = 60
    MIN_CANDLES: int = 36

    UPPER_TOUCH_THRESHOLD: float = 0.02
    LOWER_TOUCH_THRESHOLD: float = 0.02
    MIDDLE_TOUCH_THRESHOLD: float = 0.035

    IGNORED_SYMBOLS: Set[str] = field(default_factory=lambda: {
        "USDPUSDT", "USD1USDT", "TUSDUSDT", "AEURUSDT", "USDCUSDT", "EURUSDT"
    })

CONFIG = Config()

TIMEFRAME_MINUTES = {
    '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, 
    '1d': 1440, '1w': 10080
}

ACTIVE_TFS = ['15m', '30m', '1h', '2h', '4h', '1d', '1w']
MIDDLE_BAND_TFS = ['1h', '2h', '4h', '1d', '1w']
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
    touch_type: str
    direction: str = "" 
    hot: bool = False
    
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return TouchHit(**d)

@dataclass
class ScanStats:
    timeframe: str
    source: str
    total_symbols: int = 0
    successful_scans: int = 0
    hits_found: int = 0

# ==========================================
# INTELLIGENT CACHE & STATE
# ==========================================

class SmartCache:
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

    # --- Symbol List Caching ---
    async def get_cached_symbols(self) -> Optional[Dict]:
        if not self.redis: return None
        data = await self.redis.get("bb_bot:symbols_cache")
        return json.loads(data) if data else None

    async def save_cached_symbols(self, data: Dict):
        if not self.redis: return
        # Cache symbol list for 24 hours
        await self.redis.set("bb_bot:symbols_cache", json.dumps(data), ex=86400)

    # --- Scan Result Caching ---
    def _scan_key(self, tf: str, iso_time: str) -> str:
        return f"bb_touch:scan:{tf}:{iso_time}"

    async def get_scan_results(self, tf: str, iso_time: str) -> Optional[List[Dict]]:
        if not self.redis: return None
        data = await self.redis.get(self._scan_key(tf, iso_time))
        return json.loads(data) if data else None

    async def save_scan_results(self, tf: str, iso_time: str, results: List[Dict]):
        if not self.redis: return
        ttl = 604800 if tf == '1w' else 86400 
        await self.redis.set(self._scan_key(tf, iso_time), json.dumps(results), ex=ttl)

    # --- Alert State ---
    async def get_sent_state(self) -> Dict[str, str]:
        if not self.redis: return {}
        data = await self.redis.get("bb_bot:sent_state")
        return json.loads(data) if data else {}

    async def save_sent_state(self, state: Dict[str, str]):
        if not self.redis: return
        await self.redis.set("bb_bot:sent_state", json.dumps(state))

# ==========================================
# STICKY PROXY SESSION MANAGER
# ==========================================

class StickySessionManager:
    def __init__(self, proxy_url: str):
        self.proxy_url = proxy_url
        self.proxies: List[str] = []
        self.sessions: List[Tuple[aiohttp.ClientSession, str]] = [] 
        self.iterator = None
        self._lock = asyncio.Lock()

    async def init(self):
        # Fetch proxies once
        async with aiohttp.ClientSession() as temp_session:
            try:
                async with temp_session.get(self.proxy_url, timeout=10) as resp:
                    text = await resp.text()
                    for line in text.splitlines():
                        if line.strip(): self.proxies.append(line.strip())
            except: pass
        
        if not self.proxies:
            logging.error("No proxies found! Using direct connection.")
            self.sessions.append((aiohttp.ClientSession(), None))
            self.iterator = cycle(self.sessions)
            return

        logging.info(f"Loaded {len(self.proxies)} proxies. Initializing Sticky Sessions...")
        
        random.shuffle(self.proxies)
        # Create a session for top 30 proxies
        for p in self.proxies[:30]:
            connector = aiohttp.TCPConnector(ssl=False, limit=10, ttl_dns_cache=300)
            sess = aiohttp.ClientSession(connector=connector)
            p_url = p if "://" in p else f"http://{p}"
            self.sessions.append((sess, p_url))

        self.iterator = cycle(self.sessions)

    async def get_session_and_proxy(self):
        async with self._lock:
            return next(self.iterator)

    async def close(self):
        for sess, _ in self.sessions:
            await sess.close()

# ==========================================
# MARKET CLIENT
# ==========================================

async def fetch_with_sticky_session(session_mgr: StickySessionManager, url: str, params: dict = None):
    """Fetches using a persistent session to reduce handshake overhead."""
    for _ in range(CONFIG.MAX_RETRIES):
        session, proxy_url = await session_mgr.get_session_and_proxy()
        try:
            async with session.get(url, params=params, proxy=proxy_url, timeout=CONFIG.REQUEST_TIMEOUT) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 418: # Ban
                    pass 
                elif resp.status == 429:
                    await asyncio.sleep(2)
        except:
            pass
        # Rotate to next session automatically
    return None

class MarketClient:
    def __init__(self, session_mgr: StickySessionManager):
        self.mgr = session_mgr

    async def get_symbols(self, exchange: str, market: str) -> List[str]:
        url = ""
        if exchange == 'Binance':
            url = 'https://fapi.binance.com/fapi/v1/exchangeInfo' if market == 'perp' else 'https://api.binance.com/api/v3/exchangeInfo'
        else: # Bybit
            url = 'https://api.bybit.com/v5/market/instruments-info'
        
        params = {}
        if exchange == 'Bybit':
            params = {'category': 'linear' if market == 'perp' else 'spot'}

        data = await fetch_with_sticky_session(self.mgr, url, params)
        if not data: return []

        symbols = []
        if exchange == 'Binance':
            if 'symbols' in data:
                for s in data['symbols']:
                    if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT':
                        if market == 'perp' and s.get('contractType') == 'PERPETUAL': symbols.append(s['symbol'])
                        elif market == 'spot': symbols.append(s['symbol'])
        else: # Bybit
            if 'result' in data and 'list' in data['result']:
                for s in data['result']['list']:
                    if s['status'] == 'Trading' and s['quoteCoin'] == 'USDT':
                        symbols.append(s['symbol'])
        return symbols

    async def fetch_candles(self, exchange: str, market: str, symbol: str, interval: str) -> List[float]:
        url = ""
        params = {'symbol': symbol, 'limit': CONFIG.CANDLE_LIMIT}
        
        if exchange == 'Binance':
            url = 'https://fapi.binance.com/fapi/v1/klines' if market == 'perp' else 'https://api.binance.com/api/v3/klines'
            params['interval'] = interval
        else:
            url = 'https://api.bybit.com/v5/market/kline'
            params['category'] = 'linear' if market == 'perp' else 'spot'
            b_map = {'15m': '15', '30m': '30', '1h': '60', '2h': '120', '4h': '240', '1d': 'D', '1w': 'W'}
            params['interval'] = b_map.get(interval, 'D')

        data = await fetch_with_sticky_session(self.mgr, url, params)
        if not data: return []
        
        try:
            if exchange == 'Binance':
                return [float(c[4]) for c in data]
            else:
                raw = data.get('result', {}).get('list', [])
                if not raw: return []
                return [float(c[4]) for c in raw][::-1]
        except: return []

# ==========================================
# CORE LOGIC
# ==========================================

def check_bb_rsi(closes: List[float], tf: str) -> Tuple[Optional[str], Optional[str], float]:
    if len(closes) < CONFIG.MIN_CANDLES: return None, None, 0.0
    np_c = np.array(closes, dtype=float)
    rsi = talib.RSI(np_c, timeperiod=CONFIG.RSI_PERIOD)
    upper, mid, lower = talib.BBANDS(rsi, timeperiod=CONFIG.BB_LENGTH, nbdevup=CONFIG.BB_STDDEV, nbdevdn=CONFIG.BB_STDDEV, matype=0)
    
    idx = -2
    if np.isnan(rsi[idx]) or np.isnan(upper[idx]): return None, None, 0.0
    curr, up, low = rsi[idx], upper[idx], lower[idx]
    
    if curr >= up * (1 - CONFIG.UPPER_TOUCH_THRESHOLD): return "UPPER", None, curr
    if curr <= low * (1 + CONFIG.LOWER_TOUCH_THRESHOLD): return "LOWER", None, curr
    
    if tf in MIDDLE_BAND_TFS:
        m_val = mid[idx]
        if abs(curr - m_val) <= (m_val * CONFIG.MIDDLE_TOUCH_THRESHOLD):
            prev_diff = rsi[idx-1] - mid[idx-1]
            curr_diff = curr - m_val
            direction = "from above" if (prev_diff > 0 >= curr_diff) or (curr_diff > 0) else "from below"
            return "MIDDLE", direction, curr
    return None, None, 0.0

def calculate_volatility(closes: List[float]) -> float:
    if len(closes) < 24: return 0.0
    rets = [(closes[i] - closes[i-1])/closes[i-1]*100 for i in range(1, len(closes)) if closes[i-1]!=0]
    return np.std(rets) if rets else 0.0

def get_iso(tf):
    now = datetime.now(timezone.utc)
    m = TIMEFRAME_MINUTES.get(tf, 60)
    return (now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=(now.minute // m) * m)).isoformat()

# ==========================================
# MAIN BOT
# ==========================================

class RsiBot:
    def __init__(self):
        self.cache = SmartCache()
        self.sess_mgr = StickySessionManager(CONFIG.PROXY_URL)
        self.client = MarketClient(self.sess_mgr)

    async def get_all_symbols(self) -> Dict[str, List[str]]:
        # 1. Try Cache
        cached = await self.cache.get_cached_symbols()
        if cached:
            logging.info("✅ Loaded symbol list from Cache")
            return cached
        
        # 2. Fetch Fresh
        logging.info("🔄 Fetching fresh symbol list...")
        t1 = self.client.get_symbols('Binance', 'perp')
        t2 = self.client.get_symbols('Binance', 'spot')
        t3 = self.client.get_symbols('Bybit', 'perp')
        t4 = self.client.get_symbols('Bybit', 'spot')
        
        r1, r2, r3, r4 = await asyncio.gather(t1, t2, t3, t4)
        res = {"bp": r1, "bs": r2, "yp": r3, "ys": r4}
        
        if sum(len(v) for v in res.values()) > 100:
            await self.cache.save_cached_symbols(res)
        return res

    async def run(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        await self.cache.init()
        await self.sess_mgr.init()
        
        # A. Symbols
        raw_syms = await self.get_all_symbols()
        seen = set()
        pairs = []
        for key, ex, mkt in [('bp', 'Binance', 'perp'), ('bs', 'Binance', 'spot'), 
                             ('yp', 'Bybit', 'perp'), ('ys', 'Bybit', 'spot')]:
            for s in raw_syms.get(key, []):
                if s in CONFIG.IGNORED_SYMBOLS or not s.endswith('USDT'): continue
                norm = s.upper().replace('USDT', '')
                if norm not in seen:
                    seen.add(norm)
                    pairs.append((s, ex, mkt))
        
        logging.info(f"Total Targets: {len(pairs)}")

        # B. Volatility
        logging.info("Calculating Volatility...")
        vol_scores = {}
        async def calc_vol(s, ex, mkt):
            c = await self.client.fetch_candles(ex, mkt, s, '1h') 
            v = calculate_volatility(c[-25:])
            if v > 0: vol_scores[s] = v
        
        tasks = [calc_vol(s, ex, mkt) for s, ex, mkt in pairs]
        for i in range(0, len(tasks), 200): await asyncio.gather(*tasks[i:i+200])
        hot_coins = set(sorted(vol_scores, key=vol_scores.get, reverse=True)[:60])
        logging.info(f"Volatility: {len(vol_scores)}/{len(pairs)} success | Hot: {len(hot_coins)}")

        # C. Scan Logic
        sent_state = await self.cache.get_sent_state()
        scan_stats = {tf: ScanStats(tf, "Unknown", len(pairs)) for tf in ACTIVE_TFS}
        
        tfs_to_scan = []
        cached_hits = []
        
        for tf in ACTIVE_TFS:
            if tf in CACHED_TFS:
                iso = get_iso(tf)
                existing_res = await self.cache.get_scan_results(tf, iso)
                if existing_res:
                    scan_stats[tf].source = "Cached"
                    scan_stats[tf].hits_found = len(existing_res)
                    for d in existing_res: cached_hits.append(TouchHit.from_dict(d))
                else:
                    scan_stats[tf].source = "Fresh (New Candle)"
                    tfs_to_scan.append(tf)
            else:
                scan_stats[tf].source = "Fresh (Low TF)"
                tfs_to_scan.append(tf)

        final_hits = []
        
        # Execute Fresh Scans
        if tfs_to_scan:
            logging.info(f"Scanning Fresh: {tfs_to_scan}")
            fresh_results = []
            async def worker(s, ex, mkt):
                h = []
                scanned_tfs = []
                for tf in tfs_to_scan:
                    c = await self.client.fetch_candles(ex, mkt, s, tf)
                    if c:
                        scanned_tfs.append(tf)
                        type_, dir_, val = check_bb_rsi(c, tf)
                        if type_:
                            h.append(TouchHit(s, ex, mkt, tf, val, type_, dir_, s in hot_coins))
                return h, scanned_tfs

            scan_tasks = [worker(s, ex, mkt) for s, ex, mkt in pairs]
            for i in range(0, len(scan_tasks), CONFIG.MAX_CONCURRENCY):
                batch = scan_tasks[i:i+CONFIG.MAX_CONCURRENCY]
                results = await asyncio.gather(*batch)
                for hits, tfs in results:
                    fresh_results.extend(hits)
                    for tf in tfs: scan_stats[tf].successful_scans += 1
            
            # Cache results
            for tf in tfs_to_scan:
                if tf in CACHED_TFS:
                    tf_hits = [x for x in fresh_results if x.timeframe == tf]
                    await self.cache.save_scan_results(tf, get_iso(tf), [x.to_dict() for x in tf_hits])
                    scan_stats[tf].hits_found = len(tf_hits)
            final_hits.extend(fresh_results)

        # Merge Cached
        for h in cached_hits:
            h.hot = (h.symbol in hot_coins)
            final_hits.append(h)

        # Log Summary
        logging.info("="*60)
        logging.info(f"{'TF':<5} | {'Source':<25} | {'Success':<15} | {'Hits'}")
        logging.info("-" * 60)
        for tf in sorted(ACTIVE_TFS, key=lambda x: ["15m","30m","1h","2h","4h","1d","1w"].index(x)):
            st = scan_stats[tf]
            succ = "Skipped" if st.source == "Cached" else f"{st.successful_scans}/{st.total_symbols}"
            logging.info(f"[{tf:<3}] {st.source:<25} | {succ:<15} | {st.hits_found}")
        logging.info("="*60)

        # D. Alerting
        hits_to_send = []
        new_state = sent_state.copy()
        for h in final_hits:
            tf = h.timeframe
            if tf in CACHED_TFS:
                iso = get_iso(tf)
                if sent_state.get(tf) != iso:
                    hits_to_send.append(h)
                    new_state[tf] = iso
            else:
                hits_to_send.append(h)
        
        await self.send_telegram(hits_to_send)
        await self.cache.save_sent_state(new_state)
        await self.sess_mgr.close()
        await self.cache.close()

    async def send_telegram(self, hits: List[TouchHit]):
        if not hits: return
        grouped = {}
        for h in hits: grouped.setdefault(h.timeframe, {}).setdefault(h.touch_type, []).append(h)
        messages = []
        for tf in ["1w", "1d", "4h", "2h", "1h", "30m", "15m"]:
            if tf not in grouped: continue
            lines = [f"▣ TIMEFRAME: {tf}", "────────────────", ""]
            headers = {"UPPER": "⬆️ UPPER BB", "MIDDLE": "🔶 MIDDLE BB", "LOWER": "⬇️ LOWER BB"}
            found = [t for t in ["UPPER", "MIDDLE", "LOWER"] if grouped[tf].get(t)]
            for t in found:
                items = grouped[tf].get(t, [])
                items.sort(key=lambda x: x.symbol)
                lines.append(f"┌ {headers[t]}")
                for idx, item in enumerate(items):
                    prefix = "└" if idx == len(items)-1 else "│"
                    icon = "🍋" if item.exchange == "Binance" else "🍙"
                    ext = f" ({'🔻' if item.direction=='from above' else '🔹'})" if t=="MIDDLE" else ""
                    if item.hot: ext += " 🔥"
                    lines.append(f"{prefix} {icon} {item.symbol} | {item.rsi:.2f}{ext}")
                lines.append("")
            lines.append("────────────────")
            lines.append(datetime.utcnow().strftime('%d %b %H:%M UTC'))
            messages.append("\n".join(lines))
            
        async with aiohttp.ClientSession() as s:
            for msg in messages:
                for chunk in [msg[i:i+4000] for i in range(0, len(msg), 4000)]:
                    try: await s.post(f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CONFIG.CHAT_ID, "text": chunk, "parse_mode": "Markdown"}); await asyncio.sleep(0.5)
                    except: pass

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(RsiBot().run())
