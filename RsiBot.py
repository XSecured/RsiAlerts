import asyncio
import aiohttp
import logging
import os
import time
import json
import random
import ssl
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any, Set
from concurrent.futures import ProcessPoolExecutor
from functools import partial, wraps

# 3rd Party Libs
import numpy as np
import talib
import redis.asyncio as aioredis

# === CONFIGURATION ===

@dataclass(frozen=True)
class AppConfig:
    # API Endpoints
    BINANCE_FUTURES_API: str = "https://fapi.binance.com/fapi/v1"
    BINANCE_SPOT_API: str = "https://api.binance.com/api/v3"
    
    # Resources
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Strategy Params
    RSI_PERIOD: int = 14
    BB_LENGTH: int = 34
    BB_STDDEV: float = 2.0
    
    # Thresholds
    UPPER_TOUCH_THRESHOLD: float = 0.02
    LOWER_TOUCH_THRESHOLD: float = 0.02
    MIDDLE_TOUCH_THRESHOLD: float = 0.035
    
    # Limits
    CANDLE_LIMIT: int = 60
    MIN_CANDLES_TALIB: int = 36
    BATCH_SIZE: int = 30  
    
    # Logic Toggles
    TIMEFRAMES: Dict[str, bool] = field(default_factory=lambda: {
        '3m': False, '5m': False, '15m': True, '30m': True, 
        '1h': True, '2h': True, '4h': True, '1d': True, '1w': True
    })
    
    MIDDLE_BAND_ENABLED: Dict[str, bool] = field(default_factory=lambda: {
        '3m': False, '5m': False, '15m': True, '30m': True, 
        '1h': True, '2h': True, '4h': True, '1d': True, '1w': True
    })
    
    # Cache Settings
    CACHE_TTL: Dict[str, int] = field(default_factory=lambda: {
        '4h': 15000, '1d': 88000, '1w': 605000
    })
    CACHED_TFS: Set[str] = field(default_factory=lambda: {'4h', '1d', '1w'})
    RELAXED_LIMIT_TFS: Set[str] = field(default_factory=lambda: {'1d', '1w'})

    # Proxy Sources
    PROXY_SOURCES: List[str] = field(default_factory=lambda: [
        "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt",
        "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt"
    ])

CONFIG = AppConfig()

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RsiBot")

# === DATA STRUCTURES ===

# FIX: Converted to dataclass with slots=True to allow asdict() usage while keeping memory low
@dataclass(slots=True)
class ScanResult:
    symbol: str
    timeframe: str
    rsi: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    touch_type: str
    timestamp: str
    hot: bool
    direction: Optional[str]
    market_type: str

# === UTILS & DECORATORS ===

def retry_async(retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ssl.SSLError) as e:
                    last_exception = e
                    if attempt == retries:
                        break
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            return None 
        return wrapper
    return decorator

# === CORE SERVICES ===

class ProxyManager:
    def __init__(self):
        self.proxies: List[str] = []
        self.blacklist: Set[str] = set()
        self._lock = asyncio.Lock()
        
        raw_blocked = os.getenv("PROXY_BLOCK_PERM", "")
        self.env_blocked = {p.strip() for p in raw_blocked.split(",") if p.strip()}

    async def fetch_proxies(self):
        raw_proxies = set()
        async with aiohttp.ClientSession() as session:
            for source in CONFIG.PROXY_SOURCES:
                try:
                    async with session.get(source, timeout=10) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            for line in text.splitlines():
                                p = line.strip()
                                if p and p not in self.env_blocked:
                                    if "://" not in p: p = f"http://{p}"
                                    raw_proxies.add(p)
                except Exception as e:
                    logger.warning(f"Failed to fetch proxy list from {source}: {e}")
        
        async with self._lock:
            self.proxies = list(raw_proxies)
            self.blacklist.clear()
            random.shuffle(self.proxies)
            logger.info(f"Proxy Pool Refreshed: {len(self.proxies)} proxies available.")

    async def get_proxy(self) -> Optional[str]:
        async with self._lock:
            if not self.proxies: return None
            candidates = [p for p in self.proxies if p not in self.blacklist]
            if not candidates:
                self.blacklist.clear()
                candidates = self.proxies
            return random.choice(candidates[:50] if len(candidates) > 50 else candidates)

    async def report_failure(self, proxy: str):
        async with self._lock:
            self.blacklist.add(proxy)

class NetworkClient:
    def __init__(self, proxy_manager: ProxyManager):
        self.proxy_manager = proxy_manager
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        conn = aiohttp.TCPConnector(
            limit=100, 
            limit_per_host=20,
            ttl_dns_cache=300,
            ssl=False
        )
        timeout = aiohttp.ClientTimeout(total=20, connect=10, sock_read=10)
        self.session = aiohttp.ClientSession(connector=conn, timeout=timeout)

    async def close(self):
        if self.session:
            await self.session.close()

    @retry_async(retries=2, delay=0.5)
    async def request(self, method: str, url: str, params: dict = None) -> Any:
        proxy = await self.proxy_manager.get_proxy()
        try:
            async with self.session.request(
                method, url, params=params, proxy=proxy
            ) as resp:
                if resp.status in [403, 429, 451]:
                    if proxy: await self.proxy_manager.report_failure(proxy)
                    raise aiohttp.ClientError(f"Bad status {resp.status}")
                
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            if proxy: await self.proxy_manager.report_failure(proxy)
            raise e

class CacheService:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self):
        try:
            self.redis = await aioredis.from_url(
                CONFIG.REDIS_URL, encoding="utf-8", decode_responses=True
            )
            await self.redis.ping()
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis = None

    async def close(self):
        if self.redis: await self.redis.close()

    def _key(self, tf: str, open_time: str) -> str:
        return f"bot:scan:{tf}:{open_time}"

    async def get_results(self, tf: str, open_time: str) -> Optional[List[Dict]]:
        if not self.redis: return None
        try:
            data = await self.redis.get(self._key(tf, open_time))
            return json.loads(data) if data else None
        except Exception: return None

    async def set_results(self, tf: str, open_time: str, data: List[Dict]):
        if not self.redis: return
        ttl = CONFIG.CACHE_TTL.get(tf, 3600)
        try:
            await self.redis.set(self._key(tf, open_time), json.dumps(data), ex=ttl)
        except Exception: pass

    async def get_state(self) -> Dict[str, str]:
        if not self.redis: return {}
        try:
            data = await self.redis.get("bot:sent_state")
            return json.loads(data) if data else {}
        except Exception: return {}

    async def set_state(self, state: Dict[str, str]):
        if not self.redis: return
        try:
            await self.redis.set("bot:sent_state", json.dumps(state))
        except Exception: pass

class TechnicalAnalysisEngine:
    def __init__(self):
        self.executor = ProcessPoolExecutor(max_workers=4)

    def shutdown(self):
        self.executor.shutdown()

    @staticmethod
    def _calculate_cpu_bound(closes: List[float]) -> Tuple[float, float, float, float]:
        try:
            np_closes = np.array(closes, dtype=float)
            rsi_arr = talib.RSI(np_closes, timeperiod=CONFIG.RSI_PERIOD)
            upper, middle, lower = talib.BBANDS(
                rsi_arr, 
                timeperiod=CONFIG.BB_LENGTH,
                nbdevup=CONFIG.BB_STDDEV, 
                nbdevdn=CONFIG.BB_STDDEV, 
                matype=0
            )
            idx = -2
            if len(np_closes) < abs(idx): return (np.nan,)*4
            return (rsi_arr[idx], upper[idx], middle[idx], lower[idx])
        except Exception:
            return (np.nan,)*4

    async def analyze(self, closes: List[float]) -> Tuple[float, float, float, float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, self._calculate_cpu_bound, closes
        )

class NotificationService:
    def __init__(self):
        self.base_url = f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage"

    async def send(self, message: str, session: aiohttp.ClientSession):
        if not CONFIG.TELEGRAM_TOKEN or not CONFIG.TELEGRAM_CHAT_ID: return
        payload = {
            'chat_id': CONFIG.TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }
        try:
            async with session.post(self.base_url, json=payload) as resp:
                if resp.status != 200:
                    logger.error(f"Telegram failed: {resp.status}")
        except Exception as e:
            logger.error(f"Telegram connection error: {e}")

# === MAIN BOT ===

class RsiBot:
    def __init__(self):
        self.proxy_manager = ProxyManager()
        self.network = NetworkClient(self.proxy_manager)
        self.cache = CacheService()
        self.ta_engine = TechnicalAnalysisEngine()
        self.notifier = NotificationService()
        self.hot_coins: Set[str] = set()

    async def initialize(self):
        await self.cache.connect()
        await self.proxy_manager.fetch_proxies()
        await self.network.start()

    async def shutdown(self):
        await self.network.close()
        await self.cache.close()
        self.ta_engine.shutdown()

    async def get_symbols(self) -> Tuple[List[str], List[str]]:
        f_data = await self.network.request('GET', f"{CONFIG.BINANCE_FUTURES_API}/exchangeInfo")
        s_data = await self.network.request('GET', f"{CONFIG.BINANCE_SPOT_API}/exchangeInfo")
        
        f_symbols = set()
        if f_data:
            for s in f_data.get('symbols', []):
                if s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING':
                    f_symbols.add(s['symbol'])
        
        s_symbols = set()
        if s_data:
            for s in s_data.get('symbols', []):
                if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING':
                    s_symbols.add(s['symbol'])
        
        s_unique = list(s_symbols - f_symbols)
        return list(f_symbols), sorted(s_unique)

    async def get_candles(self, symbol: str, interval: str, is_futures: bool) -> List[float]:
        url = f"{CONFIG.BINANCE_FUTURES_API}/klines" if is_futures else f"{CONFIG.BINANCE_SPOT_API}/klines"
        data = await self.network.request('GET', url, params={
            'symbol': symbol, 'interval': interval, 'limit': CONFIG.CANDLE_LIMIT
        })
        if not data: return []
        return [float(k[4]) for k in data]

    async def update_volatility_rankings(self, all_symbols: List[Tuple[str, bool]]):
        logger.info("Updating volatility rankings...")
        scores = []
        sem = asyncio.Semaphore(50) 

        async def check_vol(sym, is_fut):
            async with sem:
                closes = await self.get_candles(sym, '1h', is_fut)
                if len(closes) >= 24:
                    returns = [(closes[i] - closes[i-1])/closes[i-1] for i in range(1, len(closes))]
                    if returns:
                        return (sym, float(np.std(returns)))
            return None

        tasks = [check_vol(s, f) for s, f in all_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in results:
            if r and isinstance(r, tuple): scores.append(r)
        
        scores.sort(key=lambda x: x[1], reverse=True)
        self.hot_coins = {x[0] for x in scores[:60]}
        logger.info(f"Top Volatile Coins Updated: {len(self.hot_coins)}")

    async def process_symbol_timeframe(
        self, symbol: str, timeframe: str, is_futures: bool
    ) -> Optional[ScanResult]:
        
        closes = await self.get_candles(symbol, timeframe, is_futures)
        
        min_req = CONFIG.MIN_CANDLES_TALIB
        if timeframe not in CONFIG.RELAXED_LIMIT_TFS:
             if len(closes) < CONFIG.CANDLE_LIMIT: return None
        elif len(closes) < min_req:
            return None

        rsi, upper, middle, lower = await self.ta_engine.analyze(closes)
        if np.isnan(rsi): return None

        touch_type = None
        direction = None
        
        if rsi >= upper * (1 - CONFIG.UPPER_TOUCH_THRESHOLD):
            touch_type = "UPPER"
        elif rsi <= lower * (1 + CONFIG.LOWER_TOUCH_THRESHOLD):
            touch_type = "LOWER"
        elif CONFIG.MIDDLE_BAND_ENABLED.get(timeframe):
            if abs(rsi - middle) <= middle * CONFIG.MIDDLE_TOUCH_THRESHOLD:
                touch_type = "MIDDLE"
                direction = "from above" if rsi > middle else "from below"

        if touch_type:
            now = datetime.utcnow()
            return ScanResult(
                symbol=symbol,
                timeframe=timeframe,
                rsi=round(rsi, 2),
                bb_upper=upper,
                bb_middle=middle,
                bb_lower=lower,
                touch_type=touch_type,
                timestamp=now.strftime('%Y-%m-%d %H:%M:%S UTC'),
                hot=(symbol in self.hot_coins),
                direction=direction,
                market_type='FUTURES' if is_futures else 'SPOT'
            )
        return None

    async def run_scan(self):
        start_ts = time.time()
        sent_state = await self.cache.get_state()
        f_syms, s_syms = await self.get_symbols()
        all_pairs = [(s, True) for s in f_syms] + [(s, False) for s in s_syms]
        
        logger.info(f"Scanning {len(all_pairs)} pairs.")
        await self.update_volatility_rankings(all_pairs)

        now = datetime.utcnow()
        tf_open_times = {}
        tf_min = {'15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '1d': 1440, '1w': 10080}
        to_alert_cache_tfs = set()
        
        for tf, enabled in CONFIG.TIMEFRAMES.items():
            if not enabled or tf not in tf_min: continue
            
            total_mins = now.hour * 60 + now.minute
            mins = tf_min[tf]
            intervals = total_mins // mins
            current_open = (now.replace(hour=0, minute=0, second=0, microsecond=0) + 
                          timedelta(minutes=intervals*mins)).isoformat()
            
            tf_open_times[tf] = current_open
            
            if tf in CONFIG.CACHED_TFS:
                if sent_state.get(tf) != current_open:
                    to_alert_cache_tfs.add(tf)
        
        active_tfs = [t for t, e in CONFIG.TIMEFRAMES.items() if e]
        uncached_tfs = [t for t in active_tfs if t not in CONFIG.CACHED_TFS]
        cached_tfs_to_scan = []
        
        cached_results = {}
        for tf in CONFIG.CACHED_TFS:
            if tf in active_tfs:
                res = await self.cache.get_results(tf, tf_open_times[tf])
                if res:
                    cached_results[tf] = [ScanResult(**r) for r in res]
                else:
                    cached_tfs_to_scan.append(tf)

        tfs_to_process_fresh = uncached_tfs + cached_tfs_to_scan
        final_results = []
        
        if tfs_to_process_fresh:
            logger.info(f"Fresh scanning TFs: {tfs_to_process_fresh}")
            jobs = []
            for tf in tfs_to_process_fresh:
                for sym, is_fut in all_pairs:
                    jobs.append((sym, tf, is_fut))
            
            sem = asyncio.Semaphore(CONFIG.BATCH_SIZE)
            
            async def worker(job):
                async with sem:
                    try:
                        return await self.process_symbol_timeframe(*job)
                    except Exception: return None

            results = await asyncio.gather(*[worker(j) for j in jobs], return_exceptions=True)
            
            fresh_grouped = {tf: [] for tf in tfs_to_process_fresh}
            for res in results:
                if isinstance(res, ScanResult):
                    final_results.append(res)
                    # FIX: asdict now works because ScanResult is a dataclass
                    fresh_grouped[res.timeframe].append(asdict(res))

            for tf, data in fresh_grouped.items():
                if tf in CONFIG.CACHED_TFS:
                    await self.cache.set_results(tf, tf_open_times[tf], data)
        
        for tf, items in cached_results.items():
            final_results.extend(items)

        alerts_to_send = []
        final_results.sort(key=lambda x: (x.timeframe, x.symbol))
        
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in final_results:
            grouped[r.timeframe].append(r)
            
        for tf, items in grouped.items():
            should_send = (tf in uncached_tfs) or (tf in to_alert_cache_tfs)
            if not should_send: continue
                
            msg = self.format_message(tf, items)
            alerts_to_send.append(msg)
            if tf in CONFIG.CACHED_TFS:
                sent_state[tf] = tf_open_times[tf]

        if alerts_to_send:
            for msg_chunk in alerts_to_send:
                for chunk in self.split_msg(msg_chunk):
                    await self.notifier.send(chunk, self.network.session)
        else:
            logger.info("No new alerts to trigger.")

        await self.cache.set_state(sent_state)
        logger.info(f"Run completed in {time.time() - start_ts:.2f}s")

    def format_message(self, timeframe: str, results: List[ScanResult]) -> str:
        if not results: return ""
        header = f"*🔍 BB Touches on {timeframe} ({len(results)} pairs)*\n"
        
        upper = [r for r in results if r.touch_type == 'UPPER']
        middle = [r for r in results if r.touch_type == 'MIDDLE']
        lower = [r for r in results if r.touch_type == 'LOWER']
        
        lines = [header]
        def _fmt(r: ScanResult):
            icon = "🔥" if r.hot else ""
            mkt = "[]" if r.market_type == 'FUTURES' else "[]"
            extra = f" ({'🔻' if r.direction == 'from above' else '🔹'})" if r.touch_type == "MIDDLE" else ""
            return f"• *{r.symbol}* {mkt} RSI:{r.rsi:.1f}{extra} {icon}"

        if upper:
            lines.append("\n*⬆️ UPPER BAND:*")
            lines.extend([_fmt(r) for r in upper])
        if middle:
            lines.append("\n*🔶 MIDDLE BAND:*")
            lines.extend([_fmt(r) for r in middle])
        if lower:
            lines.append("\n*⬇️ LOWER BAND:*")
            lines.extend([_fmt(r) for r in lower])
            
        lines.append(f"\n_Generated: {datetime.utcnow().strftime('%H:%M UTC')}_")
        return "\n".join(lines)

    def split_msg(self, text: str, limit: int = 4000) -> List[str]:
        if len(text) <= limit: return [text]
        chunks = []
        curr = ""
        for line in text.split('\n'):
            if len(curr) + len(line) + 1 > limit:
                chunks.append(curr)
                curr = line
            else:
                curr += "\n" + line if curr else line
        if curr: chunks.append(curr)
        return chunks

async def main():
    bot = RsiBot()
    try:
        await bot.initialize()
        await bot.run_scan()
    except Exception as e:
        logger.critical(f"Fatal Bot Error: {e}", exc_info=True)
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
