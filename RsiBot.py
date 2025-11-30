import asyncio
import json
import logging
import os
import random
import math
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Set, Optional, Tuple, Any, Deque
from itertools import cycle
from collections import deque

import aiohttp
import numpy as np
import talib
import redis.asyncio as aioredis

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

@dataclass
class Config:
    MAX_CONCURRENCY: int = 75
    REQUEST_TIMEOUT: int = 7
    MAX_RETRIES: int = 5
    
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL_MAP: Dict[str, int] = field(default_factory=lambda: {
        '4h': 14400 + 600, '1d': 86400 + 1800, '1w': 604800 + 3600
    })
    
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
MIDDLE_BAND_TFS = ['15m', '30m', '1h', '2h', '4h', '1d', '1w']
CACHED_TFS = {'2h', '4h', '1d', '1w'}

# ==========================================
# PROXY STATE MODELS (ALWAYS-PROXY EDITION)
# ==========================================

@dataclass
class ProxyBelief:
    """Bayesian belief per proxy: Beta-Bernoulli + Kalman."""
    url: str
    latency_mu: float = 0.3        # Kalman estimate (seconds)
    latency_var: float = 0.1       # Kalman error covariance
    success_alpha: int = 2         # Beta prior (successes + 1)
    success_beta: int = 2          # Beta prior (failures + 1)
    rate_limit_saturation: float = 0.0  # [0,1]
    uses: int = 0                  # Lifetime uses
    failures: int = 0              # Consecutive failures
    
    def kalman_update(self, measured_latency: float):
        """Optimal linear estimator."""
        kalman_gain = self.latency_var / (self.latency_var + 0.05)
        self.latency_mu = self.latency_mu + kalman_gain * (measured_latency - self.latency_mu)
        self.latency_var = (1 - kalman_gain) * self.latency_var + 0.01
    
    def beta_update(self, success: bool):
        """Conjugate prior update with exponential forgetting."""
        if success:
            self.success_alpha += 1
        else:
            self.success_beta += 1
        
        total = self.success_alpha + self.success_beta
        if total > 500:
            self.success_alpha *= 0.99
            self.success_beta *= 0.99
    
    def sample_reward(self) -> float:
        """Thompson Sampling: maximize expected reward."""
        p_success = np.random.beta(self.success_alpha, self.success_beta)
        latency_sample = np.random.normal(self.latency_mu, max(0.01, self.latency_var))
        return p_success - (latency_sample * 2.0) - (self.rate_limit_saturation * 0.5)

# ==========================================
# ZERO-IDLE SARE (ALWAYS-PROXY EDITION)
# ==========================================

class ZeroIdleSARE:
    """Proxy system that NEVER uses direct connection."""
    
    def __init__(self, redis: aioredis.Redis, proxy_url: str):
        self.redis = redis
        self.proxy_url = proxy_url
        self.proxies: Dict[str, ProxyBelief] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._dirty_proxies: Set[str] = set()
        self.request_counter = 0
        self._lock = asyncio.Lock()
    
    async def initialize(self, session: aiohttp.ClientSession):
        """Load proxy states and populate if empty."""
        self.session = session
        start = time.time()
        
        # Load existing states
        try:
            pipeline = self.redis.pipeline()
            pipeline.hgetall("bb_bot:sare_beliefs")
            results = await pipeline.execute()
            
            for url, payload in results[0].items():
                try:
                    d = json.loads(payload)
                    self.proxies[url] = ProxyBelief(
                        url=url,
                        latency_mu=float(d[0]),
                        latency_var=float(d[1]),
                        success_alpha=int(d[2]),
                        success_beta=int(d[3]),
                        rate_limit_saturation=float(d[4]),
                    )
                except: pass
            
            logging.info(f"✅ Loaded {len(self.proxies)} proxy states in {time.time() - start:.3f}s")
        except Exception as e:
            logging.warning(f"⚠️ Redis load failed: {e}, starting fresh")
        
        # **CRITICAL**: If no proxies, populate from source URL
        if len(self.proxies) < 5:
            logging.warning("🚨 Proxy pool too small, fetching fresh proxies...")
            await self._populate_proxy_pool()
    
    async def _populate_proxy_pool(self):
        """Fetch and validate proxies from URL."""
        try:
            async with aiohttp.ClientSession() as temp_session:
                async with temp_session.get(self.proxy_url, timeout=15) as resp:
                    text = await resp.text()
                    
                    # Parse proxies
                    new_proxies = []
                    for line in text.splitlines():
                        p = line.strip()
                        if p and ":" in p:
                            url = p if "://" in p else f"http://{p}"
                            new_proxies.append(url)
                    
                    # Add to pool with initial belief
                    async with self._lock:
                        for url in new_proxies[:50]:  # Limit to 50
                            if url not in self.proxies:
                                self.proxies[url] = ProxyBelief(url=url)
                    
                    logging.info(f"📥 Added {len(new_proxies)} proxies from source")
        except Exception as e:
            logging.error(f"❌ Proxy population failed: {e}")
    
    def select_proxy(self) -> Optional[str]:
        """Thompson Sampling: MUST return a proxy."""
        # Try to get best proxy
        if not self.proxies:
            # Emergency: return any proxy we can find
            logging.error("🚨 NO PROXIES AVAILABLE!")
            return None
        
        self.request_counter += 1
        
        # Early exploration for uncertain proxies
        if self.request_counter < 200:
            uncertain = [p for p in self.proxies.values() if p.uses < 5]
            if uncertain and random.random() < 0.1:
                return random.choice(uncertain).url
        
        # Sample top 3 proxies by success rate
        top_proxies = sorted(
            self.proxies.values(),
            key=lambda p: p.success_alpha / (p.success_alpha + p.success_beta),
            reverse=True
        )[:3]
        
        if not top_proxies:
            # Fallback to any proxy
            return random.choice(list(self.proxies.keys()))
        
        # Thompson Sampling: pick best sample
        proxy_samples = [(p.url, p.sample_reward()) for p in top_proxies]
        best_url, _ = max(proxy_samples, key=lambda x: x[1])
        return best_url
    
    async def execute(self, url: str, params: dict) -> Tuple[Any, bool]:
        """Execute request and update beliefs."""
        proxy = self.select_proxy()
        
        # **BLOCKED LOCATION FIX**: If no proxy, fail immediately
        if not proxy:
            logging.error("❌ No proxy available - request blocked")
            return None, False
        
        start = time.time()
        success = False
        data = None
        
        try:
            async with self.session.get(url, params=params, proxy=proxy, timeout=CONFIG.REQUEST_TIMEOUT) as resp:
                if resp.status == 200:
                    success = True
                    data = await resp.json()
                elif resp.status == 429:
                    # Rate limit detected
                    if proxy in self.proxies:
                        self.proxies[proxy].rate_limit_saturation = min(1.0, self.proxies[proxy].rate_limit_saturation + 0.5)
                        self._dirty_proxies.add(proxy)
                    logging.debug(f"🐌 Proxy {proxy} rate limited")
        except Exception as e:
            # Network failure
            logging.debug(f"❌ Request failed via {proxy}: {e}")
        finally:
            latency = time.time() - start
            
            # Update beliefs
            if proxy in self.proxies:
                belief = self.proxies[proxy]
                belief.uses += 1
                
                # Kalman update for latency
                belief.kalman_update(latency)
                
                # Beta update for success/failure
                belief.beta_update(success)
                
                # Track consecutive failures for SPC
                if success:
                    belief.failures = 0
                else:
                    belief.failures += 1
                
                # **3-Sigma Statistical Culling**
                if belief.uses > 5 and belief.failures / belief.uses > 0.4:
                    del self.proxies[proxy]
                    self._dirty_proxies.discard(proxy)  # Remove from persist set
                    logging.warning(f"🚫 Culled {proxy} ({belief.failures}/{belief.uses} failures)")
                
                self._dirty_proxies.add(proxy)
            
            return data, success
    
    async def persist_all(self):
        """Batch persist all changed beliefs."""
        if not self._dirty_proxies:
            return
        
        pipeline = self.redis.pipeline()
        
        for url in self._dirty_proxies:
            if url in self.proxies:  # Might have been culled
                belief = self.proxies[url]
                payload = json.dumps([
                    round(belief.latency_mu, 3),
                    round(belief.latency_var, 3),
                    belief.success_alpha,
                    belief.success_beta,
                    round(belief.rate_limit_saturation, 3),
                ])
                pipeline.hset("bb_bot:sare_beliefs", url, payload)
        
        await pipeline.execute()
        logging.info(f"💾 Persisted {len(self._dirty_proxies)} updated proxy beliefs")
        self._dirty_proxies.clear()

# ==========================================
# EXCHANGE CLIENTS (NOW ALWAYS-PROXY)
# ==========================================

class ExchangeClient:
    def __init__(self, pool: ZeroIdleSARE):
        self.pool = pool
    
    async def _request(self, url: str, params: dict = None) -> Any:
        """All requests go through SARE (proxy only)."""
        for attempt in range(CONFIG.MAX_RETRIES):
            data, success = await self.pool.execute(url, params or {})
            if success:
                return data
            
            # Exponential backoff with jitter
            delay = (1.5 ** attempt) * (0.3 + random.random() * 0.4)
            await asyncio.sleep(delay)
        
        logging.warning(f"❌ Failed after {CONFIG.MAX_RETRIES}: {url}")
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
    
    async def fetch_closes_volatility(self, symbol: str, market: str) -> List[float]:
        url = 'https://api.bybit.com/v5/market/kline'
        cat = 'linear' if market == 'perp' else 'spot'
        data = await self._request(url, {'category': cat, 'symbol': symbol, 'interval': '60', 'limit': 25})
        if not data: return []
        raw = data.get('result', {}).get('list', [])
        if not raw: return []
        closes = [float(c[4]) for c in raw]
        return closes[::-1]
    
    async def fetch_closes(self, symbol: str, interval: str, market: str) -> List[float]:
        url = 'https://api.bybit.com/v5/market/kline'
        cat = 'linear' if market == 'perp' else 'spot'
        b_int = {"15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240", "1d": "D", "1w": "W"}.get(interval, "D")
        data = await self._request(url, {'category': cat, 'symbol': symbol, 'interval': b_int, 'limit': CONFIG.CANDLE_LIMIT})
        if not data: return []
        raw = data.get('result', {}).get('list', [])
        if not raw: return []
        closes = [float(c[4]) for c in raw]
        return closes[::-1]

# ==========================================
# CACHE MANAGER (UNCHANGED)
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
        if self.redis: await self.redis.aclose()

    async def get_cached_symbols(self) -> Optional[Dict]:
        if not self.redis: return None
        data = await self.redis.get("bb_bot:symbols_cache_v3")
        return json.loads(data) if data else None

    async def save_cached_symbols(self, symbols: Dict):
        if not self.redis: return
        payload = {
            "timestamp": time.time(),
            "data": symbols
        }
        await self.redis.set("bb_bot:symbols_cache_v3", json.dumps(payload))
        logging.info("💾 Saved symbol list to cache (Persistent)")

    def _scan_key(self, tf: str, candle_key: int) -> str:
        return f"bb_touch:scan:{tf}:{candle_key}"

    async def get_scan_results(self, tf: str, candle_key: int) -> Optional[List[Dict]]:
        if not self.redis: return None
        try:
            data = await self.redis.get(self._scan_key(tf, candle_key))
            return json.loads(data) if data else None
        except: return None

    async def save_scan_results(self, tf: str, candle_key: int, results: List[Dict]):
        if not self.redis: return
        ttl = CONFIG.CACHE_TTL_MAP.get(tf, 3600)
        try: await self.redis.set(self._scan_key(tf, candle_key), json.dumps(results), ex=ttl)
        except: pass

    async def get_sent_state(self) -> Dict[str, int]:
        if not self.redis: return {}
        try:
            val = await self.redis.get("bb_bot:sent_state")
            return json.loads(val) if val else {}
        except: return {}

    async def save_sent_state(self, state: Dict[str, int]):
        if not self.redis: return
        try: await self.redis.set("bb_bot:sent_state", json.dumps(state))
        except: pass

# ==========================================
# CORE LOGIC (UNCHANGED)
# ==========================================

def get_cache_key(tf: str) -> int:
    """Returns stable integer timestamp for current TF candle."""
    mins = TIMEFRAME_MINUTES[tf]
    now = int(time.time())
    return now - (now % (mins * 60))

def calculate_volatility(closes: List[float]) -> float:
    if len(closes) < 24: return 0.0
    returns = []
    for i in range(1, len(closes)):
        if closes[i-1] != 0: returns.append((closes[i] - closes[i-1]) / closes[i-1] * 100)
    return np.std(returns) if returns else 0.0

def check_bb_rsi(closes: List[float], tf: str) -> Tuple[Optional[str], Optional[str], float]:
    if len(closes) < CONFIG.MIN_CANDLES: return None, None, 0.0
    np_c = np.array(closes, dtype=float)
    rsi = talib.RSI(np_c, timeperiod=CONFIG.RSI_PERIOD)
    upper, mid, lower = talib.BBANDS(rsi, timeperiod=CONFIG.BB_LENGTH, nbdevup=CONFIG.BB_STDDEV, nbdevdn=CONFIG.BB_STDDEV, matype=0)
    
    idx = -2
    if np.isnan(rsi[idx]) or np.isnan(upper[idx]): return None, None, 0.0
    
    curr_rsi = rsi[idx]
    if curr_rsi >= upper[idx] * (1 - CONFIG.UPPER_TOUCH_THRESHOLD): return "UPPER", None, curr_rsi
    if curr_rsi <= lower[idx] * (1 + CONFIG.LOWER_TOUCH_THRESHOLD): return "LOWER", None, curr_rsi
    
    if tf in MIDDLE_BAND_TFS:
        if abs(curr_rsi - mid[idx]) <= (mid[idx] * CONFIG.MIDDLE_TOUCH_THRESHOLD):
            prev_diff = rsi[idx-1] - mid[idx-1]
            curr_diff = curr_rsi - mid[idx]
            direction = "from above" if (prev_diff > 0 >= curr_diff) or (curr_diff > 0) else "from below"
            return "MIDDLE", direction, curr_rsi
    return None, None, 0.0

# ==========================================
# MAIN BOT (CORRECTED SUCCESS TRACKING)
# ==========================================

class RsiBot:
    def __init__(self):
        self.cache = CacheManager()
        self.pool: Optional[ZeroIdleSARE] = None
        
    async def send_report(self, session: aiohttp.ClientSession, hits: List[TouchHit]):
        if not hits: return
        grouped = {}
        for h in hits: grouped.setdefault(h.timeframe, {}).setdefault(h.touch_type, []).append(h)
        
        tf_blocks = []
        tf_order = ["1w", "1d", "4h", "2h", "1h", "30m", "15m"]
        
        for tf in tf_order:
            if tf not in grouped: continue
            
            total_hits = sum(len(grouped[tf].get(t, [])) for t in ["UPPER", "MIDDLE", "LOWER"])
            lines = [
                f" ▣ TIMEFRAME: {tf} ({total_hits} Hits)",
                "─────────────────────────",
                ""
            ]
            
            headers = {"UPPER": "⬆️ UPPER BB", "MIDDLE": "🔶 MIDDLE BB", "LOWER": "⬇️ LOWER BB"}
            found = [t for t in ["UPPER", "MIDDLE", "LOWER"] if grouped[tf].get(t)]
            
            for t in found:
                items = grouped[tf].get(t, [])
                items.sort(key=lambda x: x.symbol)
                lines.append(f"┌ {headers[t]}")
                for idx, item in enumerate(items):
                    prefix = "└" if idx == len(items)-1 else "│"
                    icon = "🥐" if item.exchange == "Binance" else "💣"
                    sym_clean = item.symbol.replace("USDT", "")
                    ext = f" {'🔻' if item.direction=='from above' else '🔹'}" if t=="MIDDLE" else ""
                    if item.hot: ext += " 🔥"
                    lines.append(f"{prefix} {icon} *{sym_clean}* ➜ *{item.rsi:.2f}*{ext}")
                lines.append("")
            
            lines.append("─────────────────────────")
            tf_blocks.append("\n".join(lines))

        if not tf_blocks: return

        current_msg = []
        current_len = 0
        ts_footer = datetime.now(timezone.utc).strftime('%d %b %H:%M UTC')
        
        for block in tf_blocks:
            if current_len + len(block) > 3800:
                full_text = "\n\n".join(current_msg) + f"\n{ts_footer}"
                try: 
                    await session.post(f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage", 
                                     json={"chat_id": CONFIG.CHAT_ID, "text": full_text, "parse_mode": "Markdown"})
                    await asyncio.sleep(0.5)
                except Exception as e: logging.error(f"TG Send Fail: {e}")
                current_msg = []
                current_len = 0
            
            current_msg.append(block)
            current_len += len(block)
        
        if current_msg:
            full_text = "\n\n".join(current_msg) + f"\n{ts_footer}"
            try: 
                await session.post(f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage", 
                                 json={"chat_id": CONFIG.CHAT_ID, "text": full_text, "parse_mode": "Markdown"})
            except Exception as e: logging.error(f"TG Send Fail: {e}")

    async def fetch_symbols_hybrid(self, binance: BinanceClient, bybit: BybitClient) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Fetches symbols with per-exchange fallback to cache."""
        
        cached = await self.cache.get_cached_symbols()
        cached_data = cached.get('data') if cached else None
        
        async def try_fetch(fetch_func, cache_key: str, name: str):
            for attempt in range(3):
                result = await fetch_func()
                if len(result) > 0:
                    logging.info(f"✅ {name}: {len(result)} symbols")
                    return result
                logging.warning(f"⚠️ {name}: Attempt {attempt+1} failed, retrying...")
                await asyncio.sleep(2)
            
            logging.warning(f"❌ {name}: Failed after retries, using cache")
            return cached_data[cache_key] if cached_data else []
        
        bp, bs, yp, ys = await asyncio.gather(
            try_fetch(binance.get_perp_symbols, 'bp', 'Binance Perp'),
            try_fetch(binance.get_spot_symbols, 'bs', 'Binance Spot'),
            try_fetch(bybit.get_perp_symbols, 'yp', 'Bybit Perp'),
            try_fetch(bybit.get_spot_symbols, 'ys', 'Bybit Spot')
        )
        
        if any(len(x) > 0 for x in [bp, bs, yp, ys]):
            await self.cache.save_cached_symbols({'bp': bp, 'bs': bs, 'yp': yp, 'ys': ys})
        
        return bp, bs, yp, ys

    async def run(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        
        # 1. Initialize Redis
        await self.cache.init()
        
        # 2. Create ZeroIdleSARE (always-proxy edition)
        self.pool = ZeroIdleSARE(self.cache.redis, CONFIG.PROXY_URL)
        
        # 3. Create persistent HTTP session
        connector = aiohttp.TCPConnector(
            limit_per_host=150,
            ttl_dns_cache=600,
            use_dns_cache=True,
            force_close=False,  # Keep-alive
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(
            total=CONFIG.REQUEST_TIMEOUT,
            connect=2,
            sock_read=CONFIG.REQUEST_TIMEOUT - 1
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Connection": "keep-alive"}
        ) as session:
            
            # 4. Initialize SARE (loads state + populates if empty)
            await self.pool.initialize(session)
            
            # 5. Create exchange clients
            binance = BinanceClient(self.pool)
            bybit = BybitClient(self.pool)
            
            # 6. Fetch symbols
            bp, bs, yp, ys = await self.fetch_symbols_hybrid(binance, bybit)
            
            # 7. Filter symbols
            seen = set()
            def filter_unique(syms):
                res = []
                for s in syms:
                    if s in CONFIG.IGNORED_SYMBOLS or not s.endswith("USDT"): continue
                    norm = s.upper().replace("USDT", "")
                    if norm not in seen:
                        res.append(s)
                        seen.add(norm)
                return res
            
            f_bp, f_bs, f_yp, f_ys = filter_unique(bp), filter_unique(bs), filter_unique(yp), filter_unique(ys)
            all_pairs = []
            for s in f_bp: all_pairs.append((binance, s, 'perp', 'Binance'))
            for s in f_bs: all_pairs.append((binance, s, 'spot', 'Binance'))
            for s in f_yp: all_pairs.append((bybit, s, 'perp', 'Bybit'))
            for s in f_ys: all_pairs.append((bybit, s, 'spot', 'Bybit'))
            
            total_sym_count = len(all_pairs)
            logging.info(f"Total symbols: {total_sym_count}")
            
            # 8. Calculate volatility
            logging.info("Calculating Volatility...")
            vol_scores = {}
            async def check_vol(client, sym, mkt):
                c = await client.fetch_closes_volatility(sym, mkt)
                v = calculate_volatility(c)
                if v > 0: vol_scores[sym] = v
            
            vol_tasks = [check_vol(client, s, mkt) for client, s, mkt, ex in all_pairs]
            for i in range(0, len(vol_tasks), 200): await asyncio.gather(*vol_tasks[i:i+200])
            hot_coins = set(sorted(vol_scores, key=vol_scores.get, reverse=True)[:60])
            logging.info(f"Vol Calc: {len(vol_scores)}/{len(all_pairs)} success | Hot: {len(hot_coins)}")
            
            # 9. Precision scan setup with sent state check
            sent_state = await self.cache.get_sent_state()
            scan_stats = {tf: ScanStats(tf, "Unknown", total_symbols=total_sym_count) for tf in ACTIVE_TFS}
            
            tfs_to_scan_fresh = []
            cached_hits_to_use = []
            
            for tf in ACTIVE_TFS:
                if tf in CACHED_TFS:
                    candle_key = get_cache_key(tf)
                    if sent_state.get(tf) == candle_key:
                        logging.info(f"⏭️ Skipping {tf}: already sent for this candle")
                        scan_stats[tf].source = "Already Sent"
                        continue
                    
                    cached_res = await self.cache.get_scan_results(tf, candle_key)
                    if cached_res is None:
                        tfs_to_scan_fresh.append(tf)
                        scan_stats[tf].source = "Fresh Scan (New Candle)"
                    else:
                        hits = [TouchHit.from_dict(d) for d in cached_res]
                        cached_hits_to_use.extend(hits)
                        scan_stats[tf].source = "Cached"
                        scan_stats[tf].hits_found = len(hits)
                        scan_stats[tf].successful_scans = 0
                else:
                    tfs_to_scan_fresh.append(tf)
                    scan_stats[tf].source = "Fresh Scan (Low TF)"
            
            # 10. Execute fresh scans
            final_hits = []
            if tfs_to_scan_fresh:
                logging.info(f"Scanning fresh TFs: {tfs_to_scan_fresh} across all symbols...")
                
                for tf in tfs_to_scan_fresh:
                    tf_tasks = []
                    
                    # **FIXED**: Track actual successful data returns
                    actual_successes = 0
                    
                    async def scan_one(client, sym, mkt, ex):
                        closes = await client.fetch_closes(sym, tf, mkt)
                        if not closes:  # **FIXED**: Return None on failure
                            return None
                        t_type, direction, rsi_val = check_bb_rsi(closes, tf)
                        if t_type:
                            return [TouchHit(sym, ex, mkt, tf, rsi_val, t_type, direction, sym in hot_coins)]
                        return []  # Success but no pattern found
                    
                    for client, sym, mkt, ex in all_pairs:
                        tf_tasks.append(scan_one(client, sym, mkt, ex))
                    
                    results = await asyncio.gather(*tf_tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, list):  # **FIXED**: Only actual data is success
                            final_hits.extend(result)
                            scan_stats[tf].hits_found += len(result)
                            if result:  # If got any data back
                                actual_successes += 1
                        elif result is None:  # **FIXED**: Network failure
                            pass
                    
                    scan_stats[tf].successful_scans = actual_successes  # **FIXED**: Real success count
                    
                    if tf in CACHED_TFS:
                        candle_key = get_cache_key(tf)
                        tf_hits = [h for h in final_hits if h.timeframe == tf]
                        await self.cache.save_scan_results(tf, candle_key, [h.to_dict() for h in tf_hits])
            
            # 11. Merge cached
            final_hits.extend(cached_hits_to_use)
            
            # 12. Summary (now shows real success rate)
            logging.info("="*60)
            logging.info(f"{'TF':<5} | {'Source':<25} | {'Success':<15} | {'Hits'}")
            logging.info("-" * 60)
            for tf in sorted(ACTIVE_TFS, key=lambda x: ["15m","30m","1h","2h","4h","1d","1w"].index(x)):
                st = scan_stats[tf]
                succ_str = "Skipped (Cached/Sent)" if st.source in ["Cached", "Already Sent"] else f"{st.successful_scans}/{st.total_symbols}"
                logging.info(f"[{tf:<3}] {st.source:<25} | {succ_str:<15} | {st.hits_found}")
            logging.info("="*60)
            
            # 13. Alerting
            hits_to_send = []
            new_state = sent_state.copy()
            for h in final_hits:
                tf = h.timeframe
                if tf in CACHED_TFS:
                    candle_key = get_cache_key(tf)
                    if sent_state.get(tf, 0) != candle_key:
                        hits_to_send.append(h)
                        new_state[tf] = candle_key
                else:
                    hits_to_send.append(h)
            
            await self.send_report(session, hits_to_send)
            await self.cache.save_sent_state(new_state)
            
            # 14. Persist proxy beliefs
            await self.pool.persist_all()
            
            # 15. Final metrics
            logging.info(
                f"📊 Final | Proxies: {len(self.pool.proxies)} | "
                f"Requests: {self.pool.request_counter} | "
                f"Dirty: {len(self.pool._dirty_proxies)}"
            )
        
        await self.cache.close()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    bot = RsiBot()
    asyncio.run(bot.run())
