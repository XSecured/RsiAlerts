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
from typing import List, Dict, Set, Optional, Tuple, Any
from itertools import cycle
from enum import Enum

import aiohttp
import numpy as np
import talib
import redis.asyncio as aioredis

# ==========================================
# CONFIGURATION & CONSTANTS (UPDATED)
# ==========================================

@dataclass
class Config:
    MAX_CONCURRENCY: int = 50
    REQUEST_TIMEOUT: int = 7
    MAX_RETRIES: int = 3
    
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

    # Number of lookback candles for middle band direction analysis
    MIDDLE_BAND_LOOKBACK: int = 5

    # Weekly scan delay in seconds after Monday 00:00 UTC
    # Gives time for the new weekly candle to form its first data
    WEEKLY_SCAN_DELAY: int = 1800  # 30 minutes

    IGNORED_SYMBOLS: Set[str] = field(default_factory=lambda: {
        "USDPUSDT", "USD1USDT", "TUSDUSDT", "AEURUSDT", "USDCUSDT", "EURUSDT", "USDY"
    })

CONFIG = Config()

TIMEFRAME_MINUTES = {
    '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, 
    '1d': 1440, '1w': 10080
}

ACTIVE_TFS = ['4h', '1d', '1w']
MIDDLE_BAND_TFS = ['4h', '1d', '1w']
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
    direction: str = ""   # "bullish" or "bearish" for middle band hits
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
# ULTRA-ROBUST ASYNC PROXY POOL - PRODUCTION GRADE
# ==========================================

class ProxyState(Enum):
    ACTIVE = "active"
    COOLING = "cooling"
    BANNED = "banned"


@dataclass
class ProxyStats:
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    total_latency_ms: float = 0.0
    last_used: float = field(default_factory=time.time)
    last_success: float = 0.0
    last_failure: float = 0.0
    state: ProxyState = ProxyState.ACTIVE
    cooldown_until: float = 0.0

    @property
    def total_uses(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.8  # Optimistic for new proxies
        return self.successes / self.total_uses

    @property
    def avg_latency_ms(self) -> float:
        if self.successes == 0:
            return 9999.0
        return self.total_latency_ms / self.successes

    def compute_score(self) -> float:
        """Higher = better. Combines success rate, latency, freshness."""
        if self.state != ProxyState.ACTIVE:
            return 0.0

        score = self.success_rate

        # Exponential penalty for consecutive failures
        score *= (0.5 ** self.consecutive_failures)

        # Latency bonus (faster = better)
        if self.avg_latency_ms < 9999:
            latency_factor = max(0.1, 1.0 - (self.avg_latency_ms / 5000))
            score *= (0.6 + 0.4 * latency_factor)

        # Freshness bonus for untested proxies
        if self.total_uses < 3:
            score += 0.15

        return max(0.001, min(1.0, score))


class RobustProxyPool:
    """
    Production-grade async proxy pool with:
    - Weighted scoring selection
    - Circuit breaker pattern
    - Auto-retry with fallback
    - Multi-source aggregation
    - Background health maintenance
    - Guaranteed request delivery
    """

    PROXY_SOURCES = [
        "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"
    ]

    def __init__(
        self,
        max_pool_size: int = 20,
        min_pool_size: int = 15,
        max_consecutive_failures: int = 2,
        cooldown_seconds: float = 90.0,
        ban_after_uses: int = 8,
        ban_below_rate: float = 0.25,
        validation_concurrency: int = 100,
        background_refresh_interval: float = 180.0,
        request_timeout: float = 7.0,
        validation_timeout: float = 4.0,
        allow_direct_fallback: bool = False,
    ):
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.max_consecutive_failures = max_consecutive_failures
        self.cooldown_seconds = cooldown_seconds
        self.ban_after_uses = ban_after_uses
        self.ban_below_rate = ban_below_rate
        self.validation_concurrency = validation_concurrency
        self.background_refresh_interval = background_refresh_interval
        self.request_timeout = request_timeout
        self.validation_timeout = validation_timeout
        self.allow_direct_fallback = allow_direct_fallback

        self._proxies: Dict[str, ProxyStats] = {}
        self._lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._custom_sources: List[str] = []
        self._initialized = False

        # Stats tracking
        self._total_requests = 0
        self._successful_requests = 0
        self._direct_fallbacks = 0

    @property
    def active_proxies(self) -> List[str]:
        now = time.time()
        active = []
        for proxy, stats in self._proxies.items():
            if stats.state == ProxyState.ACTIVE:
                active.append(proxy)
            elif stats.state == ProxyState.COOLING and now > stats.cooldown_until:
                stats.state = ProxyState.ACTIVE
                stats.consecutive_failures = 0
                active.append(proxy)
        return active

    @property
    def pool_size(self) -> int:
        return len(self.active_proxies)

    @property
    def is_healthy(self) -> bool:
        return self.pool_size >= self.min_pool_size

    async def initialize(
        self,
        session: aiohttp.ClientSession,
        additional_sources: Optional[List[str]] = None,
        start_background_tasks: bool = True,
    ) -> bool:
        self._session = session

        if additional_sources:
            self._custom_sources = list(additional_sources)

        logging.info("🚀 Initializing Robust Proxy Pool...")
        await self._populate_pool()

        if start_background_tasks:
            self._start_background_refresh()

        self._initialized = True

        if self.pool_size > 0:
            logging.info(f"✅ Proxy Pool Ready: {self.pool_size} active proxies")
            return True
        else:
            logging.error("❌ No working proxies found!")
            return False

    async def shutdown(self):
        """Gracefully shutdown background tasks."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        logging.info("🛑 Proxy Pool shut down")

    async def _fetch_from_source(self, url: str) -> Set[str]:
        """Fetch proxies from a single source URL."""
        proxies = set()
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with self._session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    for line in text.splitlines():
                        p = line.strip()
                        if p and not p.startswith('#') and '.' in p:
                            if "://" not in p:
                                p = f"http://{p}"
                            proxies.add(p)
        except Exception as e:
            logging.debug(f"Source fetch failed ({url}): {e}")
        return proxies

    async def _validate_proxy(self, proxy: str) -> Tuple[str, bool, float]:
        """
        Validate a single proxy against Binance API.
        Returns: (proxy, is_valid, latency_ms)
        """
        start = time.time()
        try:
            timeout = aiohttp.ClientTimeout(total=self.validation_timeout)
            url = "https://fapi.binance.com/fapi/v1/time"
            
            async with self._session.get(url, proxy=proxy, timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "serverTime" in data:
                        latency_ms = (time.time() - start) * 1000
                        return proxy, True, latency_ms
        except:
            pass
        return proxy, False, 0.0

    async def _populate_pool(self):
        """Fetch and validate proxies from all sources."""
        all_sources = self.PROXY_SOURCES + self._custom_sources

        logging.info(f"📥 Fetching from {len(all_sources)} proxy sources...")

        fetch_tasks = [self._fetch_from_source(url) for url in all_sources]
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        all_raw = set()
        for result in results:
            if isinstance(result, set):
                all_raw.update(result)

        new_candidates = all_raw - set(self._proxies.keys())
        logging.info(f"🔎 Validating {len(new_candidates)} new proxy candidates...")

        if not new_candidates:
            return

        sem = asyncio.Semaphore(self.validation_concurrency)
        validated_count = 0

        async def validate_with_limit(proxy: str):
            async with sem:
                return await self._validate_proxy(proxy)

        tasks = [asyncio.create_task(validate_with_limit(p)) for p in new_candidates]

        for coro in asyncio.as_completed(tasks):
            try:
                proxy, is_valid, latency_ms = await coro
                if is_valid:
                    async with self._lock:
                        if proxy not in self._proxies:
                            self._proxies[proxy] = ProxyStats(
                                successes=1,
                                total_latency_ms=latency_ms,
                                last_success=time.time(),
                            )
                            validated_count += 1

                            if len(self.active_proxies) >= self.max_pool_size:
                                break
            except:
                pass

        for t in tasks:
            if not t.done():
                t.cancel()

        await asyncio.sleep(0.05)
        logging.info(f"✨ Added {validated_count} new proxies (total active: {self.pool_size})")

    def _start_background_refresh(self):
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._background_refresh_loop())

    async def _background_refresh_loop(self):
        """Periodically refresh and health-check the pool."""
        while True:
            try:
                await asyncio.sleep(self.background_refresh_interval)

                if self.pool_size < self.min_pool_size:
                    logging.warning(f"⚠️ Pool critically low ({self.pool_size}), refreshing...")
                    await self._populate_pool()

                await self._prune_old_banned()
                await self._spot_health_check()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Background refresh error: {e}")

    async def _prune_old_banned(self):
        """Remove long-banned proxies to free memory."""
        async with self._lock:
            cutoff = time.time() - 600
            to_remove = [
                p for p, s in self._proxies.items()
                if s.state == ProxyState.BANNED and s.last_failure < cutoff
            ]
            for p in to_remove:
                del self._proxies[p]
            if to_remove:
                logging.debug(f"🧹 Pruned {len(to_remove)} old banned proxies")

    async def _spot_health_check(self):
        """Quickly test a random sample of active proxies."""
        active = self.active_proxies
        if len(active) < 5:
            return

        sample = random.sample(active, min(5, len(active)))
        tasks = [self._validate_proxy(p) for p in sample]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        async with self._lock:
            for result in results:
                if isinstance(result, tuple):
                    proxy, is_valid, latency_ms = result
                    if proxy in self._proxies:
                        if is_valid:
                            self._proxies[proxy].successes += 1
                            self._proxies[proxy].total_latency_ms += latency_ms
                            self._proxies[proxy].consecutive_failures = 0
                        else:
                            self._record_failure(proxy)

    def _select_weighted(self) -> Optional[str]:
        """Select proxy using weighted random based on scores."""
        active = self.active_proxies
        if not active:
            return None

        scored = [(p, self._proxies[p].compute_score()) for p in active]
        total = sum(s for _, s in scored)

        if total <= 0:
            return random.choice(active)

        r = random.random() * total
        cumulative = 0.0
        for proxy, score in scored:
            cumulative += score
            if cumulative >= r:
                return proxy

        return scored[-1][0]

    async def get_proxy(self) -> Optional[str]:
        """Get the best available proxy."""
        proxy = self._select_weighted()
        if proxy:
            self._proxies[proxy].last_used = time.time()
        return proxy

    def _record_failure(self, proxy: str):
        """Record failure (internal, call within lock or sync context)."""
        if proxy not in self._proxies:
            return

        stats = self._proxies[proxy]
        stats.failures += 1
        stats.consecutive_failures += 1
        stats.last_failure = time.time()

        if stats.consecutive_failures >= self.max_consecutive_failures:
            stats.state = ProxyState.COOLING
            stats.cooldown_until = time.time() + self.cooldown_seconds
            logging.debug(f"⏸️ {proxy} cooling down ({self.cooldown_seconds}s)")

        if stats.total_uses >= self.ban_after_uses and stats.success_rate < self.ban_below_rate:
            stats.state = ProxyState.BANNED
            logging.warning(f"🚫 Banned {proxy} (rate: {stats.success_rate:.0%})")

    async def report_success(self, proxy: str, latency_ms: Optional[float] = None):
        """Report successful request."""
        async with self._lock:
            if proxy not in self._proxies:
                return
            stats = self._proxies[proxy]
            stats.successes += 1
            stats.consecutive_failures = 0
            stats.last_success = time.time()
            if latency_ms:
                stats.total_latency_ms += latency_ms
            if stats.state == ProxyState.COOLING:
                stats.state = ProxyState.ACTIVE

    async def report_failure(self, proxy: str):
        """Report failed request."""
        async with self._lock:
            self._record_failure(proxy)
            
    ############UNUSED BUILT IN CORE PROXY SYSTEM FUNCTIONS, PROXY SYSTEM ONLY WORK 100% WITH THEM
    '''async def fetch(
        self,
        url: str,
        method: str = "GET",
        max_retries: int = 8,
        base_delay: float = 0.05,
        **kwargs,
    ) -> Tuple[bool, Any]:
        """
        Fetch URL with automatic proxy rotation and retry.
        
        Returns:
            (True, response_data) on success
            (False, error_message) on failure
        """
        if not self._session:
            return False, "Session not initialized"

        self._total_requests += 1
        tried_proxies: Set[str] = set()
        last_error = "Unknown error"

        for attempt in range(max_retries):
            proxy = None
            available = set(self.active_proxies) - tried_proxies
            
            if available:
                proxy = self._select_from_set(available)
            elif self.active_proxies:
                proxy = self._select_weighted()

            if proxy is None and self.allow_direct_fallback:
                logging.debug("🔄 Attempting direct connection (no proxy)")
                self._direct_fallbacks += 1

            if proxy:
                tried_proxies.add(proxy)

            start_time = time.time()

            try:
                timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                async with self._session.request(
                    method, url, proxy=proxy, timeout=timeout, **kwargs
                ) as resp:
                    
                    latency_ms = (time.time() - start_time) * 1000

                    if resp.status == 200:
                        try:
                            data = await resp.json()
                        except:
                            data = await resp.text()

                        if proxy:
                            await self.report_success(proxy, latency_ms)

                        self._successful_requests += 1
                        return True, data

                    if resp.status in (403, 407, 429, 502, 503):
                        if proxy:
                            await self.report_failure(proxy)
                        last_error = f"HTTP {resp.status}"
                    else:
                        last_error = f"HTTP {resp.status}"

            except asyncio.TimeoutError:
                if proxy:
                    await self.report_failure(proxy)
                last_error = "Timeout"

            except (aiohttp.ClientProxyConnectionError, aiohttp.ClientHttpProxyError) as e:
                if proxy:
                    await self.report_failure(proxy)
                last_error = f"Proxy connection error"

            except aiohttp.ClientError as e:
                if proxy:
                    await self.report_failure(proxy)
                last_error = f"Client error: {type(e).__name__}"

            except Exception as e:
                if proxy:
                    await self.report_failure(proxy)
                last_error = f"Error: {type(e).__name__}"

            if attempt < max_retries - 1:
                delay = base_delay * (1.5 ** attempt) + random.uniform(0, 0.05)
                await asyncio.sleep(delay)

        return False, f"All {max_retries} attempts failed. Last: {last_error}"

    def _select_from_set(self, candidates: Set[str]) -> Optional[str]:
        """Select best proxy from a specific set."""
        if not candidates:
            return None
        
        scored = [(p, self._proxies[p].compute_score()) for p in candidates if p in self._proxies]
        if not scored:
            return None
            
        total = sum(s for _, s in scored)
        if total <= 0:
            return random.choice(list(candidates))

        r = random.random() * total
        cumulative = 0.0
        for proxy, score in scored:
            cumulative += score
            if cumulative >= r:
                return proxy
        return scored[-1][0]

    async def fetch_json(self, url: str, max_retries: int = 8, **kwargs) -> Tuple[bool, Any]:
        """Convenience wrapper for JSON endpoints."""
        return await self.fetch(url, max_retries=max_retries, **kwargs)

    async def fetch_binance_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        max_retries: int = 10,
    ) -> Tuple[bool, Any]:
        """
        Optimized method for Binance klines with extra retries.
        
        Returns:
            (True, klines_list) on success
            (False, error_message) on failure
        """
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        success, data = await self.fetch(url, params=params, max_retries=max_retries)

        if success and isinstance(data, list):
            return True, data
        elif success:
            return False, f"Unexpected response type: {type(data)}"
        else:
            return False, data

    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        interval: str,
        limit: int = 500,
        concurrency: int = 10,
    ) -> Dict[str, Tuple[bool, Any]]:
        """
        Fetch klines for multiple symbols with controlled concurrency.
        Returns dict: {symbol: (success, data_or_error)}
        """
        sem = asyncio.Semaphore(concurrency)
        results = {}

        async def fetch_one(symbol: str):
            async with sem:
                return symbol, await self.fetch_binance_klines(symbol, interval, limit)

        tasks = [fetch_one(s) for s in symbols]
        for coro in asyncio.as_completed(tasks):
            symbol, result = await coro
            results[symbol] = result

        return results

    async def force_refresh(self):
        """Force an immediate pool refresh."""
        logging.info("🔄 Force refreshing proxy pool...")
        await self._populate_pool()

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed pool statistics."""
        states = {"active": 0, "cooling": 0, "banned": 0}
        total_success_rate = 0.0
        total_latency = 0.0
        count_for_avg = 0

        for proxy, stats in self._proxies.items():
            states[stats.state.value] += 1
            if stats.state == ProxyState.ACTIVE and stats.total_uses > 0:
                total_success_rate += stats.success_rate
                total_latency += stats.avg_latency_ms
                count_for_avg += 1

        request_success_rate = 0
        if self._total_requests > 0:
            request_success_rate = self._successful_requests / self._total_requests

        return {
            "pool_size": self.pool_size,
            "total_proxies": len(self._proxies),
            "active": states["active"],
            "cooling": states["cooling"],
            "banned": states["banned"],
            "avg_proxy_success_rate": (total_success_rate / count_for_avg) if count_for_avg else 0,
            "avg_latency_ms": (total_latency / count_for_avg) if count_for_avg else 0,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "request_success_rate": request_success_rate,
            "direct_fallbacks": self._direct_fallbacks,
            "is_healthy": self.is_healthy,
        }

    def get_top_proxies(self, n: int = 10) -> List[Tuple[str, float, float]]:
        """Get top N proxies by score. Returns [(proxy, score, success_rate), ...]"""
        active = self.active_proxies
        scored = [(p, self._proxies[p].compute_score(), self._proxies[p].success_rate) for p in active]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]'''

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
# EXCHANGE CLIENTS
# ==========================================

class ExchangeClient:
    def __init__(self, session: aiohttp.ClientSession, proxy_pool: RobustProxyPool):
        self.session = session
        self.proxies = proxy_pool
        limit = CONFIG.MAX_CONCURRENCY if proxy_pool.active_proxies else 5
        self.sem = asyncio.Semaphore(limit)

    async def _request(self, url: str, params: dict = None) -> Any:
        last_error = "Unknown"
        for attempt in range(CONFIG.MAX_RETRIES):
            proxy = await self.proxies.get_proxy()
            if not proxy:
                await asyncio.sleep(1)
                continue
            try:
                async with self.sem:
                    start_t = time.time()  # Track latency
                    async with self.session.get(url, params=params, proxy=proxy, timeout=CONFIG.REQUEST_TIMEOUT) as resp:
                        if resp.status == 200:
                            # FIX: Report success before returning!
                            latency = (time.time() - start_t) * 1000
                            await self.proxies.report_success(proxy, latency)
                            return await resp.json()
                        
                        # Handle non-200
                        elif resp.status == 429:
                            await self.proxies.report_failure(proxy) # Rate limit is a failure of sorts
                            logging.warning(f"⚠️ 429 Rate Limit ({proxy}). Sleeping 5s.")
                            await asyncio.sleep(5)
                            last_error = "429"
                        elif resp.status >= 500: 
                            await self.proxies.report_failure(proxy)
                            last_error = f"Server {resp.status}"
                        else: 
                            # 404 etc might not be proxy fault, but usually safest to report
                            last_error = f"HTTP {resp.status}"
            
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                await self.proxies.report_failure(proxy)
                last_error = str(e)
            except Exception as e: 
                # Don't report failure for generic python errors (bug in code vs bug in proxy)
                last_error = f"Unexpected: {str(e)}"
                
            await asyncio.sleep(0.5 + random.random() * 0.5)
            
        logging.warning(f"❌ Failed {url} after {CONFIG.MAX_RETRIES} tries. Last err: {last_error}")
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
        data = await self._request(base, {'symbol': symbol, 'interval': '1h', 'limit': 72})
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
        data = await self._request(url, {'category': cat, 'symbol': symbol, 'interval': '60', 'limit': 72})
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
# CORE LOGIC
# ==========================================

def get_cache_key(tf: str) -> int:
    """
    Returns stable integer timestamp for the current TF candle open.
    
    For weekly: Aligns to Monday 00:00 UTC.
    The Unix epoch (Jan 1, 1970) was a Thursday, so we offset by 3 days (259200s)
    to align the 604800-second weekly boundary with Monday.
    
    For daily: 86400 divides evenly from epoch and aligns with UTC midnight.
    
    For 4h: 14400 divides evenly, candles open at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC.
    """
    mins = TIMEFRAME_MINUTES[tf]
    period_seconds = mins * 60
    now = int(time.time())
    
    if tf == '1w':
        # Offset: Unix epoch was Thursday. Monday is 3 days (259200s) before the next Thursday boundary.
        # To align: subtract the Thursday offset, do modular arithmetic, then add it back.
        thursday_offset = 259200  # 3 days in seconds (Thu -> Mon)
        adjusted = now - thursday_offset
        candle_start = adjusted - (adjusted % period_seconds) + thursday_offset
        return candle_start
    else:
        # Daily and sub-daily align naturally with epoch
        return now - (now % period_seconds)


def is_weekly_scan_ready() -> bool:
    """
    Check if enough time has passed since the weekly candle open
    to allow scanning. Prevents scanning stale data right at the boundary.
    """
    now = int(time.time())
    weekly_candle_open = get_cache_key('1w')
    elapsed_since_open = now - weekly_candle_open
    return elapsed_since_open >= CONFIG.WEEKLY_SCAN_DELAY


def calculate_volatility(closes: List[float]) -> float:
    if len(closes) < 24: return 0.0
    returns = []
    for i in range(1, len(closes)):
        if closes[i-1] != 0: returns.append((closes[i] - closes[i-1]) / closes[i-1] * 100)
    return np.std(returns) if returns else 0.0


def classify_middle_band_direction(
    rsi_array: np.ndarray,
    mid_array: np.ndarray,
    idx: int,
    lookback: int = 5
) -> str:
    """
    Determine if a middle band touch is BULLISH or BEARISH using
    a 3-signal voting system:
    
    Signal 1 - Trajectory: Was RSI predominantly above or below the middle band
               over the lookback period?
               - Predominantly ABOVE → approaching from above → BULLISH (price found support)
               - Predominantly BELOW → approaching from below → BEARISH (price found resistance)
    
    Signal 2 - Crossing Direction: Did RSI cross the middle band, and in which direction?
               - Crossed from above to below → BEARISH
               - Crossed from below to above → BULLISH
               - No cross → use position relative to band
    
    Signal 3 - RSI Momentum: Is RSI rising or falling over the lookback?
               - Rising (positive delta) → BULLISH
               - Falling (negative delta) → BEARISH
    
    Convention:
        - "bullish" = RSI came from ABOVE and touched/crossed down to middle band
          (interpreted as: price was strong, pulled back to support → bounce expected)
        - "bearish" = RSI came from BELOW and touched/crossed up to middle band
          (interpreted as: price was weak, pushed up to resistance → rejection expected)
    
    Returns: "bullish" or "bearish"
    """
    # Determine safe lookback range
    safe_lookback = min(lookback, idx)
    if safe_lookback < 2:
        # Not enough data, fall back to simple position check
        if rsi_array[idx] > mid_array[idx]:
            return "bullish"  # Currently above mid = came from above
        else:
            return "bearish"  # Currently below mid = came from below
    
    start_idx = idx - safe_lookback
    
    # ── Signal 1: Trajectory Analysis ──
    # Count how many of the lookback candles had RSI above vs below the middle band
    above_count = 0
    below_count = 0
    for i in range(start_idx, idx + 1):
        if np.isnan(rsi_array[i]) or np.isnan(mid_array[i]):
            continue
        if rsi_array[i] > mid_array[i]:
            above_count += 1
        else:
            below_count += 1
    
    # Predominantly above = came from above = bullish (support touch)
    # Predominantly below = came from below = bearish (resistance touch)
    if above_count > below_count:
        trajectory_vote = "bullish"
    elif below_count > above_count:
        trajectory_vote = "bearish"
    else:
        trajectory_vote = "neutral"
    
    # ── Signal 2: Crossing Direction ──
    # Look for the most recent sign change in (RSI - mid)
    crossing_vote = "neutral"
    for i in range(idx, start_idx, -1):
        if np.isnan(rsi_array[i]) or np.isnan(rsi_array[i - 1]):
            continue
        if np.isnan(mid_array[i]) or np.isnan(mid_array[i - 1]):
            continue
        
        prev_diff = rsi_array[i - 1] - mid_array[i - 1]
        curr_diff = rsi_array[i] - mid_array[i]
        
        if prev_diff > 0 and curr_diff <= 0:
            # Crossed from above to below → came from above → bullish
            crossing_vote = "bullish"
            break
        elif prev_diff < 0 and curr_diff >= 0:
            # Crossed from below to above → came from below → bearish
            crossing_vote = "bearish"
            break
    
    # If no cross found, use current position as a weaker signal
    if crossing_vote == "neutral":
        if rsi_array[idx] > mid_array[idx]:
            crossing_vote = "bullish"
        elif rsi_array[idx] < mid_array[idx]:
            crossing_vote = "bearish"
    
    # ── Signal 3: RSI Momentum ──
    # Compare current RSI to RSI from `safe_lookback` candles ago
    momentum_start_idx = max(start_idx, 0)
    # Find the first non-NaN value in the lookback range for comparison
    momentum_start_rsi = None
    for i in range(momentum_start_idx, idx):
        if not np.isnan(rsi_array[i]):
            momentum_start_rsi = rsi_array[i]
            break
    
    if momentum_start_rsi is not None and not np.isnan(rsi_array[idx]):
        rsi_delta = rsi_array[idx] - momentum_start_rsi
        if rsi_delta > 0:
            # RSI is rising → momentum pushing up from below → bearish (approaching resistance)
            momentum_vote = "bearish"
        elif rsi_delta < 0:
            # RSI is falling → momentum pushing down from above → bullish (approaching support)
            momentum_vote = "bullish"
        else:
            momentum_vote = "neutral"
    else:
        momentum_vote = "neutral"
    
    # ── Voting ──
    votes = {"bullish": 0, "bearish": 0}
    for vote in [trajectory_vote, crossing_vote, momentum_vote]:
        if vote in votes:
            votes[vote] += 1
    
    if votes["bullish"] >= 2:
        return "bullish"
    elif votes["bearish"] >= 2:
        return "bearish"
    else:
        # Tiebreaker: use trajectory as the strongest single signal
        if trajectory_vote != "neutral":
            return trajectory_vote
        elif crossing_vote != "neutral":
            return crossing_vote
        else:
            # Ultimate fallback: position relative to middle band
            if rsi_array[idx] >= mid_array[idx]:
                return "bullish"
            else:
                return "bearish"


def check_bb_rsi(closes: List[float], tf: str) -> Tuple[Optional[str], Optional[str], float]:
    """
    Check if the RSI Bollinger Band touch condition is met.
    
    Returns:
        (touch_type, direction, rsi_value)
        touch_type: "UPPER", "MIDDLE", "LOWER", or None
        direction: "bullish", "bearish", or "" (empty for upper/lower)
        rsi_value: current RSI value
    """
    if len(closes) < CONFIG.MIN_CANDLES:
        return None, None, 0.0
    
    np_c = np.array(closes, dtype=float)
    rsi = talib.RSI(np_c, timeperiod=CONFIG.RSI_PERIOD)
    upper, mid, lower = talib.BBANDS(
        rsi,
        timeperiod=CONFIG.BB_LENGTH,
        nbdevup=CONFIG.BB_STDDEV,
        nbdevdn=CONFIG.BB_STDDEV,
        matype=0
    )
    
    # Use second-to-last candle (last completed candle)
    idx = -2
    
    if np.isnan(rsi[idx]) or np.isnan(upper[idx]):
        return None, None, 0.0
    
    curr_rsi = rsi[idx]
    
    # Check upper band touch
    if curr_rsi >= upper[idx] * (1 - CONFIG.UPPER_TOUCH_THRESHOLD):
        return "UPPER", "", curr_rsi
    
    # Check lower band touch
    if curr_rsi <= lower[idx] * (1 + CONFIG.LOWER_TOUCH_THRESHOLD):
        return "LOWER", "", curr_rsi
    
    # Check middle band touch (only for configured timeframes)
    if tf in MIDDLE_BAND_TFS:
        mid_val = mid[idx]
        if mid_val > 0 and abs(curr_rsi - mid_val) <= (mid_val * CONFIG.MIDDLE_TOUCH_THRESHOLD):
            # Use the multi-signal classification system
            direction = classify_middle_band_direction(
                rsi_array=rsi,
                mid_array=mid,
                idx=len(rsi) + idx,  # Convert negative index to positive
                lookback=CONFIG.MIDDLE_BAND_LOOKBACK
            )
            return "MIDDLE", direction, curr_rsi
    
    return None, None, 0.0

# ==========================================
# MAIN BOT
# ==========================================

class RsiBot:
    def __init__(self):
        self.cache = CacheManager()
        self.proxies = RobustProxyPool(
            validation_concurrency=100,
            allow_direct_fallback=False,
            max_pool_size=20,
            request_timeout=CONFIG.REQUEST_TIMEOUT,
            validation_timeout=4.0
        )
        
    async def send_report(self, session: aiohttp.ClientSession, hits: List[TouchHit]):
        """
        Send formatted Telegram report using HTML parse mode with <pre> tags
        for guaranteed monospace alignment across all Telegram clients.
        
        Layout per timeframe:
        - UPPER BAND section
        - MIDDLE BAND ▲ BULLISH section
        - MIDDLE BAND ▼ BEARISH section  
        - LOWER BAND section
        
        Smart batching: packs as many sections as possible into each message
        before splitting to the next one. Never sends a single section alone
        unless it genuinely fills a message by itself.
        """
        if not hits:
            return

        # ── Group hits by timeframe → section ──
        grouped: Dict[str, Dict[str, List[TouchHit]]] = {}
        for h in hits:
            tf_group = grouped.setdefault(h.timeframe, {})
            
            if h.touch_type == "MIDDLE":
                if h.direction == "bullish":
                    section_key = "MIDDLE_BULLISH"
                else:
                    section_key = "MIDDLE_BEARISH"
            else:
                section_key = h.touch_type
            
            tf_group.setdefault(section_key, []).append(h)

        tf_order = ["1w", "1d", "4h", "2h", "1h", "30m", "15m", "5m", "3m"]
        ts_footer = datetime.now(timezone.utc).strftime('%d %b %H:%M UTC')

        def clean_name(s: str) -> str:
            """Clean symbol name: remove USDT suffix and common prefixes, cap at 6 chars."""
            s = s.replace("USDT", "")
            s = re.sub(r"^(1000000|100000|10000|1000|100|10|1M)(?=[A-Z])", "", s)
            return s[:6]

        def format_cell(item: TouchHit) -> str:
            """
            Format a single symbol cell with fixed width for perfect alignment.
            Layout: 'SYM   67.3🔥' = 6 + 5 + 2 = 13 chars per cell
            """
            sym = clean_name(item.symbol)
            hot = "🔥" if item.hot else "  "
            return f"{sym:<6}{item.rsi:>5.1f}{hot}"

        def build_rows(items: List[TouchHit], cols: int = 3) -> List[str]:
            """Build formatted rows with `cols` symbols per line, separated by │"""
            rows = []
            for i in range(0, len(items), cols):
                chunk = items[i:i + cols]
                cells = []
                for item in chunk:
                    cells.append(format_cell(item))
                while len(cells) < cols:
                    cells.append(" " * 13)
                row = " │ ".join(cells)
                rows.append(f"│ {row} │")
            return rows

        # Section definitions: key, emoji/label, sort_descending
        section_defs = [
            ("UPPER",          "🔼 UPPER BAND",           True),
            ("MIDDLE_BULLISH", "💠 MIDDLE ▲ BULLISH",      True),
            ("MIDDLE_BEARISH", "💠 MIDDLE ▼ BEARISH",      False),
            ("LOWER",          "🔽 LOWER BAND",            False),
        ]

        # Box width: "│ " + cell(13) + " │ " + cell(13) + " │ " + cell(13) + " │" = 49
        box_width = 49

        def build_section_block(section_key: str, section_label: str, sort_descending: bool, items: List[TouchHit]) -> str:
            """Build a complete section block string (header + box + rows)."""
            items.sort(key=lambda x: x.rsi, reverse=sort_descending)
            count = len(items)
            header_line = f"{section_label} ({count})"
            top_border = f"┌{'─' * (box_width - 2)}┐"
            bottom_border = f"└{'─' * (box_width - 2)}┘"
            content_rows = build_rows(items, cols=3)
            
            lines = [
                "",
                header_line,
                top_border,
            ]
            lines.extend(content_rows)
            lines.append(bottom_border)
            
            return "\n".join(lines)

        # ── Process each timeframe ──
        for tf in tf_order:
            if tf not in grouped:
                continue
            
            tf_sections = grouped[tf]
            total_hits = sum(len(v) for v in tf_sections.values())
            if total_hits == 0:
                continue

            # Build all section blocks for this timeframe
            section_blocks: List[str] = []
            for section_key, section_label, sort_descending in section_defs:
                items = tf_sections.get(section_key, [])
                if not items:
                    continue
                block = build_section_block(section_key, section_label, sort_descending, items)
                section_blocks.append(block)

            if not section_blocks:
                continue

            # ── Smart Batching ──
            # Pack as many sections as possible into each message.
            # Only split to a new message when adding the next section would exceed the limit.
            
            tf_header = f"⏱ <b>{tf} Timeframe</b> ({total_hits})\n"
            
            # Overhead per message: tf_header + <pre></pre> tags + footer + padding
            # <pre>\n</pre> = 11 chars, footer ~25 chars, safety margin
            overhead = len(tf_header) + len(ts_footer) + 30  # ~30 for tags + newlines + safety
            max_content_chars = 4000 - overhead
            
            # Batch sections greedily
            current_batch: List[str] = []
            current_batch_chars: int = 0
            
            for block in section_blocks:
                block_len = len(block)
                
                # Check if adding this block to current batch would exceed limit
                if current_batch and (current_batch_chars + block_len + 1) > max_content_chars:
                    # Send current batch first
                    batch_content = "\n".join(current_batch)
                    message = tf_header + f"<pre>{batch_content}</pre>"
                    await self._safe_send(session, message, ts_footer)
                    
                    # Start new batch with this block
                    current_batch = [block]
                    current_batch_chars = block_len
                
                elif block_len > max_content_chars:
                    # This single section is too large to fit in one message by itself.
                    # Send whatever is in the current batch first.
                    if current_batch:
                        batch_content = "\n".join(current_batch)
                        message = tf_header + f"<pre>{batch_content}</pre>"
                        await self._safe_send(session, message, ts_footer)
                        current_batch = []
                        current_batch_chars = 0
                    
                    # Split this oversized section by rows.
                    # Re-parse the block into its component lines.
                    block_lines = block.split("\n")
                    
                    # Separate the header part (first 3 lines: empty, label, top border)
                    # and the footer part (last line: bottom border)
                    # from the content rows in between.
                    section_header_lines = []
                    section_footer_line = ""
                    content_lines = []
                    
                    for i, line in enumerate(block_lines):
                        if line.startswith("└"):
                            section_footer_line = line
                        elif line.startswith("┌") or line.startswith("🔼") or line.startswith("💠") or line.startswith("🔽") or line == "":
                            section_header_lines.append(line)
                        elif line.startswith("│"):
                            content_lines.append(line)
                        else:
                            # Catch any other header-like lines (section label without emoji match)
                            if not content_lines:
                                section_header_lines.append(line)
                            else:
                                content_lines.append(line)
                    
                    # Now batch the content rows with the section header repeated
                    section_header_text = "\n".join(section_header_lines)
                    section_header_len = len(section_header_text) + len(section_footer_line) + 2
                    available_for_rows = max_content_chars - section_header_len
                    
                    row_batch: List[str] = []
                    row_batch_chars: int = 0
                    
                    for row_line in content_lines:
                        row_len = len(row_line) + 1  # +1 for newline
                        
                        if row_batch and (row_batch_chars + row_len) > available_for_rows:
                            # Send this chunk with header and footer
                            chunk_lines = section_header_lines + row_batch + [section_footer_line]
                            chunk_content = "\n".join(chunk_lines)
                            message = tf_header + f"<pre>{chunk_content}</pre>"
                            await self._safe_send(session, message, ts_footer)
                            row_batch = []
                            row_batch_chars = 0
                        
                        row_batch.append(row_line)
                        row_batch_chars += row_len
                    
                    # Send remaining rows
                    if row_batch:
                        chunk_lines = section_header_lines + row_batch + [section_footer_line]
                        chunk_content = "\n".join(chunk_lines)
                        message = tf_header + f"<pre>{chunk_content}</pre>"
                        await self._safe_send(session, message, ts_footer)
                
                else:
                    # Block fits — add to current batch
                    current_batch.append(block)
                    current_batch_chars += block_len + 1  # +1 for the joining newline
            
            # ── Send remaining batch for this timeframe ──
            if current_batch:
                batch_content = "\n".join(current_batch)
                message = tf_header + f"<pre>{batch_content}</pre>"
                await self._safe_send(session, message, ts_footer)

    async def _safe_send(self, session: aiohttp.ClientSession, text: str, footer: str):
        """
        Send a single Telegram message with retry logic and rate limit handling.
        Uses HTML parse mode for guaranteed monospace rendering.
        """
        full_text = text + f"\n\n{footer}"
        
        for attempt in range(3):
            try:
                async with session.post(
                    f"https://api.telegram.org/bot{CONFIG.TELEGRAM_TOKEN}/sendMessage",
                    json={
                        "chat_id": CONFIG.CHAT_ID,
                        "text": full_text,
                        "parse_mode": "HTML"
                    }
                ) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 5))
                        logging.warning(f"⚠️ Telegram rate limit. Waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    elif resp.status != 200:
                        resp_text = await resp.text()
                        logging.error(f"Telegram send failed (HTTP {resp.status}): {resp_text}")
                        break
                    
                    # Success — add safety gap between messages
                    await asyncio.sleep(0.5)
                    return
            except Exception as e:
                logging.error(f"Telegram send exception: {e}")
                await asyncio.sleep(1)
        
        logging.error(f"Failed to send Telegram message after 3 attempts")
                
    async def fetch_symbols_hybrid(
        self,
        binance: BinanceClient,
        bybit: BybitClient
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Fetch symbols from all exchanges with cache fallback.
        Returns: (binance_perp, binance_spot, bybit_perp, bybit_spot)
        """
        cached = await self.cache.get_cached_symbols()
        cached_data = cached.get('data') if cached else None
        
        async def try_fetch(fetch_func, cache_key: str, name: str):
            result = await fetch_func()
            
            if result and len(result) > 0:
                logging.info(f"✅ {name}: {len(result)} symbols")
                return result
            
            if cached_data and cache_key in cached_data:
                logging.warning(f"⚠️ {name}: Live fetch failed. Using CACHE ({len(cached_data[cache_key])} syms)")
                return cached_data[cache_key]
            
            logging.error(f"❌ {name}: Failed and NO CACHE available.")
            return []
        
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
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s'
        )
        
        # ── Startup Config Validation ──
        if not CONFIG.TELEGRAM_TOKEN:
            logging.error("❌ TELEGRAM_BOT_TOKEN is not set! Exiting.")
            return
        if not CONFIG.CHAT_ID:
            logging.error("❌ TELEGRAM_CHAT_ID is not set! Exiting.")
            return
        
        await self.cache.init()
        
        async with aiohttp.ClientSession() as session:
            try:
                await self.proxies.initialize(session, additional_sources=[CONFIG.PROXY_URL])
                binance = BinanceClient(session, self.proxies)
                bybit = BybitClient(session, self.proxies)
                
                # ── Fetch Symbols ──
                bp, bs, yp, ys = await self.fetch_symbols_hybrid(binance, bybit)
                
                # ── Deduplicate with Priority ──
                # Priority: Binance Perp > Binance Spot > Bybit Perp > Bybit Spot
                # We process in priority order. Once a normalized symbol is seen, skip it.
                seen_normalized: Set[str] = set()
                all_pairs: List[Tuple[Any, str, str, str]] = []
                
                def add_with_dedup(
                    client: ExchangeClient,
                    symbols: List[str],
                    market: str,
                    exchange: str
                ):
                    """Add symbols to all_pairs, skipping already-seen normalized names."""
                    for s in symbols:
                        if s in CONFIG.IGNORED_SYMBOLS:
                            continue
                        if not s.endswith("USDT"):
                            continue
                        norm = s.upper().replace("USDT", "")
                        if norm not in seen_normalized:
                            seen_normalized.add(norm)
                            all_pairs.append((client, s, market, exchange))
                
                # Add in strict priority order
                add_with_dedup(binance, bp, 'perp', 'Binance')
                add_with_dedup(binance, bs, 'spot', 'Binance')
                add_with_dedup(bybit, yp, 'perp', 'Bybit')
                add_with_dedup(bybit, ys, 'spot', 'Bybit')
                
                total_sym_count = len(all_pairs)
                logging.info(f"Total unique symbols after dedup: {total_sym_count}")
                
                # ── Volatility Calculation ──
                logging.info("Calculating Volatility...")
                vol_scores: Dict[str, float] = {}
                vol_sem = asyncio.Semaphore(50)

                async def check_vol(client: ExchangeClient, sym: str, mkt: str):
                    async with vol_sem:
                        try:
                            c = await client.fetch_closes_volatility(sym, mkt)
                            if c:
                                v = calculate_volatility(c)
                                if v > 0:
                                    vol_scores[sym] = v
                        except Exception as e:
                            logging.debug(f"Volatility check failed for {sym}: {e}")
                
                vol_tasks = [check_vol(client, s, mkt) for client, s, mkt, ex in all_pairs]
                await asyncio.gather(*vol_tasks)
                
                hot_coins = set(sorted(vol_scores, key=vol_scores.get, reverse=True)[:200])
                logging.info(f"Vol Calc: {len(vol_scores)}/{len(all_pairs)} success | Hot: {len(hot_coins)}")
                
                # ── Determine Which Timeframes to Scan ──
                sent_state = await self.cache.get_sent_state()
                scan_stats = {
                    tf: ScanStats(tf, "Unknown", total_symbols=total_sym_count)
                    for tf in ACTIVE_TFS
                }
                
                tfs_to_scan_fresh: List[str] = []
                cached_hits_to_use: List[TouchHit] = []
                
                for tf in ACTIVE_TFS:
                    # ── Weekly Timing Gate ──
                    if tf == '1w' and not is_weekly_scan_ready():
                        elapsed = int(time.time()) - get_cache_key('1w')
                        remaining = CONFIG.WEEKLY_SCAN_DELAY - elapsed
                        logging.info(
                            f"⏳ Skipping 1w: Weekly candle just opened. "
                            f"Waiting {remaining}s more for data to settle."
                        )
                        scan_stats[tf].source = "Skipped (Weekly Delay)"
                        continue
                    
                    if tf in CACHED_TFS:
                        candle_key = get_cache_key(tf)
                        
                        # Already sent for this candle? Skip entirely.
                        if sent_state.get(tf) == candle_key:
                            logging.info(f"⏭️ Skipping {tf}: already sent for this candle")
                            scan_stats[tf].source = "Already Sent"
                            continue
                        
                        # Check if we have cached scan results for this candle
                        cached_res = await self.cache.get_scan_results(tf, candle_key)
                        if cached_res is not None:
                            hits = [TouchHit.from_dict(d) for d in cached_res]
                            cached_hits_to_use.extend(hits)
                            scan_stats[tf].source = "Cached"
                            scan_stats[tf].hits_found = len(hits)
                            scan_stats[tf].successful_scans = 0
                            logging.info(f"📦 Using cached results for {tf}: {len(hits)} hits")
                        else:
                            tfs_to_scan_fresh.append(tf)
                            scan_stats[tf].source = "Fresh Scan (New Candle)"
                    else:
                        tfs_to_scan_fresh.append(tf)
                        scan_stats[tf].source = "Fresh Scan (Low TF)"
                
                # ── Execute Fresh Scans ──
                final_hits: List[TouchHit] = []
                
                if tfs_to_scan_fresh:
                    logging.info(f"Scanning fresh TFs: {tfs_to_scan_fresh} across {total_sym_count} symbols...")
                    
                    for tf in tfs_to_scan_fresh:
                        tf_hits: List[TouchHit] = []
                        
                        async def scan_one(
                            client: ExchangeClient,
                            sym: str,
                            mkt: str,
                            ex: str,
                            scan_tf: str
                        ) -> List[TouchHit]:
                            """
                            Scan a single symbol on a single timeframe.
                            Returns list of TouchHit (0 or 1 items).
                            Catches all exceptions internally to prevent task failures.
                            """
                            try:
                                closes = await client.fetch_closes(sym, scan_tf, mkt)
                                if not closes:
                                    return []
                                
                                t_type, direction, rsi_val = check_bb_rsi(closes, scan_tf)
                                if t_type:
                                    return [TouchHit(
                                        symbol=sym,
                                        exchange=ex,
                                        market=mkt,
                                        timeframe=scan_tf,
                                        rsi=rsi_val,
                                        touch_type=t_type,
                                        direction=direction if direction else "",
                                        hot=sym in hot_coins
                                    )]
                                return []
                            except Exception as e:
                                logging.debug(f"Scan failed for {sym} on {scan_tf}: {e}")
                                return []
                        
                        # Build tasks for this timeframe
                        tf_tasks = [
                            scan_one(client, sym, mkt, ex, tf)
                            for client, sym, mkt, ex in all_pairs
                        ]
                        
                        results = await asyncio.gather(*tf_tasks, return_exceptions=True)
                        
                        successful_count = 0
                        for result in results:
                            if isinstance(result, list):
                                tf_hits.extend(result)
                                successful_count += 1
                            elif isinstance(result, Exception):
                                # This should rarely happen since scan_one catches internally
                                logging.error(f"⚠️ Unexpected scan task exception: {result}")
                        
                        scan_stats[tf].successful_scans = successful_count
                        scan_stats[tf].hits_found = len(tf_hits)
                        
                        final_hits.extend(tf_hits)
                        
                        # Cache the results for cacheable timeframes
                        if tf in CACHED_TFS:
                            candle_key = get_cache_key(tf)
                            await self.cache.save_scan_results(
                                tf,
                                candle_key,
                                [h.to_dict() for h in tf_hits]
                            )
                            logging.info(f"💾 Cached {len(tf_hits)} hits for {tf} (key: {candle_key})")
                
                # Merge cached hits with fresh hits
                final_hits.extend(cached_hits_to_use)
                
                # ── Print Scan Summary ──
                logging.info("=" * 65)
                logging.info(f"{'TF':<5} | {'Source':<28} | {'Scanned':<18} | {'Hits'}")
                logging.info("-" * 65)
                
                tf_display_order = [
                    t for t in ["3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w"]
                    if t in ACTIVE_TFS
                ]
                for tf in tf_display_order:
                    st = scan_stats[tf]
                    if st.source in ("Cached", "Already Sent", "Skipped (Weekly Delay)"):
                        scanned_str = "—"
                    else:
                        scanned_str = f"{st.successful_scans}/{st.total_symbols}"
                    logging.info(
                        f"[{tf:<3}] {st.source:<28} | {scanned_str:<18} | {st.hits_found}"
                    )
                logging.info("=" * 65)
                logging.info(f"Total hits to send: {len(final_hits)}")
                
                # ── Filter Hits & Update Sent State ──
                hits_to_send: List[TouchHit] = []
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
                
                # ── Send Report ──
                if hits_to_send:
                    logging.info(f"📤 Sending {len(hits_to_send)} hits to Telegram...")
                    await self.send_report(session, hits_to_send)
                    logging.info("✅ Report sent successfully")
                else:
                    logging.info("📭 No new hits to send")
                
                # ── Persist Sent State ──
                await self.cache.save_sent_state(new_state)
                
            finally:
                # ── Graceful Cleanup (4a) ──
                # Always shut down the proxy pool background tasks,
                # even if an exception occurred during scanning
                await self.proxies.shutdown()
                logging.info("🧹 Proxy pool background tasks cleaned up")
        
        # Close Redis connection
        await self.cache.close()
        logging.info("🏁 Scan cycle complete")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    bot = RsiBot()
    asyncio.run(bot.run())
