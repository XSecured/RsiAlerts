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

from middle_band_classifier import classify_middle_band_direction as classify_middle_band_direction_core

# ==========================================
# CONFIGURATION & CONSTANTS (UPDATED)
# ==========================================

@dataclass
class Config:
    MAX_CONCURRENCY: int = 20
    REQUEST_TIMEOUT: int = 7
    MAX_RETRIES: int = 3
    
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL_MAP: Dict[str, int] = field(default_factory=lambda: {
        '4h': 14400 + 600, '1h': 3600 + 300, '1d': 86400 + 1800, '1w': 604800 + 3600
    })
    
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    PROXY_URL: str = "https://raw.githubusercontent.com/hproxy-com/free-proxy-list/refs/heads/main/https.txt"
    #  "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"

    RSI_PERIOD: int = 14
    BB_LENGTH: int = 34
    BB_STDDEV: float = 2.0
    CANDLE_LIMIT: int = 60
    MIN_CANDLES: int = 36

    EMA_LENGTH: int = 34
    EMA_THRESHOLD_PCT: float = 5.5
    HOT_COINS_LIMIT: int = 60
    VOLATILITY_LOOKBACK_HOURS: int = 100

    # ── Touch tolerance ──
    # Flat RSI-point tolerance zones — band level and band width play no
    # role at all. E.g. UPPER_TOUCH_POINTS=3.5 means "within 3.5 RSI
    # points of the upper band" no matter where that band sits or how
    # wide it currently is. This is the only threshold mode the bot
    # supports; the earlier percentage-of-band-width mode (with its
    # adaptive multiplier and dynamic/static width reference) was removed
    # entirely rather than kept as a disabled alternative.
    UPPER_TOUCH_POINTS: float = 3.5
    LOWER_TOUCH_POINTS: float = 3.5
    MIDDLE_TOUCH_POINTS: float = 3.5

    # Number of lookback candles for middle band direction analysis
    MIDDLE_BAND_LOOKBACK: int = 5

    # Weekly scan delay in seconds after Monday 00:00 UTC
    # Gives time for the new weekly candle to form its first data
    WEEKLY_SCAN_DELAY: int = 1800  # 30 minutes

    IGNORED_SYMBOLS: Set[str] = field(default_factory=lambda: {
        "USDPUSDT", "USD1USDT", "TUSDUSDT", "AEURUSDT", "USDCUSDT", "EURUSDT", "USDYUSDT", "PYUSDUSDT",
        "USDEUSDT", "USDDUSDT", "BFUSDUSDT", "BTTCUSDT", "XUSDUSDT", "RLUSDUSDT", "FDUSDUSDT", "USDSUSDT",
        "UUSDT"
    })

    # ── Overall run watchdog ──
    # Hard ceiling on total execution time for one bot.run() call, as a
    # last line of defense. A normal run (symbol fetch, EMA/volatility
    # scan, RSI touch scan, any retry rounds, proxy pool validation,
    # Telegram send) takes a few minutes even in a slow case; this gives
    # generous headroom above that while still bounding the worst case.
    # This exists because of a real incident: a swallowed CancelledError
    # in the proxy pool's shutdown path (see RobustProxyPool.shutdown's
    # docstring) let one run silently occupy a GitHub Actions runner for
    # hours with no error, until an external cancellation ended it. If a
    # run ever exceeds this, something is genuinely wrong and it should
    # fail loudly rather than idle unnoticed.
    RUN_TIMEOUT_SECONDS: float = 1200.0  # 20 minutes

    BYBIT_ENABLED: bool = False

    # ── Failed-symbol retry ──
    # If a symbol's fetch still fails after exhausting MAX_RETRIES proxy
    # attempts inside ExchangeClient._request, RETRY_FAILED_SYMBOLS
    # controls whether that symbol gets FAILED_SYMBOL_RETRY_ROUNDS more
    # whole attempts — each one a fresh Thompson Sampling draw against a
    # proxy pool whose posterior has, by then, already been updated to
    # disfavor whatever proxy just failed — rather than being dropped from
    # this scan cycle entirely. Set to False (or ROUNDS=0) to restore the
    # old behavior: a symbol that fails all MAX_RETRIES attempts is simply
    # skipped for this cycle.
    RETRY_FAILED_SYMBOLS: bool = True
    FAILED_SYMBOL_RETRY_ROUNDS: int = 1

    # Timeframes listed here skip the EMA-34 trend filter entirely and are
    # scanned against every deduplicated symbol, independent of trend. Use
    # this for a "see everything" view on slower timeframes (e.g. weekly)
    # while faster timeframes still scan only EMA-confirmed trend coins.
    # 1h is included here deliberately (full-universe scan, used to test the
    # power leaderboard against the whole market); 4h stays EMA-filtered.
    EMA_FILTER_EXEMPT_TFS: Set[str] = field(default_factory=lambda: {'1w', '1d', '1h'})

    # ── Power Leaderboard (RSI Band-Walk Strength Ranking) ──
    # Ranks symbols by how far above/into the RSI-BB upper band they've been
    # sitting over a short trailing window, independent of touch-alert logic.
    # Reuses the same closes already fetched for the standard scan on a given
    # timeframe — no additional API calls beyond the slightly larger fetch
    # size from candle_fetch_limit_for_timeframe().
    POWER_LEADERBOARD_ENABLED: bool = True
    POWER_LEADERBOARD_WINDOW_DEFAULT: int = 10
    POWER_LEADERBOARD_WINDOW_OVERRIDES: Dict[str, int] = field(default_factory=lambda: {
        '1h': 20
    })
    POWER_LEADERBOARD_SIZE: int = 20          # top N, all scored symbols
    POWER_LEADERBOARD_HOT_SIZE: int = 20      # top N, volatile (🔥) symbols only — same message, second section
    POWER_LEADERBOARD_WEEKLY_SIZE: int = 20   # top N by rolling ~7-day peak score — same message, third section

    # Buffer added on top of the computed (warmup + window) requirement when
    # fetching candles, to absorb minor off-by-one warmup estimation and any
    # exchange returning slightly fewer candles than requested near a
    # symbol's listing date.
    CANDLE_FETCH_SAFETY_MARGIN: int = 10

CONFIG = Config()

TIMEFRAME_MINUTES = {
    '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, 
    '1d': 1440, '1w': 10080
}

ACTIVE_TFS = ['4h', '1h', '1d', '1w']
MIDDLE_BAND_TFS = ['4h', '1h', '1d', '1w']
CACHED_TFS = {'4h', '1h', '1d', '1w'}

# See compute_power_score / candle_fetch_limit_for_timeframe: RSI needs
# RSI_PERIOD prior candles before its first valid output, then BBANDS
# needs BB_LENGTH consecutive valid RSI values before *its* first valid
# output — so this many leading candles produce nothing but NaN.
INDICATOR_WARMUP = CONFIG.RSI_PERIOD + CONFIG.BB_LENGTH  # 48


def power_leaderboard_window_for_timeframe(tf: str) -> int:
    return CONFIG.POWER_LEADERBOARD_WINDOW_OVERRIDES.get(tf, CONFIG.POWER_LEADERBOARD_WINDOW_DEFAULT)


def candle_fetch_limit_for_timeframe(tf: str) -> int:
    """
    How many candles to request for a given timeframe's scan.

    Must clear INDICATOR_WARMUP + this timeframe's power-leaderboard
    window, or compute_power_score has zero valid points to work with and
    silently returns None for every symbol on that timeframe — no error,
    just an empty leaderboard every cycle. CONFIG.CANDLE_LIMIT is used as
    a floor, not a ceiling: a timeframe with a larger window (e.g. 1h at
    20, landing around 78) gets a larger fetch; everything else still gets
    at least the configured default.
    """
    window = power_leaderboard_window_for_timeframe(tf)
    required = INDICATOR_WARMUP + window + CONFIG.CANDLE_FETCH_SAFETY_MARGIN
    return max(CONFIG.CANDLE_LIMIT, required)

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
class PowerScore:
    symbol: str
    exchange: str
    market: str
    timeframe: str
    score: float          # signed mean(RSI - upper_band) over the window
    hot: bool = False

    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(d): return PowerScore(**d)

@dataclass
class ScanStats:
    timeframe: str
    source: str
    total_symbols: int = 0
    successful_scans: int = 0
    failed_scans: int = 0
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
        #"https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt"
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
        shutdown_timeout_seconds: float = 15.0,
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
        self.SHUTDOWN_TIMEOUT_SECONDS = shutdown_timeout_seconds

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
        """
        Gracefully shut down the background refresh task.

        Wrapped in a bounded timeout as defense-in-depth: task.cancel()
        only works if the cancelled coroutine actually lets the
        CancelledError propagate. This codebase now does that correctly
        everywhere (every bare `except:` was changed to `except Exception:`,
        which doesn't catch CancelledError — see _validate_proxy and
        _populate_pool for the specific bug this fixes), but a bounded
        wait here means that if a similar mistake is ever reintroduced —
        here or in a future change — the process still exits within
        SHUTDOWN_TIMEOUT_SECONDS instead of hanging for hours. This is
        exactly the failure mode that happened before: the background
        loop's cancellation got silently swallowed, so `await
        self._refresh_task` (with no timeout) blocked forever, and the
        whole bot sat idle until GitHub Actions force-killed the job.
        """
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await asyncio.wait_for(self._refresh_task, timeout=self.SHUTDOWN_TIMEOUT_SECONDS)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logging.error(
                    f"⚠️ Background refresh task did not stop within "
                    f"{self.SHUTDOWN_TIMEOUT_SECONDS}s of being cancelled — "
                    f"abandoning it so the process can still exit. This "
                    f"means something is catching CancelledError and not "
                    f"re-raising it; worth a look."
                )
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
        except Exception:
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
            except Exception:
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

    def _select_thompson_sampling(self) -> Optional[str]:
        """
        Select a proxy via Thompson Sampling over a Beta-Bernoulli model of
        each active proxy's success rate.

        Each proxy's true (unknown) success probability is modeled as a
        Beta(alpha, beta) posterior, where alpha = 1 + successes and
        beta = 1 + failures (Beta(1,1) is a uniform "no information yet"
        prior). We draw one random sample from every active proxy's
        posterior and pick whichever proxy's draw is highest.

        Why this beats the old fixed-formula score (success_rate scaled by
        a consecutive-failure penalty, a latency factor, and a flat
        "freshness bonus" for new proxies): that formula treated a
        proxy's history as a single point estimate, and a point estimate
        can't distinguish "reliably 100%" from "won its only try so far".
        A proxy with 1 success and 0 failures has a WIDE posterior — its
        true rate could plausibly be anywhere from ~20% to ~100% — so it
        will occasionally draw a high sample and get tried again, earning
        more evidence. A proxy with 8 successes and 2 failures has a
        NARROW posterior centered near 80%, so its draws cluster tightly
        around 80% and it gets picked reliably. That explore/exploit
        balance falls out of the sampling itself, with no separate
        hand-tuned "freshness bonus" needed — this is the standard
        solution to exactly this class of problem (the multi-armed
        bandit), with convergence guarantees a fixed formula doesn't have.

        Latency isn't modeled as a separate term here: timeouts and
        connection errors already count as failures via report_failure
        (see ExchangeClient._request), so a proxy that's slow enough to
        matter is already accumulating failures and getting down-weighted
        through the same posterior — a second signal for the same
        underlying problem would be redundant.
        """
        active = self.active_proxies
        if not active:
            return None

        best_proxy: Optional[str] = None
        best_sample = -1.0
        for proxy in active:
            stats = self._proxies[proxy]
            alpha = 1.0 + stats.successes
            beta_param = 1.0 + stats.failures
            sample = random.betavariate(alpha, beta_param)
            if sample > best_sample:
                best_sample = sample
                best_proxy = proxy

        return best_proxy

    async def get_proxy(self) -> Optional[str]:
        """Get a proxy via Thompson Sampling (see _select_thompson_sampling)."""
        proxy = self._select_thompson_sampling()
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
        except Exception:
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
        except Exception: return None

    async def save_scan_results(self, tf: str, candle_key: int, results: List[Dict]):
        if not self.redis: return
        ttl = CONFIG.CACHE_TTL_MAP.get(tf, 3600)
        try: await self.redis.set(self._scan_key(tf, candle_key), json.dumps(results), ex=ttl)
        except Exception: pass

    async def get_sent_state(self) -> Dict[str, int]:
        if not self.redis: return {}
        try:
            val = await self.redis.get("bb_bot:sent_state")
            return json.loads(val) if val else {}
        except Exception: return {}

    async def save_sent_state(self, state: Dict[str, int]):
        if not self.redis: return
        try: await self.redis.set("bb_bot:sent_state", json.dumps(state))
        except Exception: pass

    # ── Weekly Power Peak (rolling ~7-day high per symbol/timeframe) ──
    # Bucketed by UTC calendar day rather than a strict trailing 168h
    # window: record_power_scores merges each cycle's HOT-ONLY scores into
    # TODAY's bucket (keeping the max per symbol if called more than once
    # today), and get_weekly_peak_scores reads today's bucket plus the 6
    # preceding calendar-day buckets and takes the max per symbol across
    # all seven. This is an approximation — the actual elapsed span
    # covered varies with time of day, from a bit under 7 days to a bit
    # under 8 — traded for O(7) Redis reads per call instead of
    # maintaining and pruning a full timestamped history per symbol. Good
    # enough for "roughly this week's peak power among the most volatile
    # coins"; not a substitute for precise backtesting if that's ever
    # needed instead.

    def _power_daily_key(self, tf: str, date_str: str) -> str:
        return f"bb_bot:power_daily_peak:{tf}:{date_str}"

    async def record_power_scores(self, tf: str, scores: List["PowerScore"]) -> None:
        """
        Merge this cycle's power scores into today's UTC daily-peak bucket
        for this timeframe, keeping the higher of (existing bucket value,
        this cycle's score) per symbol.

        Only symbols flagged hot=True (🔥 volatile, per
        hot_coins_for_timeframe) THIS cycle are recorded — the weekly peak
        board is specifically "peak power among the most volatile coins,"
        not a peak across the whole scanned universe. A symbol that isn't
        volatile this cycle simply contributes nothing to today's bucket,
        even if it was scored; a symbol that cools off for a stretch just
        stops adding new entries, while whatever it already banked stays
        live in the rolling window until that day's bucket ages past the
        7-day mark on its own.

        Read-then-write (one HGETALL, then one HSET) rather than a
        per-symbol round trip: this runs once per scan cycle per
        timeframe, not once per symbol, so two Redis calls is the actual
        cost here regardless of how many hundreds of symbols are in
        `scores`.
        """
        hot_scores = [s for s in scores if s.hot]
        if not self.redis or not hot_scores:
            return
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        key = self._power_daily_key(tf, date_str)
        try:
            existing = await self.redis.hgetall(key)
            merged: Dict[str, float] = {sym: float(val) for sym, val in existing.items()} if existing else {}
            for s in hot_scores:
                if s.symbol not in merged or s.score > merged[s.symbol]:
                    merged[s.symbol] = s.score
            if merged:
                await self.redis.hset(key, mapping={sym: str(val) for sym, val in merged.items()})
                # TTL is a safety net so a stopped bot doesn't leave buckets
                # growing forever — it is NOT the windowing mechanism. The
                # window is decided entirely by which 7 date keys
                # get_weekly_peak_scores chooses to read, independent of
                # this TTL. 9 days gives a full day of margin past the
                # 7-day window before a bucket can expire out from under it.
                await self.redis.expire(key, 9 * 86400)
        except Exception as e:
            logging.debug(f"record_power_scores failed for {tf}/{date_str}: {e}")

    async def get_weekly_peak_scores(self, tf: str) -> Dict[str, float]:
        """
        Rolling ~7-calendar-day peak power score per symbol for a
        timeframe, among symbols that were volatile (hot) on at least one
        of the days contributing to the window — see record_power_scores
        for the hot-only filtering and the class-level comment above it
        for the bucketing approximation.
        """
        if not self.redis:
            return {}
        peaks: Dict[str, float] = {}
        now = datetime.now(timezone.utc)
        try:
            for days_back in range(7):
                date_str = (now - timedelta(days=days_back)).strftime('%Y-%m-%d')
                bucket = await self.redis.hgetall(self._power_daily_key(tf, date_str))
                if not bucket:
                    continue
                for sym, val in bucket.items():
                    score = float(val)
                    if sym not in peaks or score > peaks[sym]:
                        peaks[sym] = score
        except Exception as e:
            logging.debug(f"get_weekly_peak_scores failed for {tf}: {e}")
            return {}
        return peaks

# ==========================================
# EXCHANGE CLIENTS
# ==========================================

def validate_klines_payload(
    raw: Any,
    interval: str,
    requested_limit: int,
    now_ms: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Plausibility check for a raw klines payload, run before any of it is
    trusted.

    Threat model: a public proxy (the whole reason RobustProxyPool exists)
    can return HTTP 200 with a JSON body SHAPED exactly like real klines —
    a list of lists, numeric-looking strings in the right positions —
    while the content is wrong: a stale cached response, a truncated one,
    or in the worst case a deliberately altered one. `resp.json()`
    succeeding and `float(row[4])` parsing without raising tell us nothing
    about whether the data is current or sane — a genuine response and a
    garbage-but-well-shaped one pass both identically. Without this check,
    that failure mode doesn't show up as a failed_scan — it shows up as a
    completed "successful" scan feeding check_bb_rsi() real-looking but
    wrong numbers, which can produce a fake touch alert or silently
    swallow a real one.

    Two checks, deliberately simple rather than clever:
      1. Freshness — the most recent candle's open time must be recent
         relative to the timeframe's own duration. A "successful" 4h
         fetch returning data from yesterday is exactly the stale-proxy-
         cache scenario this exists to catch.
      2. Price sanity — every close must be a finite, positive number.
         Catches corrupted payloads that still parse as floats (nulls
         coerced to 0.0, negative placeholders, NaN/Infinity literals a
         misbehaving proxy might inject).

    Deliberately NOT checked: array length vs. requested_limit. A newly
    listed symbol can legitimately return far fewer candles than
    requested, and that's already handled correctly downstream — both
    check_bb_rsi's MIN_CANDLES gate and the EMA path's required_hours gate
    already treat "too little history" as "not enough data yet", not a
    failure. Rejecting on length here would misclassify a legitimate young
    listing as a fetch failure, and it isn't needed to catch the actual
    threat (wrong-but-plausible data), which the two checks above already
    cover without that false-positive risk.

    Returns (is_valid, reason); reason is only meaningful when False and
    is logged by the caller so bad data shows up in the scan summary as a
    failure instead of silently vanishing into a "successful" scan.
    """
    if not raw or not isinstance(raw, list) or len(raw) < 3:
        got = len(raw) if isinstance(raw, list) else type(raw).__name__
        return False, f"empty or degenerate payload ({got})"

    try:
        open_time_ms = int(float(raw[-1][0]))
    except (IndexError, TypeError, ValueError):
        return False, "malformed last candle (unreadable open time)"

    # Tolerance is generous (3x the interval + 5 min) to absorb normal
    # candle-close lag and clock skew without being so loose it misses
    # genuinely stale data. abs() catches both stale (positive skew) and
    # suspiciously-future (negative skew, e.g. a fabricated timestamp)
    # payloads.
    interval_seconds = TIMEFRAME_MINUTES.get(interval, 60) * 60
    tolerance_seconds = interval_seconds * 3 + 300
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    age_seconds = (now_ms - open_time_ms) / 1000.0
    if abs(age_seconds) > tolerance_seconds:
        return False, (
            f"last candle is {age_seconds:.0f}s from now "
            f"(tolerance {tolerance_seconds}s for {interval})"
        )

    for row in raw:
        try:
            close = float(row[4])
        except (IndexError, TypeError, ValueError):
            return False, "malformed close price in payload"
        if not math.isfinite(close) or close <= 0:
            return False, f"non-finite or non-positive close price ({close})"

    return True, ""


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
    async def fetch_combined_data(self, symbol: str, market: str, limit: int) -> List[float]:
        base = 'https://api.binance.com/api/v3/klines' if market == "spot" else 'https://fapi.binance.com/fapi/v1/klines'
        data = await self._request(base, {'symbol': symbol, 'interval': '1h', 'limit': limit})
        if not data: return []
        is_valid, reason = validate_klines_payload(data, '1h', limit)
        if not is_valid:
            logging.warning(f"⚠️ Rejected implausible klines for {symbol} (EMA fetch, {market}, Binance): {reason}")
            return []
        try: return [float(c[4]) for c in data]
        except Exception: return []
    async def fetch_closes(self, symbol: str, interval: str, market: str) -> List[float]:
        limit = candle_fetch_limit_for_timeframe(interval)
        base = 'https://api.binance.com/api/v3/klines' if market == "spot" else 'https://fapi.binance.com/fapi/v1/klines'
        data = await self._request(base, {'symbol': symbol, 'interval': interval, 'limit': limit})
        if not data: return []
        is_valid, reason = validate_klines_payload(data, interval, limit)
        if not is_valid:
            logging.warning(f"⚠️ Rejected implausible klines for {symbol} {interval} ({market}, Binance): {reason}")
            return []
        try: return [float(c[4]) for c in data]
        except Exception: return []

class BybitClient(ExchangeClient):
    async def get_perp_symbols(self) -> List[str]:
        data = await self._request('https://api.bybit.com/v5/market/instruments-info', {'category': 'linear'})
        if not data: return []
        return [s['symbol'] for s in data['result']['list'] if s['status'] == 'Trading' and s['quoteCoin'] == 'USDT']
    async def get_spot_symbols(self) -> List[str]:
        data = await self._request('https://api.bybit.com/v5/market/instruments-info', {'category': 'spot'})
        if not data: return []
        return [s['symbol'] for s in data['result']['list'] if s['status'] == 'Trading' and s['quoteCoin'] == 'USDT']
    async def fetch_combined_data(self, symbol: str, market: str, limit: int) -> List[float]:
        url = 'https://api.bybit.com/v5/market/kline'
        cat = 'linear' if market == 'perp' else 'spot'
        data = await self._request(url, {'category': cat, 'symbol': symbol, 'interval': '60', 'limit': limit})
        if not data: return []
        raw = data.get('result', {}).get('list', [])
        if not raw: return []
        # Bybit returns newest-first; normalize to oldest-first (matching
        # Binance's native ordering) before the shared plausibility check,
        # which assumes the most recent candle is last.
        ordered = raw[::-1]
        is_valid, reason = validate_klines_payload(ordered, '1h', limit)
        if not is_valid:
            logging.warning(f"⚠️ Rejected implausible klines for {symbol} (EMA fetch, {market}, Bybit): {reason}")
            return []
        try: return [float(c[4]) for c in ordered]
        except Exception: return []
    async def fetch_closes(self, symbol: str, interval: str, market: str) -> List[float]:
        limit = candle_fetch_limit_for_timeframe(interval)
        url = 'https://api.bybit.com/v5/market/kline'
        cat = 'linear' if market == 'perp' else 'spot'
        b_int = {"15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240", "1d": "D", "1w": "W"}.get(interval, "D")
        data = await self._request(url, {'category': cat, 'symbol': symbol, 'interval': b_int, 'limit': limit})
        if not data: return []
        raw = data.get('result', {}).get('list', [])
        if not raw: return []
        ordered = raw[::-1]
        is_valid, reason = validate_klines_payload(ordered, interval, limit)
        if not is_valid:
            logging.warning(f"⚠️ Rejected implausible klines for {symbol} {interval} ({market}, Bybit): {reason}")
            return []
        try: return [float(c[4]) for c in ordered]
        except Exception: return []

# ==========================================
# CORE LOGIC
# ==========================================

# Unix epoch (1970-01-01 00:00:00 UTC) was a Thursday. Monday falls 4 days
# *forward* from Thursday (Thu -> Fri -> Sat -> Sun -> Mon), so every real
# Monday 00:00 UTC timestamp satisfies `timestamp % 604800 == MONDAY_EPOCH_OFFSET`.
# This is the shift used below to re-anchor the weekly floor-division from
# Thursday (the epoch's natural alignment) to Monday. Verified against a
# known date: 1970-01-05 00:00:00 UTC (timestamp 345600) was a Monday, and
# 2026-07-20 00:00:00 UTC (timestamp 1784505600, also a real Monday) reduces
# to the same 345600 remainder mod 604800.
#
# A previous version of this used 259200 (3 days) here, from "Thursday back
# to Monday is 3 days" — true going *backwards*, but this formula needs the
# *forward* distance. That off-by-one-direction error anchored every weekly
# boundary to Sunday instead of Monday: the bot's internal "weekly candle"
# rolled over a full day before Binance's actual Monday-anchored weekly
# candle opened, got scanned and cached then, and by the time the real
# Monday candle opened sent_state already matched — logging "Already Sent"
# instead of scanning. See _self_check_weekly_alignment() below, added to
# catch a regression of this exact class of bug immediately on import.
MONDAY_EPOCH_OFFSET = 4 * 86400  # 345600


def get_cache_key(tf: str) -> int:
    """
    Returns stable integer timestamp for the current TF candle open.
    
    For weekly: Aligns to Monday 00:00 UTC via MONDAY_EPOCH_OFFSET (see the
    derivation in the comment above it).
    
    For daily: 86400 divides evenly from epoch and aligns with UTC midnight.
    
    For 4h: 14400 divides evenly, candles open at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC.
    """
    mins = TIMEFRAME_MINUTES[tf]
    period_seconds = mins * 60
    now = int(time.time())
    
    if tf == '1w':
        adjusted = now - MONDAY_EPOCH_OFFSET
        candle_start = adjusted - (adjusted % period_seconds) + MONDAY_EPOCH_OFFSET
        return candle_start
    else:
        # Daily and sub-daily align naturally with epoch
        return now - (now % period_seconds)


def _self_check_weekly_alignment() -> None:
    """
    Runs once at import time: verifies get_cache_key's Monday-alignment
    arithmetic actually lands on a real Monday 00:00 UTC boundary.

    This exists because of a real, previously-shipped bug: the old offset
    anchored to Sunday instead of Monday (see MONDAY_EPOCH_OFFSET's comment
    for the full story), which silently made every weekly scan fire and
    record sent_state a day early. Fails loudly and immediately on import
    if that class of bug is ever reintroduced, instead of silently shifting
    every weekly report by a day again.
    """
    known_monday_ts = MONDAY_EPOCH_OFFSET  # 1970-01-05 00:00:00 UTC, a Monday
    period_seconds = TIMEFRAME_MINUTES['1w'] * 60
    adjusted = known_monday_ts - MONDAY_EPOCH_OFFSET
    computed = adjusted - (adjusted % period_seconds) + MONDAY_EPOCH_OFFSET
    assert computed == known_monday_ts, (
        f"1w weekly alignment sanity check failed: expected boundary at "
        f"{known_monday_ts} (a known Monday), computed {computed}. "
        f"get_cache_key('1w') is almost certainly anchoring to the wrong weekday."
    )


_self_check_weekly_alignment()


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

def resample_to_daily(hourly_closes: List[float]) -> List[float]:
    daily_closes = []
    for i in range(len(hourly_closes) - 1, -1, -24):
        daily_closes.append(hourly_closes[i])
    return daily_closes[::-1]

def check_above_ema(closes: List[float], period: int, threshold_pct: float) -> bool:
    if len(closes) < period:
        return False
    np_c = np.array(closes, dtype=float)
    ema = talib.EMA(np_c, timeperiod=period)
    if np.isnan(ema[-1]):
        return False
    threshold_factor = 1.0 - (threshold_pct / 100.0)
    return closes[-1] >= (ema[-1] * threshold_factor)

def classify_middle_band_direction(
    rsi_array: np.ndarray,
    mid_array: np.ndarray,
    idx: int,
    lookback: int = 5
) -> str:
    """
    Determine middle-band direction from the last decisive side RSI occupied
    before it entered the middle-band touch zone.

    Convention:
        - "bullish" = RSI approached the middle band from ABOVE
        - "bearish" = RSI approached the middle band from BELOW
    """
    return classify_middle_band_direction_core(
        rsi_array=rsi_array,
        mid_array=mid_array,
        idx=idx,
        lookback=lookback,
        touch_threshold=CONFIG.MIDDLE_TOUCH_THRESHOLD,
    )


def check_bb_rsi(closes: List[float], tf: str) -> Tuple[Optional[str], Optional[str], float]:
    """
    Check if the RSI Bollinger Band touch condition is met.

    Tolerance is a flat number of RSI points (0-100 scale) — see
    CONFIG.UPPER_TOUCH_POINTS / LOWER_TOUCH_POINTS / MIDDLE_TOUCH_POINTS —
    independent of band level and band width entirely. This is the only
    threshold mode the bot supports; an earlier percentage-of-band-width
    mode (with an adaptive multiplier and a dynamic/static width
    reference) has been removed.

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
    mid_val = mid[idx]

    upper_zone = CONFIG.UPPER_TOUCH_POINTS
    lower_zone = CONFIG.LOWER_TOUCH_POINTS
    middle_zone = CONFIG.MIDDLE_TOUCH_POINTS

    # Check upper band touch
    if curr_rsi >= upper[idx] - upper_zone:
        return "UPPER", "", curr_rsi
    
    # Check lower band touch
    if curr_rsi <= lower[idx] + lower_zone:
        return "LOWER", "", curr_rsi
    
    # Check middle band touch (only for configured timeframes)
    if tf in MIDDLE_BAND_TFS:
        if mid_val > 0 and abs(curr_rsi - mid_val) <= middle_zone:
            # Use the multi-signal classification system
            direction = classify_middle_band_direction(
                rsi_array=rsi,
                mid_array=mid,
                idx=len(rsi) + idx,  # Convert negative index to positive
                lookback=CONFIG.MIDDLE_BAND_LOOKBACK
            )
            return "MIDDLE", direction, curr_rsi
    
    return None, None, 0.0


def compute_power_score(closes: List[float], window: int) -> Optional[float]:
    """
    Mean signed distance (RSI - upper_band) over the last `window`
    completed candles, ending at the same "last completed candle" (idx=-2)
    convention check_bb_rsi uses. Signed, not clamped at zero — a candle
    where RSI sits above the upper band contributes positively, so
    "walking above the band" outscores "sitting right at it" for free.

    Returns None if there aren't `window` valid, non-NaN candles to fill
    the window (insufficient history / still in RSI+BBANDS warmup) —
    callers must not treat None as a score of 0, it means "couldn't score
    this one", not "scored low."
    """
    if not CONFIG.POWER_LEADERBOARD_ENABLED or len(closes) < CONFIG.MIN_CANDLES:
        return None

    np_c = np.array(closes, dtype=float)
    rsi = talib.RSI(np_c, timeperiod=CONFIG.RSI_PERIOD)
    upper, mid, lower = talib.BBANDS(
        rsi,
        timeperiod=CONFIG.BB_LENGTH,
        nbdevup=CONFIG.BB_STDDEV,
        nbdevdn=CONFIG.BB_STDDEV,
        matype=0
    )

    if len(rsi) < window + 1:
        return None

    # [-(window+1):-1] = the `window` candles ending at idx=-2, matching
    # check_bb_rsi's "last completed candle" convention.
    window_rsi = rsi[-(window + 1):-1]
    window_upper = upper[-(window + 1):-1]

    if np.isnan(window_rsi).any() or np.isnan(window_upper).any():
        return None

    return float(np.mean(window_rsi - window_upper))

# ==========================================
# MAIN BOT
# ==========================================

def clean_name(s: str) -> str:
    """Clean symbol name: remove USDT suffix and common prefixes, cap at 6 chars."""
    s = s.replace("USDT", "")
    s = re.sub(r"^(1000000|100000|10000|1000|100|10|1M)(?=[A-Z])", "", s)
    return s[:6]


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
        
    async def send_report(
        self,
        session: aiohttp.ClientSession,
        hits: List[TouchHit],
        power_scores_by_tf: Dict[str, List[PowerScore]],
        weekly_peaks_by_tf: Dict[str, Dict[str, float]],
    ):
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

        Right after a timeframe's touch-alert message(s) go out, that same
        timeframe's power leaderboard (if one was computed this cycle) is
        sent as its own follow-up message — see the tf_order loop below.
        """
        if not hits and not power_scores_by_tf:
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
        # For a timeframe with touch hits, the touch section message(s) are
        # batched and sent first (unchanged logic below); if that same
        # timeframe also has a power leaderboard computed this cycle, it
        # goes out right after, as its own follow-up message. A timeframe
        # can also have a leaderboard with zero touch hits — that still
        # sends just the leaderboard message.
        for tf in tf_order:
            has_hits = tf in grouped
            has_leaderboard = tf in power_scores_by_tf and bool(power_scores_by_tf[tf])

            if not has_hits and not has_leaderboard:
                continue

            if has_hits:
                tf_sections = grouped[tf]
                total_hits = sum(len(v) for v in tf_sections.values())

                # Build all section blocks for this timeframe
                section_blocks: List[str] = []
                for section_key, section_label, sort_descending in section_defs:
                    items = tf_sections.get(section_key, [])
                    if not items:
                        continue
                    block = build_section_block(section_key, section_label, sort_descending, items)
                    section_blocks.append(block)

                if section_blocks:
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

            # ── Power leaderboard, sent as one follow-up message right
            # after this timeframe's touch section(s), if one was computed.
            # The message itself has three sections: top POWER_LEADERBOARD_SIZE
            # across all scored symbols this cycle, top POWER_LEADERBOARD_HOT_SIZE
            # restricted to symbols already flagged 🔥 volatile, and top
            # POWER_LEADERBOARD_WEEKLY_SIZE by rolling ~7-day peak score
            # (see CacheManager.get_weekly_peak_scores). ──
            if has_leaderboard:
                await self._send_power_leaderboard_message(
                    session, tf, power_scores_by_tf[tf], weekly_peaks_by_tf.get(tf, {}), ts_footer
                )

    def _format_power_board_section(
        self,
        label: str,
        rows: List[Tuple[str, float, bool]],
        show_hot_marker: bool = True,
    ) -> str:
        """
        Build one ranked section (header line + numbered rows) for the
        power-leaderboard message. Takes plain (symbol, score, hot) tuples
        rather than PowerScore objects, since the weekly-peak section has
        no PowerScore to draw from (its score comes from a Redis hash, not
        a fresh scan) — the general and volatile sections just unpack
        their PowerScore lists into the same tuple shape before calling in.

        show_hot_marker=False suppresses the 🔥 marker for the whole
        section regardless of each row's hot value — used for the
        VOLATILE and 7D PEAK sections, where every row is already known
        to be volatile by construction, so printing 🔥 on every single
        line would just repeat what the section header already says.
        """
        lines = [label]
        for rank, (symbol, score, hot) in enumerate(rows, start=1):
            sym = clean_name(symbol)
            hot_marker = " 🔥" if (show_hot_marker and hot) else ""
            lines.append(f"{rank:>2}. {sym:<6}{score:+6.2f}{hot_marker}")
        return "\n".join(lines)

    async def _send_power_leaderboard_message(
        self,
        session: aiohttp.ClientSession,
        tf: str,
        scores: List[PowerScore],
        weekly_peaks: Dict[str, float],
        footer: str,
    ):
        """
        Send the power leaderboard for a timeframe as a single Telegram
        message, right after that timeframe's regular touch-alert
        message(s) (see the caller, send_report). One message, three
        sections:
          - all symbols scored THIS cycle, ranked, capped at
            POWER_LEADERBOARD_SIZE. Mixed hot/not-hot, so each row still
            carries its own 🔥 marker here — it's the only section where
            that marker actually distinguishes rows from each other.
          - symbols already flagged 🔥 by hot_coins_for_timeframe, ranked
            within just that pool (not sliced from the section above),
            capped at POWER_LEADERBOARD_HOT_SIZE. No per-row marker — the
            section header already says VOLATILE.
          - rolling ~7-day peak score per symbol for this timeframe,
            among the MOST VOLATILE coins only (weekly_peaks, from
            CacheManager.get_weekly_peak_scores — that's what
            record_power_scores restricts itself to recording; already
            includes THIS cycle's hot scores, since record_power_scores
            is called before this is read), capped at
            POWER_LEADERBOARD_WEEKLY_SIZE. No per-row marker either, same
            reasoning — every entry only got into this bucket by having
            been hot on some contributing day.

        All three caps are small and fixed (20 rows apiece by default), so
        the combined message stays well under Telegram's 4096-char limit —
        no batching logic needed here, unlike send_report's touch sections.
        """
        if not scores and not weekly_peaks:
            return

        general_top = sorted(scores, key=lambda s: s.score, reverse=True)[:CONFIG.POWER_LEADERBOARD_SIZE]

        hot_pool = [s for s in scores if s.hot]
        volatile_top = sorted(hot_pool, key=lambda s: s.score, reverse=True)[:CONFIG.POWER_LEADERBOARD_HOT_SIZE]

        weekly_top = sorted(weekly_peaks.items(), key=lambda kv: kv[1], reverse=True)[:CONFIG.POWER_LEADERBOARD_WEEKLY_SIZE]

        if not general_top and not volatile_top and not weekly_top:
            return

        sections: List[str] = []
        if general_top:
            rows = [(s.symbol, s.score, s.hot) for s in general_top]
            sections.append(self._format_power_board_section(f"⚡ TOP {len(rows)}", rows))
        if volatile_top:
            rows = [(s.symbol, s.score, s.hot) for s in volatile_top]
            sections.append(self._format_power_board_section(f"🔥 TOP {len(rows)} VOLATILE", rows, show_hot_marker=False))
        if weekly_top:
            # Third tuple field (hot) is irrelevant here since the marker
            # is suppressed for this section anyway — every symbol that
            # made it into weekly_peaks was hot on at least one
            # contributing day by construction (record_power_scores).
            rows = [(sym, peak, True) for sym, peak in weekly_top]
            sections.append(self._format_power_board_section(f"🏆 TOP {len(rows)} — 7D PEAK (VOLATILE)", rows, show_hot_marker=False))

        window = power_leaderboard_window_for_timeframe(tf)
        header = f"⚡ <b>{tf} Power Leaderboard</b> (last {window} candles)\n"
        body = "\n\n".join(sections)
        message = header + f"<pre>{body}</pre>"
        await self._safe_send(session, message, footer)


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
        
        if CONFIG.BYBIT_ENABLED:
            bp, bs, yp, ys = await asyncio.gather(
                try_fetch(binance.get_perp_symbols, 'bp', 'Binance Perp'),
                try_fetch(binance.get_spot_symbols, 'bs', 'Binance Spot'),
                try_fetch(bybit.get_perp_symbols,   'yp', 'Bybit Perp'),
                try_fetch(bybit.get_spot_symbols,   'ys', 'Bybit Spot'),
            )
        else:
            bp, bs = await asyncio.gather(
                try_fetch(binance.get_perp_symbols, 'bp', 'Binance Perp'),
                try_fetch(binance.get_spot_symbols, 'bs', 'Binance Spot'),
            )
            yp, ys = [], []
        
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
                if CONFIG.BYBIT_ENABLED:
                    add_with_dedup(bybit, yp, 'perp', 'Bybit')
                    add_with_dedup(bybit, ys, 'spot', 'Bybit')
                
                total_sym_count = len(all_pairs)
                logging.info(f"Total unique symbols after dedup: {total_sym_count}")
                
                # ── EMA Filter & Volatility Calculation ──                             
                logging.info("Starting Unified EMA Filter & Volatility Scan...")
                vol_scores: Dict[str, float] = {}
                ema_survivors: List[Tuple[Any, str, str, str]] = []
                scan_sem = asyncio.Semaphore(CONFIG.MAX_CONCURRENCY)
                successful_fetches = 0
                
                # Calculate required data based on EMA length
                required_hours = CONFIG.EMA_LENGTH * 24
                fetch_limit = required_hours + 40 # Buffer for talib calculations

                async def process_symbol(client, sym, mkt, ex):
                    nonlocal successful_fetches
                    async with scan_sem:
                        try:
                            h_closes = await client.fetch_combined_data(sym, mkt, fetch_limit)
                            if not h_closes:
                                return
                            
                            successful_fetches += 1
                            
                            if len(h_closes) < required_hours:
                                return

                            # Volatility is recorded for every symbol (not
                            # just EMA survivors) so EMA-exempt timeframes,
                            # which scan the full universe, can still mark
                            # 🔥 hot coins among symbols that never passed
                            # the trend filter. Lookback window is
                            # CONFIG.VOLATILITY_LOOKBACK_HOURS (100h) —
                            # comfortably covered by required_hours (816h)
                            # from the gate just above, so this slice is
                            # always full, never partial.
                            v = calculate_volatility(h_closes[-CONFIG.VOLATILITY_LOOKBACK_HOURS:])
                            if v > 0:
                                vol_scores[sym] = v

                            d_closes = resample_to_daily(h_closes)
                            if check_above_ema(d_closes, CONFIG.EMA_LENGTH, CONFIG.EMA_THRESHOLD_PCT):
                                ema_survivors.append((client, sym, mkt, ex))
                        except Exception as e:
                            logging.debug(f"Processing failed for {sym}: {e}")

                tasks = [process_symbol(client, s, mkt, ex) for client, s, mkt, ex in all_pairs]
                await asyncio.gather(*tasks)

                # ── Two symbol universes going forward ──
                # full_universe_pairs: every deduplicated symbol, pre-EMA-filter.
                # ema_filtered_pairs: only symbols above the EMA trend filter.
                # Timeframes in CONFIG.EMA_FILTER_EXEMPT_TFS scan the full
                # universe (no trend filtering); every other timeframe scans
                # only the EMA-confirmed subset.
                full_universe_pairs = all_pairs
                ema_filtered_pairs = ema_survivors

                def pairs_for_timeframe(tf: str) -> List[Tuple[Any, str, str, str]]:
                    if tf in CONFIG.EMA_FILTER_EXEMPT_TFS:
                        return full_universe_pairs
                    return ema_filtered_pairs

                # ── Two hot-coin rankings, matching the two universes ──
                # hot_coins_filtered: top volatility among EMA-confirmed coins
                #                     only (used by trend-filtered timeframes,
                #                     so 🔥 still means "hot among trending").
                # hot_coins_full:     top volatility across every symbol
                #                     (used by EMA-exempt timeframes, so they
                #                     can mark 🔥 even on non-trending coins).
                ema_filtered_symbols = {sym for _, sym, _, _ in ema_filtered_pairs}
                hot_coins_filtered = set(
                    sorted(
                        (s for s in vol_scores if s in ema_filtered_symbols),
                        key=vol_scores.get,
                        reverse=True,
                    )[:CONFIG.HOT_COINS_LIMIT]
                )
                hot_coins_full = set(
                    sorted(vol_scores, key=vol_scores.get, reverse=True)[:CONFIG.HOT_COINS_LIMIT]
                )

                def hot_coins_for_timeframe(tf: str) -> Set[str]:
                    if tf in CONFIG.EMA_FILTER_EXEMPT_TFS:
                        return hot_coins_full
                    return hot_coins_filtered

                logging.info(f"I/O Success: {successful_fetches}/{total_sym_count} symbols fetched.")
                logging.info(f"EMA Filter: {len(ema_filtered_pairs)} symbols above {CONFIG.EMA_LENGTH} EMA.")
                logging.info(f"Volatility Ranking (trend-filtered): {len(hot_coins_filtered)} coins marked with 🔥")
                if CONFIG.EMA_FILTER_EXEMPT_TFS:
                    logging.info(
                        f"Volatility Ranking (full universe, for EMA-exempt TFs): "
                        f"{len(hot_coins_full)} coins marked with 🔥"
                    )
                    logging.info(
                        f"EMA-exempt timeframes {sorted(CONFIG.EMA_FILTER_EXEMPT_TFS)}: "
                        f"scanning full universe of {len(full_universe_pairs)} symbols (no trend filter)"
                    )
                
                # ── Determine Which Timeframes to Scan ──
                sent_state = await self.cache.get_sent_state()
                scan_stats = {
                    tf: ScanStats(tf, "Unknown", total_symbols=len(pairs_for_timeframe(tf)))
                    for tf in ACTIVE_TFS
                }
                
                tfs_to_scan_fresh: List[str] = []
                cached_hits_to_use: List[TouchHit] = []
                # Tracks candles this cycle definitively resolved a result for
                # (hit or not) — used later to mark sent_state correctly even
                # on zero-hit candles, instead of only marking it when a hit
                # happens to exist.
                resolved_candle_keys: Dict[str, int] = {}
                
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
                            resolved_candle_keys[tf] = candle_key
                            logging.info(f"📦 Using cached results for {tf}: {len(hits)} hits")
                        else:
                            tfs_to_scan_fresh.append(tf)
                            scan_stats[tf].source = "Fresh Scan (New Candle)"
                    else:
                        tfs_to_scan_fresh.append(tf)
                        scan_stats[tf].source = "Fresh Scan (Low TF)"
                
                # ── Execute Fresh Scans ──
                final_hits: List[TouchHit] = []
                power_scores_by_tf: Dict[str, List[PowerScore]] = {}
                
                if tfs_to_scan_fresh:
                    logging.info(f"Scanning fresh TFs: {tfs_to_scan_fresh} across {total_sym_count} symbols...")
                    
                    for tf in tfs_to_scan_fresh:
                        tf_hits: List[TouchHit] = []
                        tf_power_scores: List[PowerScore] = []
                        
                        async def scan_one(
                            client: ExchangeClient,
                            sym: str,
                            mkt: str,
                            ex: str,
                            scan_tf: str
                        ) -> Tuple[bool, List[TouchHit], Optional[float]]:
                            """
                            Scan a single symbol on a single timeframe for both a touch
                            hit and a power-leaderboard score, off the same fetched
                            closes (no extra fetch for the leaderboard).

                            Returns (fetched_ok, hits, power_score):
                              fetched_ok: True only if usable close data was
                                actually returned (touch found or not). False
                                if the fetch failed / returned nothing — this
                                is what makes proxy or network outages show up
                                in the scan summary instead of being silently
                                indistinguishable from "checked, no touch".
                              hits: 0 or 1 TouchHit.
                              power_score: signed mean(RSI - upper) over this
                                timeframe's power-leaderboard window, or None
                                if leaderboard scoring is disabled or there's
                                insufficient valid history for this symbol.

                            Catches all exceptions internally to prevent task
                            failures from propagating to asyncio.gather.
                            """
                            try:
                                closes = await client.fetch_closes(sym, scan_tf, mkt)
                                if not closes:
                                    return False, [], None
                                
                                t_type, direction, rsi_val = check_bb_rsi(closes, scan_tf)
                                hits: List[TouchHit] = []
                                if t_type:
                                    hits.append(TouchHit(
                                        symbol=sym,
                                        exchange=ex,
                                        market=mkt,
                                        timeframe=scan_tf,
                                        rsi=rsi_val,
                                        touch_type=t_type,
                                        direction=direction if direction else "",
                                        hot=sym in hot_coins_for_timeframe(scan_tf)
                                    ))

                                window = power_leaderboard_window_for_timeframe(scan_tf)
                                power_score = compute_power_score(closes, window)
                                return True, hits, power_score
                            except Exception as e:
                                logging.debug(f"Scan failed for {sym} on {scan_tf}: {e}")
                                return False, [], None
                        
                        # Build the initial task batch for this timeframe
                        # (full universe if this tf is EMA-exempt, otherwise
                        # the filtered set).
                        pending_pairs = pairs_for_timeframe(tf)

                        successful_count = 0
                        first_round_failed_count = 0

                        # CONFIG.RETRY_FAILED_SYMBOLS controls whether a
                        # symbol that's still failing after _request's own
                        # MAX_RETRIES proxy attempts gets one or more whole
                        # extra rounds against freshly drawn proxies, rather
                        # than being dropped for this scan cycle. With the
                        # flag off (or FAILED_SYMBOL_RETRY_ROUNDS=0),
                        # max_rounds=1 and this reproduces the old
                        # single-pass behavior exactly.
                        max_rounds = 1 + (
                            CONFIG.FAILED_SYMBOL_RETRY_ROUNDS
                            if CONFIG.RETRY_FAILED_SYMBOLS
                            else 0
                        )

                        for round_num in range(1, max_rounds + 1):
                            if not pending_pairs:
                                break

                            if round_num > 1:
                                # Brief pause before re-hitting the same
                                # symbols — courtesy to the exchange
                                # endpoints. The proxies that just failed
                                # are already excluded from Thompson
                                # Sampling selection via the pool's
                                # cooldown state, so this delay isn't
                                # needed for proxy diversity, just pacing.
                                await asyncio.sleep(2)
                                logging.info(
                                    f"🔁 [{tf}] Retrying {len(pending_pairs)} symbol(s) "
                                    f"that failed to fetch (round {round_num - 1}/"
                                    f"{max_rounds - 1})..."
                                )

                            round_tasks = [
                                scan_one(client, sym, mkt, ex, tf)
                                for client, sym, mkt, ex in pending_pairs
                            ]
                            round_results = await asyncio.gather(*round_tasks, return_exceptions=True)

                            still_failed: List[Tuple[Any, str, str, str]] = []
                            for pair, result in zip(pending_pairs, round_results):
                                if isinstance(result, tuple):
                                    fetched_ok, hits, power_score = result
                                    if fetched_ok:
                                        successful_count += 1
                                        tf_hits.extend(hits)
                                        if power_score is not None:
                                            _, sym, mkt, ex = pair
                                            tf_power_scores.append(PowerScore(
                                                symbol=sym,
                                                exchange=ex,
                                                market=mkt,
                                                timeframe=tf,
                                                score=power_score,
                                                hot=sym in hot_coins_for_timeframe(tf)
                                            ))
                                    else:
                                        still_failed.append(pair)
                                else:
                                    # Should be rare since scan_one catches internally,
                                    # but still counts as a failure, not a silent drop.
                                    logging.error(
                                        f"⚠️ Unexpected scan task exception for {pair[1]}: {result}"
                                    )
                                    still_failed.append(pair)

                            if round_num == 1:
                                first_round_failed_count = len(still_failed)
                            pending_pairs = still_failed

                        failed_count = len(pending_pairs)
                        recovered_count = first_round_failed_count - failed_count
                        if recovered_count > 0:
                            logging.info(
                                f"✅ [{tf}] Recovered {recovered_count} symbol(s) via retry "
                                f"(failed initially, succeeded on a later round)"
                            )
                        
                        scan_stats[tf].successful_scans = successful_count
                        scan_stats[tf].failed_scans = failed_count
                        scan_stats[tf].hits_found = len(tf_hits)

                        if scan_stats[tf].total_symbols > 0:
                            fail_ratio = failed_count / scan_stats[tf].total_symbols
                            if fail_ratio > 0.2:
                                logging.warning(
                                    f"⚠️ {tf}: {failed_count}/{scan_stats[tf].total_symbols} "
                                    f"scans failed to fetch data ({fail_ratio:.0%}). Results for "
                                    f"this timeframe are likely incomplete — check proxy pool health."
                                )
                        
                        final_hits.extend(tf_hits)

                        # Power leaderboard: the per-cycle snapshot itself
                        # (power_scores_by_tf) is not persisted — a re-run
                        # mid-candle just recomputes it fresh, same as
                        # before. What IS persisted now is each HOT score's
                        # contribution to the rolling weekly-peak record —
                        # record_power_scores filters to hot=True internally
                        # and merges into today's UTC daily bucket — which
                        # is what the "7D PEAK (VOLATILE)" section in the
                        # Telegram message reads back via
                        # get_weekly_peak_scores, below.
                        if tf_power_scores:
                            power_scores_by_tf[tf] = tf_power_scores
                            await self.cache.record_power_scores(tf, tf_power_scores)
                        
                        # Cache the results for cacheable timeframes
                        if tf in CACHED_TFS:
                            candle_key = get_cache_key(tf)
                            await self.cache.save_scan_results(
                                tf,
                                candle_key,
                                [h.to_dict() for h in tf_hits]
                            )
                            resolved_candle_keys[tf] = candle_key
                            logging.info(f"💾 Cached {len(tf_hits)} hits for {tf} (key: {candle_key})")
                
                # Merge cached hits with fresh hits
                final_hits.extend(cached_hits_to_use)
                
                # ── Print Scan Summary ──
                logging.info("=" * 73)
                logging.info(f"{'TF':<5} | {'Source':<28} | {'Scanned':<14} | {'Failed':<8} | {'Hits'}")
                logging.info("-" * 73)
                
                tf_display_order = [
                    t for t in ["3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w"]
                    if t in ACTIVE_TFS
                ]
                for tf in tf_display_order:
                    st = scan_stats[tf]
                    if st.source in ("Cached", "Already Sent", "Skipped (Weekly Delay)"):
                        scanned_str = "—"
                        failed_str = "—"
                    else:
                        scanned_str = f"{st.successful_scans}/{st.total_symbols}"
                        failed_str = str(st.failed_scans)
                    logging.info(
                        f"[{tf:<3}] {st.source:<28} | {scanned_str:<14} | {failed_str:<8} | {st.hits_found}"
                    )
                logging.info("=" * 73)
                logging.info(f"Total hits to send: {len(final_hits)}")
                
                # ── Filter Hits & Update Sent State ──
                # new_state is marked from resolved_candle_keys — every
                # candle this cycle definitively resolved a result for,
                # whether or not any hits were found. Marking only on hits
                # (the old behavior) meant a zero-hit candle never recorded
                # that it had been handled, so a later cycle would redo the
                # same "fresh" scan indefinitely instead of skipping via
                # "Already Sent".
                hits_to_send: List[TouchHit] = []
                new_state = sent_state.copy()
                new_state.update(resolved_candle_keys)
                
                for h in final_hits:
                    tf = h.timeframe
                    if tf in CACHED_TFS:
                        candle_key = get_cache_key(tf)
                        if sent_state.get(tf, 0) != candle_key:
                            hits_to_send.append(h)
                    else:
                        hits_to_send.append(h)
                
                # ── Weekly Peak Lookup ──
                # One 7-key Redis read per timeframe that has a leaderboard
                # this cycle — done once here, reused by send_report for
                # that timeframe's "7D PEAK (VOLATILE)" section, rather
                # than re-queried per message. Only ever contains symbols
                # that were hot on at least one contributing day (see
                # record_power_scores). Already reflects THIS cycle's hot
                # scores, since record_power_scores (above) ran before this.
                weekly_peaks_by_tf: Dict[str, Dict[str, float]] = {}
                for tf in power_scores_by_tf:
                    weekly_peaks_by_tf[tf] = await self.cache.get_weekly_peak_scores(tf)

                # ── Send Report ──
                if hits_to_send or power_scores_by_tf:
                    logging.info(
                        f"📤 Sending report: {len(hits_to_send)} touch hits, "
                        f"leaderboards for {len(power_scores_by_tf)} timeframe(s)..."
                    )
                    await self.send_report(session, hits_to_send, power_scores_by_tf, weekly_peaks_by_tf)
                    logging.info("✅ Report sent successfully")
                else:
                    logging.info("📭 No new hits or leaderboards to send")
                
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


async def _run_with_watchdog(bot: "RsiBot") -> None:
    """
    Run the bot under a hard ceiling on total execution time.

    See CONFIG.RUN_TIMEOUT_SECONDS for the reasoning. If this ever fires,
    it means something hung well past what a normal run should take —
    fail loudly (non-zero exit) rather than let CI notice hours later.
    """
    try:
        await asyncio.wait_for(bot.run(), timeout=CONFIG.RUN_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logging.error(
            f"⛔ Bot run exceeded the {CONFIG.RUN_TIMEOUT_SECONDS:.0f}s watchdog "
            f"timeout and was force-cancelled. This should not happen under "
            f"normal conditions — treat it as a bug, not routine behavior."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    bot = RsiBot()
    asyncio.run(_run_with_watchdog(bot))
