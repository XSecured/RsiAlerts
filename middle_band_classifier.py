from typing import Optional

import numpy as np


def _normalize_index(length: int, idx: int) -> int:
    return idx if idx >= 0 else length + idx


def _touch_tolerance(mid_value: float, touch_threshold: float) -> float:
    return abs(mid_value) * touch_threshold


def _classify_side(diff: float, tolerance: float) -> Optional[str]:
    if np.isnan(diff) or np.isnan(tolerance):
        return None
    if diff > tolerance:
        return "above"
    if diff < -tolerance:
        return "below"
    return "inside"


def _direction_from_side(side: str) -> str:
    return "bullish" if side == "above" else "bearish"


def classify_middle_band_direction(
    rsi_array: np.ndarray,
    mid_array: np.ndarray,
    idx: int,
    lookback: int = 5,
    touch_threshold: float = 0.035,
) -> str:
    """
    Classify a middle-band touch by the side RSI approached the touch zone from.

    The key signal is the last decisive state before RSI entered the middle-band
    touch zone:
      - above midline -> bullish support retest
      - below midline -> bearish resistance retest

    When RSI has hugged the middle band for the whole lookback, fall back to a
    recency-weighted history of decisive above/below states.
    """
    if len(rsi_array) == 0 or len(mid_array) == 0:
        return "bullish"

    idx = _normalize_index(len(rsi_array), idx)
    idx = max(0, min(idx, len(rsi_array) - 1))
    start_idx = max(0, idx - max(lookback, 1))

    states = []
    for i in range(start_idx, idx + 1):
        rsi_value = rsi_array[i]
        mid_value = mid_array[i]
        if np.isnan(rsi_value) or np.isnan(mid_value):
            states.append((i, np.nan, np.nan, None))
            continue

        diff = rsi_value - mid_value
        tolerance = _touch_tolerance(mid_value, touch_threshold)
        side = _classify_side(diff, tolerance)
        states.append((i, diff, tolerance, side))

    current_side = states[-1][3]
    if current_side is None:
        return "bullish"

    touch_cluster_start = len(states) - 1
    while touch_cluster_start > 0 and states[touch_cluster_start - 1][3] == "inside":
        touch_cluster_start -= 1

    for pos in range(touch_cluster_start - 1, -1, -1):
        side = states[pos][3]
        if side in {"above", "below"}:
            return _direction_from_side(side)

    score = 0.0
    weight = 1.0
    for pos in range(len(states) - 2, -1, -1):
        side = states[pos][3]
        if side == "above":
            score += weight
            weight += 1.0
        elif side == "below":
            score -= weight
            weight += 1.0

    if score > 0:
        return "bullish"
    if score < 0:
        return "bearish"

    current_diff = states[-1][1]
    if np.isnan(current_diff):
        return "bullish"
    return "bullish" if current_diff >= 0 else "bearish"
