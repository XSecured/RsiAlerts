import unittest

import numpy as np

from middle_band_classifier import classify_middle_band_direction


class TestMiddleBandClassifier(unittest.TestCase):
    def test_classifies_touch_from_above_as_bullish(self):
        mid = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        rsi = np.array([58.0, 56.0, 54.0, 52.4, 51.1, 50.6])

        direction = classify_middle_band_direction(rsi, mid, idx=5, lookback=5)

        self.assertEqual(direction, "bullish")

    def test_classifies_touch_from_below_as_bearish(self):
        mid = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        rsi = np.array([42.0, 44.0, 46.0, 48.0, 49.2, 50.1])

        direction = classify_middle_band_direction(rsi, mid, idx=5, lookback=5)

        self.assertEqual(direction, "bearish")

    def test_keeps_entry_side_when_touch_zone_spans_multiple_candles(self):
        mid = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        rsi = np.array([57.0, 55.0, 52.5, 51.1, 49.9, 50.3])

        direction = classify_middle_band_direction(rsi, mid, idx=5, lookback=5)

        self.assertEqual(direction, "bullish")

    def test_uses_latest_decisive_side_in_noisy_history(self):
        mid = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        rsi = np.array([55.0, 53.0, 48.2, 47.8, 49.1, 49.7, 50.0])

        direction = classify_middle_band_direction(rsi, mid, idx=6, lookback=6)

        self.assertEqual(direction, "bearish")


if __name__ == "__main__":
    unittest.main()
