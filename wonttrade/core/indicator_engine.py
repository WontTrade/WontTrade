"""Technical indicator computations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..models import Candle, IndicatorWindow, RawSymbolWindow, SymbolMarketData


def _safe_tail(values: Sequence[float]) -> float:
    return values[-1] if values else 0.0


def _to_floats(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def _closing_prices(candles: Sequence[Candle]) -> list[float]:
    return [float(c.close) for c in candles]


@dataclass(slots=True)
class IndicatorEngine:
    """Enrich raw market windows with derived indicators."""

    ema_period: int = 20
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    rsi_short_period: int = 7
    rsi_long_period: int = 14
    atr_short_period: int = 3
    atr_long_period: int = 14

    def enrich(self, window: RawSymbolWindow) -> SymbolMarketData:
        """Produce `SymbolMarketData` for the supplied raw window."""
        closes = _closing_prices(window.candles_3m)
        ema_series = self._ema_series(closes, self.ema_period)
        macd_series = self._macd_series(closes)
        rsi_short_series = self._rsi_series(closes, self.rsi_short_period)
        rsi_long_series = self._rsi_series(closes, self.rsi_long_period)

        atr_short_series = self._atr_series(window.candles_4h, self.atr_short_period)
        atr_long_series = self._atr_series(window.candles_4h, self.atr_long_period)

        closes_4h = _closing_prices(window.candles_4h)
        long_term_ema20 = self._ema(closes_4h, 20)
        long_term_ema50 = self._ema(closes_4h, 50)
        long_term_macd_series = self._macd_series(closes_4h)
        long_term_rsi_series = self._rsi_series(closes_4h, self.rsi_long_period)

        indicator_window = IndicatorWindow(
            timeframe="3m",
            mid_prices=closes,
            ema=ema_series,
            macd=macd_series,
            rsi_short=rsi_short_series,
            rsi_long=rsi_long_series,
            atr_short=atr_short_series,
            atr_long=atr_long_series,
        )

        return SymbolMarketData(
            symbol=window.symbol,
            current_price=_safe_tail(closes),
            ema20=_safe_tail(ema_series),
            macd=_safe_tail(macd_series),
            rsi_short=_safe_tail(rsi_short_series),
            open_interest=float(window.open_interest or 0.0),
            funding_rate=float(window.funding_rate or 0.0),
            volume=float(window.current_volume or 0.0),
            indicator_window=indicator_window,
            long_term_ema20=long_term_ema20,
            long_term_ema50=long_term_ema50,
            long_term_macd=_safe_tail(long_term_macd_series),
            long_term_rsi=_safe_tail(long_term_rsi_series),
            atr_short=_safe_tail(atr_short_series),
            atr_long=_safe_tail(atr_long_series),
        )

    def _ema_series(self, values: Sequence[float], period: int) -> list[float]:
        if not values:
            return []
        multiplier = 2 / (period + 1)
        ema_values: list[float] = []
        ema_prev = float(values[0])
        for price in values:
            ema_prev = (float(price) - ema_prev) * multiplier + ema_prev
            ema_values.append(ema_prev)
        return ema_values

    def _ema(self, values: Sequence[float], period: int) -> float:
        ema_series = self._ema_series(values, period)
        return ema_series[-1] if ema_series else 0.0

    def _macd_series(self, values: Sequence[float]) -> list[float]:
        if not values:
            return []
        fast_ema = self._ema_series(values, self.macd_fast_period)
        slow_ema = self._ema_series(values, self.macd_slow_period)
        min_len = min(len(fast_ema), len(slow_ema))
        if min_len == 0:
            return []
        macd_line = [fast_ema[-min_len + idx] - slow_ema[-min_len + idx] for idx in range(min_len)]
        # Pad to match input length
        if len(macd_line) < len(values):
            padding = [macd_line[0]] * (len(values) - len(macd_line))
            macd_line = padding + macd_line
        return macd_line

    def _rsi_series(self, values: Sequence[float], period: int) -> list[float]:
        length = len(values)
        if length == 0:
            return []
        if length <= period:
            return [50.0] * length
        gains: list[float] = []
        losses: list[float] = []
        for prev, current in zip(values[:-1], values[1:], strict=True):
            delta = current - prev
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        rsi_values: list[float] = [50.0] * period
        rsi_values.append(self._rsi_from_averages(avg_gain, avg_loss))
        for idx in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[idx]) / period
            avg_loss = (avg_loss * (period - 1) + losses[idx]) / period
            rsi_values.append(self._rsi_from_averages(avg_gain, avg_loss))
        # Trim or pad to match length of input values
        if len(rsi_values) < length:
            rsi_values = [rsi_values[0]] * (length - len(rsi_values)) + rsi_values
        elif len(rsi_values) > length:
            rsi_values = rsi_values[-length:]
        return rsi_values

    @staticmethod
    def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _atr_series(self, candles: Sequence[Candle], period: int) -> list[float]:
        if not candles:
            return []
        true_ranges: list[float] = []
        for idx, candle in enumerate(candles):
            high_low = candle.high - candle.low
            if idx == 0:
                true_ranges.append(high_low)
            else:
                prev_close = candles[idx - 1].close
                true_ranges.append(
                    max(
                        high_low,
                        abs(candle.high - prev_close),
                        abs(candle.low - prev_close),
                    )
                )
        if not true_ranges:
            return []
        atr_values: list[float] = []
        atr_prev = (
            sum(true_ranges[:period]) / period if len(true_ranges) >= period else true_ranges[0]
        )
        for tr in true_ranges:
            atr_prev = (atr_prev * (period - 1) + tr) / period
            atr_values.append(atr_prev)
        return atr_values
