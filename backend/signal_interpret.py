# backend/signal_interpret.py
from __future__ import annotations
import time
from typing import Iterable, List, Optional

class SignalInterpreter:
    """
    Turns raw multi-channel samples into coarse directions.
    Current heuristic:
      H = ch1 - ch3  (1-based -> indices 0 and 2)
      V = ch8 - ch2  (1-based -> indices 7 and 1)
    Then threshold + debounce to emit: left/right/up/down
    """

    def __init__(
        self,
        h_idx_a: int = 0, h_idx_b: int = 2,    # ch1 - ch3
        v_idx_a: int = 7, v_idx_b: int = 1,    # ch8 - ch2
        ema_alpha: float = 0.2,                # smoothing (0..1)
        thresh_h: float = 0.02,                # tune these for your signal scale
        thresh_v: float = 0.02,
        cooldown_ms: int = 250                 # don’t spam directions too fast
    ):
        self.h_idx_a = h_idx_a
        self.h_idx_b = h_idx_b
        self.v_idx_a = v_idx_a
        self.v_idx_b = v_idx_b

        self.ema_alpha = ema_alpha
        self.h_ema = 0.0
        self.v_ema = 0.0

        self.thresh_h = thresh_h
        self.thresh_v = thresh_v

        self.cooldown_ms = cooldown_ms
        self._last_emit_t = 0.0
        self._last_cmd = None

    def reset(self):
        self.h_ema = 0.0
        self.v_ema = 0.0
        self._last_emit_t = 0.0
        self._last_cmd = None

    def _maybe_emit(self, h: float, v: float) -> Optional[str]:
        now = time.time() * 1000.0
        if now - self._last_emit_t < self.cooldown_ms:
            return None

        cmd = None
        # Horizontal dominance
        if abs(h) > max(abs(v), self.thresh_h):
            cmd = "right" if h > 0 else "left"
        # Vertical dominance
        elif abs(v) > max(abs(h), self.thresh_v):
            cmd = "up" if v > 0 else "down"

        # Don’t emit the exact same command repeatedly if nothing changed
        if cmd and cmd != self._last_cmd:
            self._last_cmd = cmd
            self._last_emit_t = now
            return cmd
        return None

    def process_chunk(self, chunk: List[List[float]]) -> Optional[str]:
        """
        chunk: list of samples; each sample is a list[channels]
        Returns a single direction if one is detected in this chunk (latest wins).
        """
        cmd_out = None
        for s in chunk:
            try:
                h = float(s[self.h_idx_a]) - float(s[self.h_idx_b])
                v = float(s[self.v_idx_a]) - float(s[self.v_idx_b])
            except Exception:
                continue

            # EMA smoothing
            a = self.ema_alpha
            self.h_ema = (1 - a) * self.h_ema + a * h
            self.v_ema = (1 - a) * self.v_ema + a * v

            maybe = self._maybe_emit(self.h_ema, self.v_ema)
            if maybe:
                cmd_out = maybe
        return cmd_out
