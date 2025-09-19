from __future__ import annotations
from dataclasses import dataclass
import time
from typing import List, Optional

@dataclass
class CalibrationConfig:
    # indices for H and V
    h_idx_a: int = 0
    h_idx_b: int = 2
    v_idx_a: int = 7
    v_idx_b: int = 1

    # baseline (center fixation)
    h_baseline: float = 0.0
    v_baseline: float = 0.0

    # gains to normalize comfortable looks to ~±1.0
    h_gain: float = 1.0
    v_gain: float = 1.0

    # deadzone (noise floor)
    h_deadzone: float = 0.05
    v_deadzone: float = 0.05

    # decision thresholds (after gain/centering)
    thresh_h: float = 0.25
    thresh_v: float = 0.25

    # smoothing + debounce
    ema_alpha: float = 0.2
    cooldown_ms: int = 200

class SignalInterpreter:
    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.cfg = config or CalibrationConfig()
        self.h_ema = 0.0
        self.v_ema = 0.0
        self._last_emit_t = 0.0
        self._last_cmd = None

    def update_config(self, cfg: CalibrationConfig):
        self.cfg = cfg
        self.h_ema = 0.0
        self.v_ema = 0.0
        self._last_emit_t = 0.0
        self._last_cmd = None

    def reset(self):
        self.h_ema = 0.0
        self.v_ema = 0.0
        self._last_emit_t = 0.0
        self._last_cmd = None

    def _maybe_emit(self, h: float, v: float) -> Optional[str]:
        now = time.time() * 1000.0
        if now - self._last_emit_t < self.cfg.cooldown_ms:
            return None

        # deadzone
        if abs(h) < self.cfg.h_deadzone and abs(v) < self.cfg.v_deadzone:
            return None

        cmd = None
        # choose dominant axis first, then threshold
        if abs(h) >= abs(v) and abs(h) >= self.cfg.thresh_h:
            cmd = "right" if h > 0 else "left"
        elif abs(v) > abs(h) and abs(v) >= self.cfg.thresh_v:
            cmd = "up" if v > 0 else "down"

        if cmd and cmd != self._last_cmd:
            self._last_cmd = cmd
            self._last_emit_t = now
            return cmd
        return None

    def process_chunk(self, chunk: List[List[float]]) -> Optional[str]:
        a = self.cfg.ema_alpha
        cmd_out = None
        for s in chunk:
            try:
                raw_h = float(s[self.cfg.h_idx_a]) - float(s[self.cfg.h_idx_b])
                raw_v = float(s[self.cfg.v_idx_a]) - float(s[self.cfg.v_idx_b])
            except Exception:
                continue

            # center + gain
            h = (raw_h - self.cfg.h_baseline) * self.cfg.h_gain
            v = (raw_v - self.cfg.v_baseline) * self.cfg.v_gain

            # EMA smoothing
            self.h_ema = (1 - a) * self.h_ema + a * h
            self.v_ema = (1 - a) * self.v_ema + a * v

            maybe = self._maybe_emit(self.h_ema, self.v_ema)
            if maybe:
                cmd_out = maybe
        return cmd_out
