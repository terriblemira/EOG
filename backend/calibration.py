# backend/calibration.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import statistics as stats

from signal_interpret import CalibrationConfig

# Simple robust stats
def robust_center(xs: List[float]) -> float:
    return stats.median(xs) if xs else 0.0

def robust_scale(xs: List[float]) -> float:
    # MAD ~ 1.4826 * median(|x - median|)
    if not xs: return 1.0
    m = stats.median(xs)
    mad = stats.median([abs(x - m) for x in xs]) or 1e-6
    return 1.4826 * mad

@dataclass
class CalPhaseData:
    # raw axis values (already H = chA - chB, V = chA - chB)
    H: List[float]
    V: List[float]

class CalibrationSession:
    """
    Collects samples for named phases: center, left, right, up, down.
    Converts to a CalibrationConfig with baselines, gains, deadzones, thresholds.
    """
    def __init__(self, h_idx=(0,2), v_idx=(7,1)):
        self.h_idx = h_idx
        self.v_idx = v_idx
        self.data: Dict[str, CalPhaseData] = {
            "center": CalPhaseData([], []),
            "left":   CalPhaseData([], []),
            "right":  CalPhaseData([], []),
            "up":     CalPhaseData([], []),
            "down":   CalPhaseData([], []),
        }

    def feed_chunk(self, phase: str, chunk: List[List[float]]):
        hi, hj = self.h_idx
        vi, vj = self.v_idx
        store = self.data.get(phase)
        if store is None: return
        for s in chunk:
            try:
                H = float(s[hi]) - float(s[hj])
                V = float(s[vi]) - float(s[vj])
            except Exception:
                continue
            store.H.append(H)
            store.V.append(V)

    def compute_config(self) -> CalibrationConfig:
        # Baseline from center
        h0 = robust_center(self.data["center"].H)
        v0 = robust_center(self.data["center"].V)

        # Peak magnitudes (after baseline) from directional looks
        def centered(vals): return [x - (h0 if axis=="H" else v0) for x in vals]
        axis = "H"  # just placeholder var to reuse centered()

        left_mag  = [-(x - h0) for x in self.data["left"].H]   # H < 0 for left, flip sign
        right_mag = [ (x - h0) for x in self.data["right"].H]  # H > 0 for right
        up_mag    = [ (y - v0) for y in self.data["up"].V]     # V > 0 for up
        down_mag  = [-(y - v0) for y in self.data["down"].V]   # V < 0 for down, flip

        # Robust peak (use 80th percentile of magnitudes)
        def pctl(xs, p):
            if not xs: return 0.0
            xs2 = sorted(xs); k = max(0, min(len(xs2)-1, int(p/100.0 * (len(xs2)-1))))
            return xs2[k]

        h_peak = max(pctl(left_mag, 80), pctl(right_mag, 80), 1e-6)
        v_peak = max(pctl(up_mag,   80), pctl(down_mag,  80), 1e-6)

        # Gains: map 80th percentile “comfortable look” to ~1.0
        h_gain = 1.0 / h_peak
        v_gain = 1.0 / v_peak

        # Noise floor from center fixations
        h_noise = robust_scale([x - h0 for x in self.data["center"].H])  # ~MAD
        v_noise = robust_scale([y - v0 for y in self.data["center"].V])

        # Deadzone: a few noise sigmas after gain
        h_dead = 3.0 * h_noise * h_gain
        v_dead = 3.0 * v_noise * v_gain

        # Decision thresholds between deadzone and peak (e.g., 35% of normalized peak)
        # You can make this asymmetric if needed.
        thresh_h = max(h_dead * 1.2, 0.35)   # normalized units
        thresh_v = max(v_dead * 1.2, 0.35)

        cfg = CalibrationConfig(
            h_idx_a=self.h_idx[0], h_idx_b=self.h_idx[1],
            v_idx_a=self.v_idx[0], v_idx_b=self.v_idx[1],
            h_baseline=h0, v_baseline=v0,
            h_gain=h_gain, v_gain=v_gain,
            h_deadzone=h_dead, v_deadzone=v_dead,
            thresh_h=thresh_h, thresh_v=thresh_v,
            ema_alpha=0.2, cooldown_ms=200
        )
        return cfg
