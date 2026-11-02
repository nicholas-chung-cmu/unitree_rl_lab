"""Unitree stick mapping and local, timestamped wireless input reader."""
import json
import time
from pathlib import Path
import numpy as np


def stick_velocity(lx, ly, rx, deadband=.08, limits=(.05,.04,.10)):
    axes = np.array([ly, -lx, -rx], dtype=float)
    if not np.isfinite(axes).all() or np.max(np.abs(axes)) > 1.05:
        raise ValueError('Invalid wireless stick values')
    axes = np.clip(axes, -1, 1)
    axes = np.sign(axes)*np.maximum(np.abs(axes)-deadband, 0)/(1-deadband)
    # Conservative dance range; tempo remains 90 BPM at all stick positions.
    limits=np.asarray(limits,dtype=float)
    if limits.shape!=(3,) or not np.isfinite(limits).all() or (limits<0).any():
        raise ValueError('Invalid joystick command limits')
    return axes*limits


class WirelessInput:
    """Reads an atomic JSON snapshot from wireless_receiver; never opens DDS."""
    def __init__(self, path, timeout=.25, limits=(.05,.04,.10)):
        self.path = Path(path)
        self.timeout = timeout
        self.status = 'waiting'
        self.stop = False
        self.limits=limits

    def read(self, now=None):
        now = time.monotonic() if now is None else now
        try:
            value = json.loads(self.path.read_text())
            age = now-float(value['monotonic_seconds'])
            if not 0 <= age <= self.timeout:
                self.status = 'stale'
                return np.zeros(3)
            command = stick_velocity(value['lx'], value['ly'], value['rx'],limits=self.limits)
            # SDK key bit 9 is B. Latch simulation stop until process restart.
            self.stop |= bool(int(value['keys']) & (1 << 9))
            self.status = 'connected'
            return command
        except (OSError, ValueError, KeyError, TypeError):
            self.status = 'unavailable'
            return np.zeros(3)


def scripted_sticks(seconds):
    """Includes neutral, forward/backward, lateral, turns, mixed and neutral again."""
    cases = [(0, 0, 0), (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0),
             (0, 0, -1), (0, 0, 1), (0, .816, -1), (0, 0, 0)]
    return cases[min(int(seconds/4.1), len(cases)-1)]
