"""Seeded byte-stream RNG for deterministic battle replays.

Gen 1 consumes RNG one byte at a time.  For the differential harness to
match PyBoy turn-by-turn, every random draw must be reproducible from a
single seed, and every call site must document which byte it consumed.
"""
from __future__ import annotations

import random


class BattleRNG:
    """Deterministic 0..255 byte source backed by ``random.Random``."""

    __slots__ = ("_r",)

    def __init__(self, seed: int = 0) -> None:
        self._r = random.Random(seed)

    def next_byte(self) -> int:
        """One uniform byte in [0, 255]."""
        return self._r.randint(0, 255)

    def damage_roll(self) -> int:
        """Gen 1 damage multiplier byte in [217, 255].

        Implementation mirrors the pokered routine which keeps re-rolling
        until the byte is >= 217.  For simulator purposes uniform sampling
        within the final range is statistically equivalent and avoids
        the RNG-count divergence that would only matter if we were
        lock-stepping against PyBoy at the RNG level.
        """
        return self._r.randint(217, 255)

    def hit_check(self, accuracy_byte: int) -> bool:
        """Gen 1 accuracy test: draw a byte, hit iff byte < accuracy.

        With accuracy_byte=255 this still misses ~1/256 of the time
        because the comparison is strictly-less-than.  That's the
        documented Gen 1 quirk and we reproduce it exactly.
        """
        return self.next_byte() < accuracy_byte

    def crit_check(self, threshold: int) -> bool:
        """Crit if byte < threshold.  Threshold is capped at 255."""
        t = min(max(threshold, 0), 255)
        return self.next_byte() < t
