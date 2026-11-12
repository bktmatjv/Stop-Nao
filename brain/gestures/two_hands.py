"""Two-hands hold detector (pure logic, no camera): e.g. both thumbs up for ~1 s."""

from __future__ import annotations

import time
from typing import Callable, Sequence


class TwoHandsHold:
    """Fires once when >= ``min_hands`` hands all show ``label`` for ``hold`` seconds.

    After firing, input is ignored for ``cooldown`` seconds. Any frame that
    breaks the condition resets the timer.
    """

    def __init__(self, label: str = "THUMBS_UP", hold: float = 1.0, min_hands: int = 2,
                 cooldown: float = 2.0, clock: Callable[[], float] = time.perf_counter):
        self.label = label
        self.hold = hold
        self.min_hands = min_hands
        self.cooldown = cooldown
        self._clock = clock
        self._since: float | None = None
        self._cooldown_until: float | None = None

    def reset(self) -> None:
        self._since = None
        self._cooldown_until = None

    def update(self, labels: Sequence[str], now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        if self._cooldown_until is not None:
            if now < self._cooldown_until:
                return False
            self._cooldown_until = None
        if len(labels) < self.min_hands or any(l != self.label for l in labels):
            self._since = None
            return False
        if self._since is None:
            self._since = now
        if now - self._since >= self.hold:
            self._since = None
            self._cooldown_until = now + self.cooldown
            return True
        return False
