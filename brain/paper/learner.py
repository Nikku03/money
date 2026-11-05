"""The trader's learning: a linear model of next-week excess return, updated by every closed trade.

Recursive least squares with a ridge prior and forgetting:
    prior      every weight starts at 0 with variance ``PRIOR``: it trades on nothing until outcomes
               pull a weight away from 0.
    update     when a position closes, its features x and the excess return y it earned update the
               weights (exactly the ridge regression on all closed trades so far, discounted).
    forgetting each update first discounts old evidence by ``FORGET`` (0.995 per trade-week: a
               half-life of about 140 updates), so lessons that stop working fade.
It never sees an outcome before the trade that produced it has closed.
"""
from __future__ import annotations

import numpy as np

PRIOR = 1e-4          # prior variance of each weight (returns are ~1e-2 per week; features are z-scores)
FORGET = 0.995


class Learner:
    def __init__(self, names: list[str], prior: float = PRIOR, forget: float = FORGET):
        self.names = list(names)
        k = len(names)
        self.w = np.zeros(k)
        self.P = np.eye(k) * prior
        self.forget = forget
        self.updates = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def update(self, X: np.ndarray, y: np.ndarray, noise: float = 1e-3) -> None:
        """One batch of closed trades (rows of X, outcomes y), as one forgetting step."""
        self.P /= self.forget
        for x, t in zip(np.atleast_2d(X), np.atleast_1d(y)):
            Px = self.P @ x
            g = Px / (noise + x @ Px)
            self.w = self.w + g * (t - x @ self.w)
            self.P = self.P - np.outer(g, Px)
        self.updates += 1

    def state(self) -> dict:
        return {"names": self.names, "w": self.w.tolist(), "P": self.P.tolist(), "forget": self.forget,
                "updates": self.updates}

    def extend(self, names: list[str], prior: float = PRIOR) -> None:
        """Add features it has never seen: weight 0, prior variance, uncorrelated with what it knows."""
        new = [n for n in names if n not in self.names]
        k, m = len(self.names), len(new)
        P = np.eye(k + m) * prior
        P[:k, :k] = self.P
        self.P, self.w, self.names = P, np.concatenate([self.w, np.zeros(m)]), self.names + new

    @classmethod
    def load(cls, s: dict) -> "Learner":
        L = cls(s["names"], forget=s["forget"])
        L.w, L.P, L.updates = np.array(s["w"]), np.array(s["P"]), s["updates"]
        return L
