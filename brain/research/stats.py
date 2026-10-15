"""Small, dependency-free statistics: exact binomial tests, a Clopper-Pearson bound, Benjamini-Hochberg."""
from __future__ import annotations

import math


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pmf[k]
    return min(1.0, sum(x for x in pmf if x <= obs * (1 + 1e-9)))


def _pmf(i: int, n: int, p: float) -> float:
    if n <= 1000:
        return math.comb(n, i) * p**i * (1 - p) ** (n - i)
    # math.comb outgrows a float past n = 1000 or so: the same term through logs
    return math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * math.log(p) + (n - i) * math.log1p(-p))


def binom_one_sided_greater(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p): how often a coin with hit rate p does at least this well."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return min(1.0, sum(_pmf(i, n, p) for i in range(k, n + 1)))


def binom_lower_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """The exact one-sided Clopper-Pearson lower bound on a hit rate from k hits in n: the rate L at which
    P(X >= k | L) = alpha, so a true rate below L would show k or more hits less than alpha of the time.
    0 when there are no hits. Found by bisection on ``binom_one_sided_greater`` (it rises with the rate)."""
    if n <= 0 or k <= 0:
        return 0.0
    k = min(k, n)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom_one_sided_greater(k, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return lo


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), same order as input."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, pvals[i] * n / rank)
        adj[i] = running
    return adj
