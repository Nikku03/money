"""Attention over memory: the trader learns which past situations to attend to.

The kNN recall of P8 (memory.Analogs) averages the 50 nearest past situations, near by a fixed
distance where every feature counts the same. Attention learns what "near" should mean. One head,
scaled dot-product, as in a transformer, with its sharpness bounded (NORM): a first version put
91% of one pick's attention on a single week of March 2020, which is matching an outlier, not
learning from many cases. NORM was chosen on a synthetic problem with a planted answer only
(3: recall/truth correlation 0.75, ~7,700 of 9,600 cases attended; 6: 0.83, ~4,300; 10: 0.84,
~4,100), never on market data.

    query    q = Wq z            z: the present situation's features, standardized
    keys     k_j = Wk z_j        z_j: a remembered situation (a stock, a week) whose outcome is known
    weights  a_j = softmax_j(q . k_j / sqrt(D))
    recall   sum_j a_j v_j       v_j: how that situation turned out: its net under each of the five
                                 exits (winsorized at +-40%) and whether it hit +20%

Wq and Wk (D x features, D = 8) are learned by gradient descent (Adam, written out below) to make
the recall predict outcomes. It learns only from finished situations:
- each query is a situation from the latest 26 finished weeks;
- its keys come from situations at least 3 weeks older still, so every key's outcome was known
  before the query's decision;
- the last fifth of query weeks is held out to stop training at its best;
- it retrains every quarter.

At a decision it attends over every situation finished before that open (the 20,000 most recent).
``explain`` names the past situations it attended to most, for the journal.

It starts near uniform attention (small random weights), where its recall is just the average past
outcome. Anything it learns beyond that must come from the data.
"""
from __future__ import annotations

import numpy as np

D = 8
STEPS, BATCH_Q, BATCH_K, LR, L2 = 200, 1500, 6000, 0.01, 1e-3
NORM = 6.0          # each projection's Frobenius norm is held at or under this, so no logit can exceed
                    # about NORM**2 / sqrt(D) and one remembered case cannot take nearly all the weight
QUERY_WEEKS, GAP_WEEKS, CAP = 26, 3, 20000


class AttentionMemory:
    def __init__(self, d_in: int, n_values: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.Wq = self.rng.normal(scale=0.05, size=(D, d_in))
        self.Wk = self.rng.normal(scale=0.05, size=(D, d_in))
        self.mu = self.sd = None
        self.vscale = np.ones(n_values)
        self.rows = []                                     # (week, resolved session, X, V, meta)
        self.trained_at = None
        self.val_loss = None
        self.last_effective_n = None

    def add(self, week, resolved, X, V, meta):
        self.rows.append((week, int(resolved), X.astype(np.float32), V.astype(np.float32), list(meta)))

    def _z(self, X):
        """Standardized and scaled by 1/sqrt(features), so a situation's vector has length about 1."""
        return (np.clip((X - self.mu) / self.sd, -5, 5) / np.sqrt(X.shape[1])).astype(np.float32)

    def _known(self, d, cap=CAP, before_week=None):
        out, n = [], 0
        for r in reversed(self.rows):
            if r[1] >= d or (before_week is not None and r[0] >= before_week):
                continue
            if n >= cap:
                break
            out.append(r)
            n += len(r[2])
        return out

    @staticmethod
    def _forward(Zq, Zk, Wq, Wk, Vk):
        Q, K = Zq @ Wq.T, Zk @ Wk.T
        S = Q @ K.T / np.sqrt(D)
        S -= S.max(1, keepdims=True)
        A = np.exp(S)
        A /= A.sum(1, keepdims=True)
        return Q, K, A, A @ Vk

    def fit(self, d: int, week: int) -> None:
        """Retrain on finished situations only (see the module text)."""
        fin = [r for r in self.rows if r[1] < d]
        weeks = sorted({r[0] for r in fin})
        if len(weeks) < QUERY_WEEKS + GAP_WEEKS + 26:
            return
        qweeks = weeks[-QUERY_WEEKS:]
        if self.mu is None:
            A = np.vstack([r[2] for r in fin])
            self.mu, self.sd = A.mean(0), A.std(0) + 1e-9
            self.vscale = np.vstack([r[3] for r in fin]).std(0) + 1e-9
        qrows = [r for r in fin if r[0] in set(qweeks)]
        cut = qweeks[int(len(qweeks) * 0.8)]
        krows = self._known(d, before_week=qweeks[0] - GAP_WEEKS)
        Zk_all = self._z(np.vstack([r[2] for r in krows]))
        Vk_all = np.vstack([r[3] for r in krows]) / self.vscale
        tr = [r for r in qrows if r[0] < cut]
        va = [r for r in qrows if r[0] >= cut]
        Zq_tr, Yq_tr = self._z(np.vstack([r[2] for r in tr])), np.vstack([r[3] for r in tr]) / self.vscale
        Zq_va, Yq_va = self._z(np.vstack([r[2] for r in va])), np.vstack([r[3] for r in va]) / self.vscale
        m = {"Wq": np.zeros_like(self.Wq), "Wk": np.zeros_like(self.Wk)}
        v = {"Wq": np.zeros_like(self.Wq), "Wk": np.zeros_like(self.Wk)}
        W = {"Wq": self.Wq.copy(), "Wk": self.Wk.copy()}

        def val_loss(W):
            ks = self.rng.choice(len(Zk_all), size=min(BATCH_K, len(Zk_all)), replace=False)
            qs = self.rng.choice(len(Zq_va), size=min(BATCH_Q, len(Zq_va)), replace=False)
            _, _, _, Yh = self._forward(Zq_va[qs], Zk_all[ks], W["Wq"], W["Wk"], Vk_all[ks])
            return float(((Yh - Yq_va[qs]) ** 2).mean())

        best = (val_loss(W), {k: x.copy() for k, x in W.items()})
        for step in range(1, STEPS + 1):
            ks = self.rng.choice(len(Zk_all), size=min(BATCH_K, len(Zk_all)), replace=False)
            qs = self.rng.choice(len(Zq_tr), size=min(BATCH_Q, len(Zq_tr)), replace=False)
            Zq, Zk, Vk, Y = Zq_tr[qs], Zk_all[ks], Vk_all[ks], Yq_tr[qs]
            Q, K, A, Yh = self._forward(Zq, Zk, W["Wq"], W["Wk"], Vk)
            dY = 2 * (Yh - Y) / Y.size
            dA = dY @ Vk.T
            dS = A * (dA - (dA * A).sum(1, keepdims=True)) / np.sqrt(D)
            g = {"Wq": (dS @ K).T @ Zq + L2 * W["Wq"], "Wk": (dS.T @ Q).T @ Zk + L2 * W["Wk"]}
            for k in W:                                      # Adam
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                W[k] -= LR * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
                nrm = np.linalg.norm(W[k])
                if nrm > NORM:
                    W[k] *= NORM / nrm
            if step % 25 == 0:
                vl = val_loss(W)
                if vl < best[0]:
                    best = (vl, {k: x.copy() for k, x in W.items()})
        self.Wq, self.Wk = best[1]["Wq"], best[1]["Wk"]
        self.val_loss, self.trained_at = best[0], week

    def recall(self, d: int, X: np.ndarray, top: int = 0):
        """Attention-weighted recall of finished situations for each row of X; with ``top``, also the
        most-attended past situations of each row: (meta, weight, value)."""
        if self.mu is None:
            return np.zeros((len(X), len(self.vscale))), None
        known = self._known(d)
        if not known:
            return np.zeros((len(X), len(self.vscale))), None
        Zk = self._z(np.vstack([r[2] for r in known]))
        Vk = np.vstack([r[3] for r in known])
        _, _, A, Yh = self._forward(self._z(X), Zk, self.Wq, self.Wk, Vk)
        self.last_effective_n = float(np.mean(1.0 / (A * A).sum(1)))   # how many cases it effectively attends to
        if not top:
            return Yh, None
        meta = [m for r in known for m in r[4]]
        idx = np.argsort(-A, axis=1)[:, :top]
        return Yh, [[(meta[j], float(A[i, j]), Vk[j]) for j in idx[i]] for i in range(len(X))]
