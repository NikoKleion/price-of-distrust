from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ALPHA_DB_PER_KM = 0.2
C_FIBER_KM_PER_S = 200_000.0

Q_STAR = 0.1100278644


def binary_entropy(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = (x > 0.0) & (x < 1.0)
    xm = x[m]
    out[m] = -xm * np.log2(xm) - (1.0 - xm) * np.log2(1.0 - xm)
    return out


def werner_w(fidelity):
    return (4.0 * np.asarray(fidelity, dtype=float) - 1.0) / 3.0


def werner_fidelity(w):
    return (1.0 + 3.0 * np.asarray(w, dtype=float)) / 4.0


def qber(w):
    return (1.0 - np.asarray(w, dtype=float)) / 2.0


def secret_key_fraction(w):
    return np.maximum(0.0, 1.0 - 2.0 * binary_entropy(qber(w)))


def link_success_probability(length_km, alpha_db_per_km=ALPHA_DB_PER_KM):
    return 10.0 ** (-alpha_db_per_km * np.asarray(length_km, dtype=float) / 10.0)


def n_star(fidelity):
    w0 = float(werner_w(fidelity))
    if w0 <= 0.0:
        return 0.0
    if w0 >= 1.0:
        return math.inf
    return math.log(1.0 - 2.0 * Q_STAR) / math.log(w0)


@dataclass(frozen=True)
class Chain:
    n_edges: int
    edge_km: float = 20.0
    fidelity: float = 0.96
    coherence_s: float = 10.0
    alpha_db_per_km: float = ALPHA_DB_PER_KM

    @property
    def p(self) -> float:
        return float(link_success_probability(self.edge_km, self.alpha_db_per_km))

    @property
    def t_att(self) -> float:
        return self.edge_km / C_FIBER_KM_PER_S

    @property
    def w0(self) -> float:
        return float(werner_w(self.fidelity))


def tn_mode_skr(chain: Chain) -> float:
    return float(chain.p / chain.t_att * secret_key_fraction(chain.w0))


def ent_mode_skr(chain: Chain, trials: int = 200_000, rng=None):
    rng = np.random.default_rng(12345) if rng is None else rng
    n, p, t_att, T = chain.n_edges, chain.p, chain.t_att, chain.coherence_s

    X = rng.geometric(p, size=(trials, n))

    if n > 1:
        waits = np.abs(np.diff(X, axis=1)).sum(axis=1) * t_att
        decoherence = np.exp(-waits / T)
    else:
        decoherence = np.ones(trials)

    w_e2e = (chain.w0 ** n) * decoherence
    skf = secret_key_fraction(w_e2e)

    cycle_s = X.max(axis=1) * t_att
    skr = float(skf.mean() / cycle_s.mean())
    return skr, float(w_e2e.mean()), float(skf.mean()), float(cycle_s.mean())


def price_of_distrust(chain: Chain, trials: int = 200_000, rng=None) -> float:
    tn = tn_mode_skr(chain)
    if tn <= 0.0:
        return float("nan")
    return ent_mode_skr(chain, trials, rng)[0] / tn


def distrust_horizon(edge_km, fidelity, coherence_s, n_max=40, trials=40_000,
                     rng=None) -> int:
    last_good = 0
    for n in range(1, n_max + 1):
        ch = Chain(n_edges=n, edge_km=edge_km, fidelity=fidelity,
                   coherence_s=coherence_s)
        if ent_mode_skr(ch, trials, rng)[0] > 0.0:
            last_good = n
        else:
            break
    return last_good
