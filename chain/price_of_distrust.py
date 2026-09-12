from __future__ import annotations

import math

import numpy as np

from repeater_model import (
    ALPHA_DB_PER_KM,
    Q_STAR,
    Chain,
    binary_entropy,
    distrust_horizon,
    ent_mode_skr,
    link_success_probability,
    n_star,
    price_of_distrust,
    secret_key_fraction,
    tn_mode_skr,
    werner_fidelity,
    werner_w,
)

NO_MEMORY_NOISE_S = 1.0e12


def expected_max_geometric(p: float, n: int, k_max: int = 2_000_000) -> float:
    k = np.arange(0, k_max, dtype=float)
    tail = 1.0 - (1.0 - (1.0 - p) ** k) ** n
    return float(tail.sum())


def decompose(chain: Chain, trials: int = 200_000, rng=None):
    _, _, mean_skf, mean_cycle = ent_mode_skr(chain, trials, rng)
    fidelity_factor = mean_skf / float(secret_key_fraction(chain.w0))
    rate_factor = (chain.t_att / chain.p) / mean_cycle
    return fidelity_factor * rate_factor, fidelity_factor, rate_factor


def validate(verbose: bool = True) -> bool:
    checks: list[tuple[str, bool]] = []
    rng = np.random.default_rng(7)

    def near(a, b, tol=1e-9):
        return abs(float(a) - float(b)) <= tol

    checks.append(("Werner F<->w round trip", near(werner_fidelity(werner_w(0.87)), 0.87)))
    checks.append(("F=1 gives w=1", near(werner_w(1.0), 1.0)))
    checks.append(("F=1/4 (max mixed) gives w=0", near(werner_w(0.25), 0.0)))
    checks.append(("SKF(w=1) = 1", near(secret_key_fraction(1.0), 1.0)))
    checks.append(("SKF(w=0) = 0", near(secret_key_fraction(0.0), 0.0)))

    checks.append(("h(Q*) = 1/2", near(binary_entropy(Q_STAR), 0.5, 1e-7)))
    w_thresh = 1.0 - 2.0 * Q_STAR
    checks.append(("SKF vanishes exactly at Q*",
                   float(secret_key_fraction(w_thresh * 1.0001)) > 0.0
                   and near(secret_key_fraction(w_thresh * 0.9999), 0.0)))

    checks.append(("p = 10^(-aL/10): 50 km at 0.2 dB/km gives 0.1",
                   near(link_success_probability(50.0, ALPHA_DB_PER_KM), 0.1, 1e-12)))

    p_test, n_test = 0.2, 5
    X = rng.geometric(p_test, size=(400_000, n_test))
    checks.append(("MC E[max Geom] matches exact series",
                   abs(X.max(axis=1).mean() - expected_max_geometric(p_test, n_test)) < 0.02))

    for F in (0.90, 0.93, 0.95, 0.96, 0.97, 0.99):
        h = distrust_horizon(20.0, F, NO_MEMORY_NOISE_S, n_max=60, trials=4_000, rng=rng)
        checks.append((f"horizon = floor(N*) at F={F}  (N*={n_star(F):.2f})",
                       h == math.floor(n_star(F))))

    ch1 = Chain(n_edges=1, edge_km=20.0, fidelity=0.95, coherence_s=1.0)
    tn1 = tn_mode_skr(ch1)
    ent1 = ent_mode_skr(ch1, 500_000, rng)[0]
    checks.append(("N=1: ENT converges to TN (rel. < 1%)", abs(ent1 - tn1) / tn1 < 0.01))

    ok_plob = True
    for L in (10.0, 20.0, 50.0, 100.0):
        eta = float(link_success_probability(L))
        per_use = float(link_success_probability(L)) * float(secret_key_fraction(werner_w(0.99)))
        if per_use > -math.log2(1.0 - eta) + 1e-12:
            ok_plob = False
    checks.append(("TN edge rate respects the PLOB bound", ok_plob))

    prices = [price_of_distrust(Chain(n_edges=n, edge_km=20.0, fidelity=0.97,
                                      coherence_s=10.0), 40_000, rng)
              for n in range(1, 9)]
    checks.append(("price decays monotonically in N",
                   all(b <= a + 1e-3 for a, b in zip(prices, prices[1:]))))

    warm = distrust_horizon(20.0, 0.97, 0.05, n_max=40, trials=4_000, rng=rng)
    cold = distrust_horizon(20.0, 0.97, NO_MEMORY_NOISE_S, n_max=40, trials=4_000, rng=rng)
    checks.append(("finite coherence never exceeds infinite", warm <= cold))

    if verbose:
        print("VALIDATION GATE")
        for name, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print()
    return all(ok for _, ok in checks)


def sweep_horizon() -> None:
    print("=" * 78)
    print("TABLE 1. Distrust horizon vs link fidelity and coherence time")
    print("edge 20 km, alpha 0.2 dB/km")
    print("=" * 78)
    coherences = [0.05, 0.2, 1.0, 4.0, 10.0, NO_MEMORY_NOISE_S]
    labels = ["50 ms", "200 ms", "1 s", "4 s", "10 s", "no noise"]
    print("\n  F_link |" + "".join(f"{lab:>10}" for lab in labels) + "   |  N* (cap)")
    print("  " + "-" * 76)
    for F in (0.90, 0.93, 0.95, 0.96, 0.97, 0.99):
        cells = ""
        for T in coherences:
            cells += f"{distrust_horizon(20.0, F, T, n_max=40, trials=40_000):>10}"
        print(f"   {F:.2f}  |{cells}   |  {n_star(F):>6.2f}")


def sweep_price() -> None:
    print("\n" + "=" * 78)
    print("TABLE 2. Price of distrust, F=0.96, T=10 s, edge 20 km")
    print("price = SKR_ENT / SKR_TN = fidelity factor x rate factor")
    print("=" * 78)
    print("\n    N |     price |  fidelity factor |  rate factor |   ENT SKR (Hz)")
    print("  " + "-" * 70)
    for n in (1, 2, 3, 4, 5, 6, 7, 8):
        ch = Chain(n_edges=n, edge_km=20.0, fidelity=0.96, coherence_s=10.0)
        price, fid, rate = decompose(ch, trials=200_000)
        skr = ent_mode_skr(ch, 200_000)[0]
        shown = f"{price * 100:8.2f}%" if price > 1e-6 else "      --"
        print(f"   {n:>2} | {shown} | {fid * 100:15.2f}% | {rate * 100:10.2f}% | {skr:>12.1f}")
    print(f"\n  Trusted-relay reference rate, any N: "
          f"{tn_mode_skr(Chain(1, 20.0, 0.96, 10.0)):.1f} Hz")


def sweep_decomposition() -> None:
    print("\n" + "=" * 78)
    print("TABLE 3. Rate factor vs fidelity factor, same hardware as Table 2")
    print("=" * 78)
    for n in (2, 4, 6, 8, 10):
        ch = Chain(n_edges=n, edge_km=20.0, fidelity=0.96, coherence_s=10.0)
        _, fid, rate = decompose(ch, trials=200_000)
        ratio = f"{rate / fid:.1f}x" if fid > 0.0 else "inf"
        print(f"   N={n:>2}:  rate {rate:6.3f}   fidelity {fid:6.3f}   "
              f"fidelity smaller by {ratio:>5}")


def sweep_edge_length() -> None:
    print("\n" + "=" * 78)
    print("TABLE 4. Distrust horizon vs edge length, F=0.97")
    print("=" * 78)
    coherences = [0.05, 0.2, 1.0, 10.0, NO_MEMORY_NOISE_S]
    labels = ["50 ms", "200 ms", "1 s", "10 s", "no noise"]
    print("\n  edge km |  1/p (rounds) |  mean wait |" + "".join(f"{lab:>10}" for lab in labels))
    print("  " + "-" * 76)
    for L in (10.0, 20.0, 50.0, 75.0, 100.0):
        p = float(link_success_probability(L))
        t_att = L / 200_000.0
        ch = Chain(n_edges=4, edge_km=L, fidelity=0.97, coherence_s=1.0)
        mean_cycle = ent_mode_skr(ch, 40_000)[3]
        cells = ""
        for T in coherences:
            cells += f"{distrust_horizon(L, 0.97, T, n_max=20, trials=40_000):>10}"
        print(f"  {L:>7.0f} | {1 / p:>13.1f} | {mean_cycle * 1e3:>8.2f} ms |{cells}")


if __name__ == "__main__":
    ok = validate()
    sweep_horizon()
    sweep_price()
    sweep_decomposition()
    sweep_edge_length()
    print("\n" + ("validation: all checks passed" if ok else "validation: FAILURES PRESENT"))
