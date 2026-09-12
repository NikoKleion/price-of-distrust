from dataclasses import dataclass
from itertools import product
from typing import Callable, Sequence


@dataclass(frozen=True)
class Algebra:
    name: str
    weights: Sequence
    labels: Sequence
    extend: Callable
    at_least_as_good: Callable


def check_monotone(alg: Algebra):
    return [
        (l, w, alg.extend(l, w))
        for l, w in product(alg.labels, alg.weights)
        if not alg.at_least_as_good(w, alg.extend(l, w))
    ]


def check_isotone(alg: Algebra):
    out = []
    for w1, w2 in product(alg.weights, alg.weights):
        if not alg.at_least_as_good(w1, w2):
            continue
        for l in alg.labels:
            e1, e2 = alg.extend(l, w1), alg.extend(l, w2)
            if not alg.at_least_as_good(e1, e2):
                out.append((l, w1, w2, e1, e2))
    return out


INF = float("inf")

shortest_path = Algebra(
    name="shortest-path (additive cost, smaller better)",
    weights=list(range(0, 13)),
    labels=[1, 2, 3],
    extend=lambda l, w: w + l,
    at_least_as_good=lambda a, b: a <= b,
)

widest_path = Algebra(
    name="widest-path (bottleneck capacity, larger better)",
    weights=[1, 2, 3, 4, 5, INF],
    labels=[1, 2, 3, 4, 5],
    extend=lambda l, w: min(l, w),
    at_least_as_good=lambda a, b: a >= b,
)


F_MIN = 0.75

def swap_fidelity(f1: float, f2: float) -> float:
    return f1 * f2 + (1.0 - f1) * (1.0 - f2) / 3.0

def ent_extend(label, weight):
    (lr, lf), (wr, wf) = label, weight
    return (min(lr, wr), swap_fidelity(lf, wf))

def ent_at_least_as_good(a, b):
    (ar, af), (br, bf) = a, b
    a_ok, b_ok = af >= F_MIN, bf >= F_MIN
    if a_ok != b_ok:
        return a_ok
    if not a_ok:
        return True
    return (ar, af) >= (br, bf)

RATES = [1, 2, 3, 4, 5]
FIDS = [0.78, 0.85, 0.92, 0.97, 1.0]

ent_fidelity_threshold = Algebra(
    name=f"ENT-mode (bottleneck rate, swap fidelity, floor F_min={F_MIN})",
    weights=[(r, f) for r in RATES for f in FIDS],
    labels=[(r, f) for r in [2, 5] for f in [0.85, 0.92, 1.0]],
    extend=ent_extend,
    at_least_as_good=ent_at_least_as_good,
)


def report(alg: Algebra) -> None:
    mono = check_monotone(alg)
    iso = check_isotone(alg)
    print(f"\n{alg.name}")
    print(f"  |S| = {len(alg.weights)}, |L| = {len(alg.labels)}")
    print(f"  monotone: {'YES' if not mono else f'NO ({len(mono)} counterexamples)'}")
    print(f"  isotone:  {'YES' if not iso else f'NO ({len(iso)} counterexamples)'}")
    if mono:
        l, w, e = mono[0]
        print(f"    e.g. extend({l}, {w}) = {e} improved on {w}")
    if iso:
        l, w1, w2, e1, e2 = iso[0]
        print(f"    e.g. {w1} >= {w2}, but extend by {l} flips the order:")
        print(f"         extend({l}, {w1}) = {e1}  <  extend({l}, {w2}) = {e2}")


if __name__ == "__main__":
    print("Routing-algebra property check (Sobrinho 2005 conditions, T1)")
    for alg in (shortest_path, widest_path, ent_fidelity_threshold):
        report(alg)
    
