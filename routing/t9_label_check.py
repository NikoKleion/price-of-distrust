from itertools import chain, combinations

from isotonicity_check import Algebra, report

OWNERS = "BCD"
NON_ALLIED = {"C", "D"}


def subsets(s):
    return [frozenset(c) for c in chain.from_iterable(combinations(s, k) for k in range(len(s) + 1))]


def legal(lam):
    return len(lam & NON_ALLIED) <= 1


RATES = [1, 2, 3, 4]
WEIGHTS = [(r, lam) for r in RATES for lam in subsets(OWNERS)]
LABELS = [(r, o) for r in (2, 4) for o in OWNERS]


def extend(label, weight):
    (lr, lo), (wr, wlam) = label, weight
    return (min(lr, wr), wlam | {lo})


def flat_pref(a, b):
    la, lb = legal(a[1]), legal(b[1])
    if la != lb:
        return la
    if not la:
        return True
    return a[0] >= b[0]


def dominance_pref(a, b):
    return a[0] >= b[0] and a[1] <= b[1]


flat = Algebra(
    name="T9 flat: (bottleneck rate, owner set) with legality floor, total preorder",
    weights=WEIGHTS, labels=LABELS, extend=extend, at_least_as_good=flat_pref,
)
dominance = Algebra(
    name="T9 dominance: same algebra under (rate >=, owner-set subset) partial order",
    weights=WEIGHTS, labels=LABELS, extend=extend, at_least_as_good=dominance_pref,
)

if __name__ == "__main__":
    print("T9 groundwork: label-augmented TN-mode routing (WEFT_IDEA section 5)")
    report(flat)
    report(dominance)
    
