"""SO(5) irreps from the quadratic Casimir (normalisation C2(5)=2, C2(10)=3, C2(14)=5)."""
from math import ceil, sqrt


def types(c):
    """All B2 highest weights [a;b] (Dynkin) with Casimir c -> list of (a, b, dim)."""
    out = []
    for x in range(0, int(ceil(2 * sqrt(2 * max(c, 0)))) + 3):
        l1 = x / 2
        if l1 * (l1 + 3) / 2 > c + 1e-8:
            continue
        for y in range(x % 2, x + 1, 2):
            l2 = y / 2
            if abs((l1 * (l1 + 3) + l2 * (l2 + 1)) / 2 - c) > 1e-8:
                continue
            a, b = (x - y) // 2, y
            if b % 2:            # spinorial weights do not occur in this (fermion-bilinear) model
                continue
            dim = (a + 1) * (b + 1) * (a + b + 2) * (2 * a + b + 3) // 6
            out.append((a, b, dim))
    return out


def rep_name(c):
    """Own naming: dimension; the [0;4] 35 is '35p'. Ambiguous Casimirs -> 'C2=c{d1|d2}'."""
    t = types(c)
    if len(t) == 1:
        a, b, d = t[0]
        return "35p" if (a, b) == (0, 4) else str(d)
    if not t:
        return f"C2={c}?"
    return f"C2={c}{{" + "|".join(str(d) for _, _, d in t) + "}"
