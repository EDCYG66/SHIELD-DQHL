"""Utilities for exact MPR realization in mixed CAV/HV experiments."""

from __future__ import annotations

from fractions import Fraction
from math import ceil, gcd
from typing import Iterable, Sequence, Tuple


def _lcm(a: int, b: int) -> int:
    a = max(1, int(a))
    b = max(1, int(b))
    return abs(a * b) // gcd(a, b)


def mpr_fraction(value: float, *, max_denominator: int = 1000) -> Fraction:
    clipped = min(1.0, max(0.0, float(value)))
    return Fraction(str(clipped)).limit_denominator(max_denominator)


def exact_cav_count(total_vehicles: int, mpr_cav: float, *, max_denominator: int = 1000) -> int:
    total_vehicles = max(0, int(total_vehicles))
    frac = mpr_fraction(mpr_cav, max_denominator=max_denominator)
    if total_vehicles <= 0:
        return 0
    if total_vehicles % frac.denominator == 0:
        return int(frac.numerator * (total_vehicles // frac.denominator))
    return int(round(float(frac) * float(total_vehicles)))


def exact_total_for_mprs(
    base_total: int,
    mpr_values: Sequence[float] | Iterable[float],
    *,
    require_even: bool = True,
    max_denominator: int = 1000,
) -> int:
    base_total = max(2, int(base_total))
    exact_total = 1
    values = list(mpr_values)
    if not values:
        values = [0.5]
    for value in values:
        exact_total = _lcm(exact_total, mpr_fraction(value, max_denominator=max_denominator).denominator)
    if require_even and exact_total % 2 != 0:
        exact_total *= 2
    multiplier = max(1, int(ceil(float(base_total) / float(exact_total))))
    exact_total *= multiplier
    if require_even and exact_total % 2 != 0:
        exact_total *= 2
    return int(exact_total)


def resolve_exact_vehicle_counts(
    n_up: int,
    n_down: int,
    mpr_values: Sequence[float] | Iterable[float],
    *,
    max_denominator: int = 1000,
) -> Tuple[int, int, int]:
    n_up = max(1, int(n_up))
    n_down = max(1, int(n_down))
    base_total = n_up + n_down
    exact_total = exact_total_for_mprs(base_total, mpr_values, require_even=True, max_denominator=max_denominator)
    if n_up == n_down:
        return exact_total // 2, exact_total // 2, exact_total

    ratio_up = float(n_up) / float(max(1, base_total))
    resolved_up = max(1, int(round(ratio_up * float(exact_total))))
    resolved_up = min(exact_total - 1, resolved_up)
    resolved_down = max(1, exact_total - resolved_up)
    return int(resolved_up), int(resolved_down), int(exact_total)
