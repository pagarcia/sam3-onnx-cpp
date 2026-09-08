"""Fail-closed parity gates without model/runtime dependencies."""
import math
from numbers import Real


def _finite_number(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def check_thresholds(summary: dict, thresholds: dict) -> tuple[bool, list[dict]]:
    failures = []
    if (not isinstance(summary, dict) or not isinstance(thresholds, dict)
        or set(thresholds) - {"min", "max"}
        or not all(isinstance(v, dict) for v in thresholds.values())
        or not any(thresholds.values())):
        return False, [{"kind": "invalid-gates", "message": "Explicit nonempty min/max thresholds are required."}]
    for kind in ("min", "max"):
        for metric, bound in thresholds.get(kind, {}).items():
            actual = summary.get(metric)
            if not _finite_number(bound):
                failures.append({"metric": metric, "kind": "invalid-threshold"})
            elif metric not in summary:
                failures.append({"metric": metric, "kind": "missing-metric"})
            elif not _finite_number(actual):
                failures.append({"metric": metric, "kind": "invalid-metric"})
            elif (kind == "min" and actual < bound) or (kind == "max" and actual > bound):
                failures.append({"metric": metric, "kind": kind,
                                 "expected": float(bound), "actual": float(actual)})
    return not failures, failures
