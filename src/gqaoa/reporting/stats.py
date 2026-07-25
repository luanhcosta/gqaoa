from typing import Sequence

import numpy as np


def report_stats(values: Sequence[float], label: str) -> dict:
    """Compute and print percentile statistics for a batch of run results.

    Replaces the min/p10/p25/median/p75/p90/max/mean/std block that used to
    be copy-pasted across stability_check.py, stability_bracket.py and
    benchmark_gd.py.
    """
    values = np.asarray(values, dtype=float)
    stats = {
        "min": values.min(),
        "p10": np.percentile(values, 10),
        "p25": np.percentile(values, 25),
        "median": np.median(values),
        "p75": np.percentile(values, 75),
        "p90": np.percentile(values, 90),
        "max": values.max(),
        "mean": values.mean(),
        "std": values.std(),
    }

    print(f"\n{'='*45}")
    print(f"{label} — {len(values)} runs")
    print(f"{'='*45}")
    for k, v in stats.items():
        print(f"  {k:8s}: {v:+.6f}")
    print(f"  values  : {np.round(values, 6).tolist()}")

    return stats
