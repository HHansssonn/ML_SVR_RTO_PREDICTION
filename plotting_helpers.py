from typing import Optional

def percent_improvement(value: float, baseline: float, min_baseline: float = 1e-9) -> Optional[float]:
    if baseline <= min_baseline:
        return None
    return ((value - baseline) / baseline) * 100.0

def format_percent_label(value: float, baseline: float, min_baseline: float = 1e-9) -> str:
    pct = percent_improvement(value, baseline, min_baseline=min_baseline)
    return "n/a" if pct is None else f"{pct:+.1f}%"
