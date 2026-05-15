from __future__ import annotations
import os
import subprocess
import sys

os.environ["PYTHONWARNINGS"] = "ignore"

SCRIPTS = [
    "data_from_sim.py",
    "train_linear.py",
    "train_kernel.py",
    "train_drop.py",
    "evaluate_policy.py",
    "generate_report.py",
]


def run(script):
    print(f"\n{'='*60}")
    print(f">>> {script}")
    print('='*60)
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    subprocess.run(
        [sys.executable, "-W", "ignore", script],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    for script in SCRIPTS:
        run(script)
    print("\n" + "="*60)
    print("ALL DONE.")
    print("  Plots  -> plots/extended_report/*.png")
    print("  Report -> plots/extended_report/results_report.html")
    print("  Tables -> plots/extended_report/table_*.csv")
    print("="*60)
