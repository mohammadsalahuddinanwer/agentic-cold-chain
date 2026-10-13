#!/usr/bin/env python3
"""
cleanup_repo.py
===============
Removes files left over from development that should not ship with the
submitted codebase, and repairs the one import that depends on them.

"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Files to delete, with the reason each one goes ───────────────────────
FILES = [
    ("0.46.1",
     "stray pip install log, saved as a filename by accident"),
    ("McNemar.py",
     "superseded by verify_paper_numbers.py, which runs the exact binomial "
     "test the paper actually reports"),
    ("test_monitor.py",
     "development scratch script, not part of the evaluation pipeline"),
    ("evaluation/ablation_study.py",
     "dead code, superseded by the top-level run_ablation_study.py"),
    ("data/knowledge_base/anomaly_cases _bkp.json",
     "backup of the knowledge base (note the space in the filename)"),
    ("outputs/a2_seconds_backup.csv",
     "safety copy of a2_summary.csv taken while testing the minutes variant"),
]

# ── Directories to delete ────────────────────────────────────────────────
DIRS = [
    ("data/synthetic_bkp",
     "320 MB duplicate of data/synthetic/"),
]

# ── The import line that must go with evaluation/ablation_study.py ───────
INIT_PATH = Path("evaluation/__init__.py")
INIT_EDITS = [
    ("from .ablation_study import AblationStudy\n", ""),
    ('__all__ = ["ColdChainSimulator", "MetricsCalculator", "ExperimentRunner", "AblationStudy"]',
     '__all__ = ["ColdChainSimulator", "MetricsCalculator", "ExperimentRunner"]'),
    ('"""Evaluation framework: metrics, simulator, experiments, ablation."""',
     '"""Evaluation framework: metrics, simulator, experiments."""'),
]


def human(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def dir_size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be removed without removing it")
    args = ap.parse_args()

    freed = 0
    removed = 0
    missing = 0

    print("=" * 74)
    print("REPO CLEANUP" + ("  (dry run — nothing will be changed)" if args.dry_run else ""))
    print("=" * 74)

    for rel, reason in FILES:
        p = ROOT / rel
        if not p.exists():
            print(f"  --  already gone   {rel}")
            missing += 1
            continue
        size = p.stat().st_size
        print(f"  rm  {human(size):>8}   {rel}")
        print(f"                       {reason}")
        if not args.dry_run:
            p.unlink()
        freed += size
        removed += 1

    for rel, reason in DIRS:
        p = ROOT / rel
        if not p.exists():
            print(f"  --  already gone   {rel}/")
            missing += 1
            continue
        size = dir_size(p)
        print(f"  rm  {human(size):>8}   {rel}/")
        print(f"                       {reason}")
        if not args.dry_run:
            shutil.rmtree(p)
        freed += size
        removed += 1

    # Empty __pycache__ directories
    caches = [d for d in ROOT.rglob("__pycache__") if d.is_dir()]
    if caches:
        print(f"  rm               {len(caches)} __pycache__ director"
              f"{'y' if len(caches) == 1 else 'ies'} (now gitignored)")
        if not args.dry_run:
            for d in caches:
                shutil.rmtree(d, ignore_errors=True)

    # ── Repair evaluation/__init__.py ────────────────────────────────────
    init = ROOT / INIT_PATH
    if init.exists():
        src = init.read_text(encoding="utf-8")
        new = src
        for old, repl in INIT_EDITS:
            new = new.replace(old, repl)
        if new != src:
            print(f"\n  edit             {INIT_PATH}")
            print("                       drop the AblationStudy import, which "
                  "would otherwise break")
            print("                       evaluate_resolver.py once the module "
                  "is deleted")
            if not args.dry_run:
                init.write_text(new, encoding="utf-8")
        else:
            print(f"\n  --  no change    {INIT_PATH} (already repaired)")

    print("-" * 74)
    verb = "would free" if args.dry_run else "freed"
    print(f"{removed} item(s) removed, {missing} already gone, {verb} {human(freed)}")

    if args.dry_run:
        print("\nRe-run without --dry-run to apply.")
    else:
        print("\nNow confirm nothing broke:")
        print("    python -c \"from evaluation import ColdChainSimulator; print('ok')\"")
        print("    python verify_paper_numbers.py")
        print("Expected: 'ok', then 72 passed, 0 FAILED.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
