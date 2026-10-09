"""Generate the synthetic dataset, train the RF + DT ensemble and save it to models/.

Run:  python scripts/train.py [--per-disease 150] [--seed 42]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurosy.model import MODEL_PATH, train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-disease", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    clf = train(per_disease=a.per_disease, seed=a.seed)
    print(f"Saved {MODEL_PATH}")
