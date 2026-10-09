"""Evaluate NeuroSy-RAG and print the numbers quoted in the README.

1. Held-out split of the synthetic training distribution (ML ensemble only).
2. "Hard" patients: only 2-3 of their symptoms reported plus noise, from a
   different random seed. Compares ML-only with neuro-symbolic.
3. Out-of-distribution inputs: random unrelated symptoms. How often does the
   system abstain instead of inventing a diagnosis?
4. End-to-end free-text vignettes (data/vignettes.json, 34 hand-written
   descriptions, one or more per disease) through NLP, ML, rules and fusion.

Run:  python scripts/evaluate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neurosy.data import sample_patient  # noqa: E402
from neurosy.knowledge import load_kb  # noqa: E402
from neurosy.model import train  # noqa: E402
from neurosy.pipeline import NeuroSyRAG  # noqa: E402


def topk(ranked, truth, k):
    return truth in ranked[:k]


def main():
    kb = load_kb()
    clf = train(kb, verbose=False, save=True)
    print(f"1) Held-out synthetic split: ML ensemble accuracy {clf.meta['holdout_accuracy']:.3f} "
          f"({clf.meta['n_classes']} diseases, {clf.meta['n_samples']} synthetic patients, 80/20 stratified)")

    eng = NeuroSyRAG(kb, clf)
    rng = np.random.default_rng(2026)
    stats = {"ml1": 0, "ml3": 0, "ns1": 0, "ns3": 0, "ns_abstain": 0, "n": 0}
    for key in sorted(kb.diseases):
        for _ in range(40):
            syms = set(sample_patient(kb, key, rng, noise_p=0.5, keep=int(rng.integers(2, 4))))
            reported = {s for s in syms if not (kb.children(s) & syms)}
            ml = clf.predict_proba(syms)
            ml_rank = sorted(ml, key=ml.get, reverse=True)
            res = eng.reasoner.reason(ml, syms, reported, set(), top_k=3)
            ns_rank = [c.disease for c in res["candidates"]]
            stats["n"] += 1
            stats["ml1"] += topk(ml_rank, key, 1)
            stats["ml3"] += topk(ml_rank, key, 3)
            stats["ns1"] += (not res["abstain"]) and topk(ns_rank, key, 1)
            stats["ns3"] += topk(ns_rank, key, 3)
            stats["ns_abstain"] += res["abstain"]
    n = stats["n"]
    answered = n - stats["ns_abstain"]
    print(f"\n2) Hard synthetic patients (2-3 reported symptoms + 50% noise), n={n}")
    print(f"   ML only        : top-1 {stats['ml1'] / n:.3f}   top-3 {stats['ml3'] / n:.3f}")
    print(f"   Neuro-symbolic : top-1 {stats['ns1'] / n:.3f}   top-3 {stats['ns3'] / n:.3f}   "
          f"abstained {stats['ns_abstain'] / n:.1%}   accuracy when it answers {stats['ns1'] / max(answered, 1):.3f}")

    ood_abstain, ood_n, ml_conf = 0, 300, []
    for _ in range(ood_n):
        syms = set(rng.choice(kb.symptom_keys, size=3, replace=False))
        syms = kb.expand(syms)
        ml = clf.predict_proba(syms)
        ml_conf.append(max(ml.values()))
        res = eng.reasoner.reason(ml, syms, syms, set())
        ood_abstain += res["abstain"]
    print(f"\n3) Out-of-distribution (3 random unrelated symptoms), n={ood_n}")
    print(f"   ML alone still names a disease with mean top probability {np.mean(ml_conf):.2f}")
    print(f"   Neuro-symbolic abstains on {ood_abstain / ood_n:.1%} of them")

    run_vignettes(eng, "vignettes.json", "4) Free-text vignettes end to end (NLP, ML, rules; used during development)")
    run_vignettes(eng, "vignettes_holdout.json",
                  "5) Held-out free-text vignettes (written before tuning, never used to tune)")


def run_vignettes(eng, filename, title):
    vig = json.loads((ROOT / "data" / filename).read_text())
    ok1 = ok3 = abst = 0
    t0 = time.perf_counter()
    misses = []
    for v in vig:
        r = eng.diagnose(v["text"], find_providers=False)
        keys = [p["disease"] for p in r["predictions"]]
        hit1 = not r["abstained"] and keys[:1] == [v["expected"]]
        ok1 += hit1
        ok3 += v["expected"] in keys
        abst += r["abstained"]
        if not hit1:
            misses.append((v["expected"], keys[:1] or ["abstained"], v["text"]))
    dt = (time.perf_counter() - t0) / len(vig) * 1000
    print(f"\n{title}, n={len(vig)}")
    print(f"   top-1 {ok1}/{len(vig)} = {ok1 / len(vig):.1%}   top-3 {ok3}/{len(vig)} = {ok3 / len(vig):.1%}   "
          f"abstained {abst}   mean latency {dt:.0f} ms (CPU)")
    for exp, got, text in misses:
        print(f"   miss: expected {exp}, got {got[0]}: \"{text[:70]}\"")


if __name__ == "__main__":
    main()
