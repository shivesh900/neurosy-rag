"""Synthetic patient generator.

Each synthetic patient is sampled from a disease profile in the knowledge base:
every symptom is included with its prevalence, the disease's required symptoms
are guaranteed, and some noise symptoms are added. Seeded and reproducible.

Synthetic data is used because no openly licensed, clinically validated
symptom-to-diagnosis dataset fits this project. Accuracy numbers on it measure
how well the system recovers the knowledge-base profiles under noise and
missing symptoms. They are NOT clinical accuracy.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .knowledge import KnowledgeBase, load_kb


def sample_patient(kb: KnowledgeBase, disease_key: str, rng: np.random.Generator,
                   noise_p: float = 0.35, keep: Optional[int] = None) -> List[str]:
    d = kb.diseases[disease_key]
    for _ in range(50):
        chosen = {s for s, p in d.symptoms.items() if rng.random() < p}
        expanded = kb.expand(chosen)
        if all(any(s in expanded for s in g) for g in d.requires) and len(chosen) >= 2:
            break
    else:  # force the requirements
        for g in d.requires:
            cands = [s for s in d.symptoms if s in g or set(g) & kb.expand([s])]
            chosen.add(rng.choice(cands or g))
    chosen = sorted(chosen)
    if keep is not None and len(chosen) > keep:
        # incomplete reporting: the patient mentions the hallmark finding(s) plus a few others
        must = []
        for g in d.requires:
            opts = [s for s in chosen if s in g or set(g) & kb.expand([s])]
            if opts and not set(must) & set(opts):
                must.append(str(rng.choice(opts)))
        rest = [s for s in chosen if s not in must]
        extra = max(0, keep - len(must))
        chosen = must + (list(rng.choice(rest, size=min(extra, len(rest)), replace=False)) if rest else [])
    if rng.random() < noise_p:  # an unrelated symptom (comorbidity / misreport)
        others = [s for s in kb.symptom_keys if s not in d.symptoms]
        chosen.append(str(rng.choice(others)))
    return sorted(kb.expand(chosen))


def generate(kb: Optional[KnowledgeBase] = None, per_disease: int = 150, seed: int = 42,
             noise_p: float = 0.35, keep: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    """Return a one-hot DataFrame (one column per symptom) plus a ``disease`` column."""
    kb = kb or load_kb()
    rng = np.random.default_rng(seed)
    rows = []
    for key in sorted(kb.diseases):
        for _ in range(per_disease):
            k = int(rng.integers(keep[0], keep[1] + 1)) if keep else None
            syms = set(sample_patient(kb, key, rng, noise_p=noise_p, keep=k))
            row = {s: int(s in syms) for s in kb.symptom_keys}
            row["disease"] = key
            rows.append(row)
    return pd.DataFrame(rows)
