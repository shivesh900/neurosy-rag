"""The neural/statistical half: a soft-voting ensemble of Random Forest + Decision Tree."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .data import generate
from .knowledge import KnowledgeBase, load_kb

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "ensemble.joblib"


def build_ensemble(seed: int = 42) -> VotingClassifier:
    # 100 trees with min_samples_leaf=3: ~1.5 points less held-out accuracy than 300 fully grown trees,
    # but 16x smaller (it ships to the browser demo) and smoother probabilities for the fusion step.
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    dt = DecisionTreeClassifier(max_depth=18, min_samples_leaf=2, class_weight="balanced", random_state=seed)
    return VotingClassifier([("rf", rf), ("dt", dt)], voting="soft", weights=[3, 1])


class DiseaseClassifier:
    def __init__(self, model: VotingClassifier, features: List[str], meta: Optional[dict] = None):
        self.model, self.features, self.meta = model, features, meta or {}
        self.classes: List[str] = list(model.classes_)
        for _, est in getattr(model, "named_estimators_", {}).items():
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1  # single-row predictions are faster without a thread pool

    def vectorize(self, symptoms: Iterable[str]) -> np.ndarray:
        s = set(symptoms)
        return np.array([[1 if f in s else 0 for f in self.features]])

    def predict_proba(self, symptoms: Iterable[str]) -> Dict[str, float]:
        probs = self.model.predict_proba(self.vectorize(symptoms))[0]
        return {c: float(p) for c, p in zip(self.classes, probs)}

    # persistence ---------------------------------------------------------------
    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features, "meta": self.meta}, path, compress=3)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DiseaseClassifier":
        blob = joblib.load(path)
        return cls(blob["model"], blob["features"], blob.get("meta"))


def train(kb: Optional[KnowledgeBase] = None, per_disease: int = 150, seed: int = 42,
          save: bool = True, verbose: bool = True) -> DiseaseClassifier:
    kb = kb or load_kb()
    df = generate(kb, per_disease=per_disease, seed=seed)
    X = df[kb.symptom_keys].to_numpy(dtype=np.int8)
    y = df["disease"].astype(str).to_numpy(dtype=object)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    model = build_ensemble(seed).fit(X_tr, y_tr)
    acc = float((model.predict(X_te) == y_te).mean())
    meta = {"n_samples": int(len(df)), "n_classes": int(len(set(y))), "n_features": len(kb.symptom_keys),
            "holdout_accuracy": round(acc, 4), "seed": seed, "per_disease": per_disease}
    clf = DiseaseClassifier(model, kb.symptom_keys, meta)
    if verbose:
        print(f"Trained RF+DT ensemble on {len(X_tr)} synthetic patients, "
              f"{meta['n_classes']} diseases; held-out accuracy {acc:.3f}")
    if save:
        clf.save()
    return clf


def load_or_train() -> DiseaseClassifier:
    if MODEL_PATH.exists():
        try:
            clf = DiseaseClassifier.load()
            if clf.features == load_kb().symptom_keys:
                return clf
        except Exception:
            pass
    return train(verbose=False)
