"""End-to-end: text -> symptoms -> ML ensemble -> symbolic validation -> RAG explanation -> care."""
from __future__ import annotations

import time
from typing import Optional

from . import DISCLAIMER
from .geo import find_care
from .knowledge import KnowledgeBase, load_kb
from .model import DiseaseClassifier, load_or_train
from .nlp import SymptomExtractor
from .rag import Explainer
from .symbolic import NeuroSymbolicReasoner


class NeuroSyRAG:
    def __init__(self, kb: Optional[KnowledgeBase] = None, classifier: Optional[DiseaseClassifier] = None):
        self.kb = kb or load_kb()
        self.extractor = SymptomExtractor(self.kb)
        self.classifier = classifier or load_or_train()
        self.reasoner = NeuroSymbolicReasoner(self.kb)
        self.explainer = Explainer(self.kb)

    def diagnose(self, text: str, place: Optional[str] = None, lat: Optional[float] = None,
                 lon: Optional[float] = None, top_k: int = 3, find_providers: bool = True,
                 offline: bool = False) -> dict:
        t0 = time.perf_counter()
        ex = self.extractor.extract(text)
        ml = self.classifier.predict_proba(ex.present)
        res = self.reasoner.reason(ml, ex.present, ex.reported, ex.negated, top_k=top_k)
        top = None if res["abstain"] else res["candidates"][0]
        triage = self.reasoner.triage(ex.present, top, ex.duration_days)

        out = {
            "input": text,
            "symptoms": ex.to_dict(self.kb),
            "abstained": res["abstain"],
            "abstain_reason": res["abstain_reason"],
            "predictions": [c.to_dict() for c in res["candidates"]],
            "rejected_by_rules": [c.to_dict() for c in res["rejected"]],
            "ml_only_top": self.kb.diseases[res["ml_top"]].name if res["ml_top"] else None,
            "triage": {"level": triage.level, "reasons": triage.reasons},
            "explanation": None,
            "care": None,
            "disclaimer": DISCLAIMER,
        }
        if top is not None:
            out["explanation"] = self.explainer.explain(top.disease, sorted(ex.reported), text)
        if find_providers and (place or lat is not None or offline):
            d = self.kb.diseases[top.disease] if top else None
            out["care"] = find_care(place, lat, lon,
                                    specialty=d.specialty if d else "General Medicine",
                                    risk_level=d.risk_level if d else "medium", offline=offline)
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out


_engine: Optional[NeuroSyRAG] = None


def get_engine() -> NeuroSyRAG:
    global _engine
    if _engine is None:
        _engine = NeuroSyRAG()
    return _engine
