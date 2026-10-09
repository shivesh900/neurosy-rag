"""Loads the medical knowledge base (diseases, symptoms, synonyms, rules)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Set

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KB_PATH = DATA_DIR / "knowledge_base.json"


@dataclass
class Disease:
    key: str
    name: str
    icd10: str
    specialty: str
    risk_level: str
    symptoms: Dict[str, float]          # symptom -> approximate prevalence (0..1)
    requires: List[List[str]]           # every group needs at least one present symptom
    red_flags: List[str]
    description: str
    precautions: List[str]
    see_doctor: str


class KnowledgeBase:
    def __init__(self, raw: dict):
        self.symptoms: Dict[str, dict] = raw["symptoms"]
        self.diseases: Dict[str, Disease] = {k: Disease(key=k, **v) for k, v in raw["diseases"].items()}
        self.symptom_keys: List[str] = sorted(self.symptoms)
        self._children: Dict[str, Set[str]] = {}
        for child, info in self.symptoms.items():
            for parent in info.get("implies", []):
                self._children.setdefault(parent, set()).add(child)

    # symptom helpers -------------------------------------------------------
    def label(self, symptom: str) -> str:
        return self.symptoms.get(symptom, {}).get("label", symptom.replace("_", " "))

    def expand(self, symptoms: Iterable[str]) -> Set[str]:
        """Add parent symptoms implied by specific ones (high_fever -> fever)."""
        out = set(symptoms)
        stack = list(out)
        while stack:
            for parent in self.symptoms.get(stack.pop(), {}).get("implies", []):
                if parent not in out:
                    out.add(parent)
                    stack.append(parent)
        return out

    def children(self, symptom: str) -> Set[str]:
        return self._children.get(symptom, set())

    def disease_names(self) -> Dict[str, str]:
        return {k: d.name for k, d in self.diseases.items()}


@lru_cache(maxsize=1)
def load_kb(path: str | None = None) -> KnowledgeBase:
    with open(path or KB_PATH, encoding="utf-8") as fh:
        return KnowledgeBase(json.load(fh))
