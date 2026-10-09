"""The symbolic half: a disease-symptom knowledge graph plus deterministic clinical rules.

The ML ensemble proposes diagnoses. This layer checks each one against
the knowledge base and explains its decisions:

* R1 necessary findings: a disease is rejected if a required symptom group is
  absent (e.g. dengue requires fever).
* R2 contradiction: a disease is rejected if the patient explicitly denied all
  of a required group ("no fever"). A denied hallmark symptom costs confidence.
* R3 knowledge-graph coverage: how much of the disease profile is observed, and
  how many of the patient's symptoms it explains.
* R4 abstention: if no candidate is supported well enough, the system says
  "insufficient evidence" instead of guessing. This mitigates hallucinated
  diagnoses.
* R5 triage red flags: emergency symptom patterns raise the urgency no matter
  what the model predicts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set

from .knowledge import KnowledgeBase, load_kb

URGENCY_ORDER = ["self-care", "see a doctor", "urgent", "emergency"]


@dataclass
class Candidate:
    disease: str
    name: str
    ml_prob: float
    kg_support: float = 0.0
    kg_explains: float = 0.0
    kg_score: float = 0.0
    final: float = 0.0
    status: str = "validated"          # validated | adjusted | rejected
    reasons: List[str] = field(default_factory=list)
    matched: List[str] = field(default_factory=list)
    missing_hallmarks: List[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        for k in ("ml_prob", "kg_support", "kg_explains", "kg_score", "final"):
            d[k] = round(d[k], 4)
        return d


@dataclass
class Triage:
    level: str
    reasons: List[str]


class NeuroSymbolicReasoner:
    def __init__(self, kb: Optional[KnowledgeBase] = None, ml_weight: float = 0.55,
                 abstain_below: float = 0.30):
        self.kb = kb or load_kb()
        self.ml_weight = ml_weight
        self.abstain_below = abstain_below

    # knowledge graph ------------------------------------------------------------
    def coverage(self, disease_key: str, present: Set[str], reported: Set[str]):
        d = self.kb.diseases[disease_key]
        total = sum(d.symptoms.values())
        got, matched = 0.0, []
        for s, w in d.symptoms.items():
            if s in present:
                got += w
                matched.append(s)
            elif (self.kb.expand([s]) - {s}) & present:      # e.g. disease wants high_fever, patient has fever
                got += 0.6 * w
                matched.append(s)
        support = got / total if total else 0.0
        profile = self.kb.expand(d.symptoms)
        explained = [s for s in reported if s in profile or self.kb.children(s) & set(d.symptoms)]
        explains = len(explained) / len(reported) if reported else 0.0
        hallmarks = [s for s, w in d.symptoms.items() if w >= 0.85 and s not in present
                     and not ((self.kb.expand([s]) - {s}) & present)]
        return support, explains, matched, hallmarks

    # rules ---------------------------------------------------------------------
    def check(self, cand: Candidate, present: Set[str], reported: Set[str], negated: Set[str]) -> Candidate:
        d = self.kb.diseases[cand.disease]
        lab = self.kb.label
        for group in d.requires:
            if not any(s in present for s in group):
                denied = [s for s in group if s in negated]
                names = " or ".join(lab(s) for s in group)
                if denied and len(denied) == len(group):
                    cand.reasons.append(f"R2 contradiction: {d.name} needs {names}, which you said you don't have.")
                else:
                    cand.reasons.append(f"R1 necessary finding missing: {d.name} needs {names}.")
                cand.status = "rejected"
        support, explains, matched, hallmarks = self.coverage(cand.disease, present, reported)
        cand.kg_support, cand.kg_explains = support, explains
        cand.kg_score = 0.6 * support + 0.4 * explains
        cand.matched = [lab(s) for s in matched]
        cand.missing_hallmarks = [lab(s) for s in hallmarks]

        denied_hallmarks = [s for s, w in d.symptoms.items() if w >= 0.85 and s in negated]
        penalty = 0.7 ** len(denied_hallmarks)
        if denied_hallmarks:
            cand.reasons.append("R2 denied hallmark symptom: " + ", ".join(lab(s) for s in denied_hallmarks) + ".")

        if cand.status == "rejected":
            cand.final = 0.0
            return cand
        cand.final = (self.ml_weight * cand.ml_prob + (1 - self.ml_weight) * cand.kg_score) * penalty
        if cand.kg_score < 0.35:
            cand.status = "adjusted"
            cand.reasons.append(f"R3 weak knowledge-graph support ({cand.kg_score:.2f}): "
                                "few of this disease's typical symptoms were reported.")
        else:
            cand.reasons.append(f"R3 knowledge graph supports it: {len(matched)} typical symptoms matched "
                                f"(coverage {support:.0%}, explains {explains:.0%} of your symptoms).")
        return cand

    def reason(self, ml_probs: Dict[str, float], present: Set[str], reported: Set[str],
               negated: Set[str], top_k: int = 3, pool: int = 8) -> dict:
        ml_rank = sorted(ml_probs, key=ml_probs.get, reverse=True)
        # candidates: the model's top `pool` plus any disease the KG strongly supports
        pool_keys = list(dict.fromkeys(ml_rank[:pool] + [
            k for k in self.kb.diseases if self.coverage(k, present, reported)[0] >= 0.5]))
        cands = [self.check(Candidate(k, self.kb.diseases[k].name, ml_probs.get(k, 0.0)), present, reported, negated)
                 for k in pool_keys]
        cands.sort(key=lambda c: c.final, reverse=True)
        accepted = [c for c in cands if c.status != "rejected"]
        rejected = [c for c in cands if c.status == "rejected" and c.ml_prob >= 0.05]
        if accepted and ml_rank and accepted[0].disease != ml_rank[0]:
            accepted[0].reasons.insert(0, f"Ranked first after symbolic validation "
                                          f"(the ML model alone preferred {self.kb.diseases[ml_rank[0]].name}).")
        abstain = not accepted or accepted[0].final < self.abstain_below or len(reported) < 2
        abstain_reason = None
        if not reported:
            abstain_reason = "No recognisable symptoms were found in the text."
        elif len(reported) < 2:
            abstain_reason = "Only one symptom was given. That's not enough to suggest a condition safely."
        elif not accepted:
            abstain_reason = "Every candidate the model proposed failed a clinical rule."
        elif accepted[0].final < self.abstain_below:
            abstain_reason = f"The best candidate's combined confidence ({accepted[0].final:.2f}) is below the {self.abstain_below} safety threshold."
        return {"candidates": accepted[:top_k], "rejected": rejected, "abstain": abstain,
                "abstain_reason": abstain_reason, "ml_top": ml_rank[0] if ml_rank else None}

    # triage --------------------------------------------------------------------
    def triage(self, present: Set[str], top: Optional[Candidate], duration_days: Optional[int]) -> Triage:
        p, reasons, level = present, [], "self-care"

        def bump(new, why):
            nonlocal level
            reasons.append(why)
            if URGENCY_ORDER.index(new) > URGENCY_ORDER.index(level):
                level = new

        if "confusion" in p:
            bump("emergency", "Confusion or disorientation is a red-flag symptom.")
        if "coughing_blood" in p:
            bump("emergency", "Coughing up blood needs immediate assessment.")
        if {"chest_pain", "shortness_of_breath"} <= p:
            bump("emergency", "Chest pain with breathlessness can be a heart or lung emergency.")
        if "nosebleed_or_gum_bleeding" in p and "fever" in p:
            bump("emergency", "Bleeding with fever can be a warning sign of severe dengue.")
        if "lower_right_abdominal_pain" in p and ({"fever", "vomiting"} & p):
            bump("emergency", "Lower-right abdominal pain with fever or vomiting can mean appendicitis.")
        if {"severe_headache", "neck_pain", "fever"} <= p:
            bump("emergency", "Severe headache with neck stiffness and fever can mean meningitis.")
        if "shortness_of_breath" in p:
            bump("urgent", "Breathlessness should be checked by a doctor soon.")
        if "yellow_skin" in p:
            bump("urgent", "Jaundice (yellow skin or eyes) needs liver tests.")
        if "high_fever" in p:
            bump("see a doctor", "High fever should be checked by a doctor within 24 hours.")
        if duration_days and duration_days >= 14:
            bump("see a doctor", f"Symptoms for {duration_days} days need a medical review.")
        if top is not None:
            risk = self.kb.diseases[top.disease].risk_level
            if risk == "high":
                bump("urgent", f"The leading candidate ({top.name}) is a high-risk condition.")
            elif risk == "medium":
                bump("see a doctor", f"{top.name} usually needs a doctor's diagnosis.")
        if not reasons:
            reasons.append("No red-flag symptoms were detected.")
        return Triage(level, reasons)
