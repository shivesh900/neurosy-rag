"""Symptom extraction from free text.

Turns "had high fever for 4 days, terrible headache, no rash" into structured
symptoms: present = {high_fever, fever, severe_headache, headache},
negated = {rash}, duration_days = 4.

Pipeline: normalise -> clause split -> synonym phrase matching (longest first)
-> fuzzy matching for typos -> negation scope detection -> duration rules ->
implication expansion. It is rule-based on purpose: transparent, fast, no GPU.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .knowledge import KnowledgeBase, load_kb

NEGATION_CUES = [
    "no", "not", "without", "never", "denies", "deny", "dont have", "don't have", "do not have",
    "didn't have", "did not have", "haven't had", "free of", "absence of", "negative for", "nor", "none",
]
SCOPE_BREAKERS = re.compile(r"\b(but|however|although|though|except|yet|just|only|and i have|and have|i have|i do have|there is)\b")
CLAUSE_SPLIT = re.compile(r"[.;!?\n]|,\s*(?=but\b)|\bbut\b")
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "a couple of": 2, "a few": 3, "several": 4, "a": 1, "an": 1}
DURATION_RE = re.compile(
    r"(?:for|since|past|last)\s+(?:the\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a couple of|a few|several|a|an)\s*"
    r"(day|days|week|weeks|month|months)", re.I)
UNIT_DAYS = {"day": 1, "week": 7, "month": 30}


@dataclass
class Mention:
    symptom: str
    text: str
    negated: bool
    fuzzy: bool = False


@dataclass
class Extraction:
    present: Set[str] = field(default_factory=set)       # expanded with implied parents
    reported: Set[str] = field(default_factory=set)      # what the user actually said (not expanded)
    negated: Set[str] = field(default_factory=set)
    mentions: List[Mention] = field(default_factory=list)
    duration_days: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self, kb: Optional[KnowledgeBase] = None) -> dict:
        lab = (kb.label if kb else (lambda s: s))
        return {
            "present": sorted(self.present), "reported": sorted(self.reported), "negated": sorted(self.negated),
            "present_labels": [lab(s) for s in sorted(self.reported)],
            "negated_labels": [lab(s) for s in sorted(self.negated)],
            "duration_days": self.duration_days, "notes": self.notes,
            "mentions": [m.__dict__ for m in self.mentions],
        }


# "my eyes have turned yellow" -> "my yellow eyes"; "urine is dark" -> "dark urine"
BODY = r"(eyes?|skin|urine|joints?|toes?|knees?|wrists?|fingers?|throat|nose|head|chest|stomach|neck|heart)"
STATE = r"(yellow|dark|red|swollen|painful|sore|stiff|blocked|itchy|tight|pounding|racing|hot|dry|scaly|aching)"
# "pain behind my eyes" should match "pain behind eyes"
DETERMINERS = re.compile(r"(?<=\s)(my|the|both|his|her|your|their|our)\s+")
INVERSION = re.compile(
    rf"\b{BODY}\s+(?:is|are|was|were|feels?|felt|looks?|has|have|got|gets|turned|became|keeps? getting)"
    rf"(?:\s+(?:turned|become|been|gone|getting|so|very|really|quite|all|a bit))*\s+{STATE}\b")


def _normalise(text: str) -> str:
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9'\s.,;!?/-]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = INVERSION.sub(lambda m: f"{m.group(0)}, {m.group(2)} {m.group(1)}", text)
    return re.sub(r" {2,}", " ", DETERMINERS.sub(" ", text))


class SymptomExtractor:
    def __init__(self, kb: Optional[KnowledgeBase] = None, fuzzy_cutoff: float = 0.86):
        self.kb = kb or load_kb()
        self.fuzzy_cutoff = fuzzy_cutoff
        phrases: List[Tuple[str, str]] = []
        for key, info in self.kb.symptoms.items():
            for syn in set(info["synonyms"] + [info["label"]]):
                phrases.append((re.sub(r" {2,}", " ", DETERMINERS.sub(" ", " " + syn.lower())).strip(), key))
        # longest first; ties broken alphabetically so the order is deterministic (and matches the JS port)
        phrases = sorted(set(phrases), key=lambda p: (-len(p[0]), p[0], p[1]))
        self.phrases = phrases
        self.patterns = [(re.compile(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])"), p, k) for p, k in phrases]
        self.fuzzy_vocab = {p: k for p, k in phrases if len(p) >= 6 and not p.isdigit()}

    @staticmethod
    def _negated(clause: str, start: int) -> bool:
        before = clause[:start]
        # negation scope ends at a contrastive word ("no rash but fever")
        last_break = None
        for m in SCOPE_BREAKERS.finditer(before):
            last_break = m
        if last_break:
            before = before[last_break.end():]
        words = re.findall(r"[a-z']+", before)[-6:]
        window = " ".join(words)
        return any(re.search(rf"(?<![a-z']){re.escape(cue)}(?![a-z'])", window) for cue in NEGATION_CUES)

    def _fuzzy(self, clause: str, taken: List[Tuple[int, int]]) -> List[Tuple[int, int, str, str]]:
        hits = []
        tokens = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[a-z']+", clause)]
        for n in (3, 2, 1):
            for i in range(len(tokens) - n + 1):
                s, e = tokens[i][0], tokens[i + n - 1][1]
                if any(s < te and e > ts for ts, te in taken):
                    continue
                gram = clause[s:e]
                if len(gram) < 6:
                    continue
                gw = gram.split()
                for cand in difflib.get_close_matches(gram, list(self.fuzzy_vocab), n=3, cutoff=self.fuzzy_cutoff):
                    cw = cand.split()
                    # typo tolerance per word: "diarrea" -> "diarrhea", but never "swollen and" -> "swollen glands"
                    if len(cw) == len(gw) and all(
                            difflib.SequenceMatcher(None, a, b).ratio() >= 0.8 for a, b in zip(gw, cw)):
                        hits.append((s, e, gram, self.fuzzy_vocab[cand]))
                        taken.append((s, e))
                        break
        return hits

    def extract(self, text: str) -> Extraction:
        ex = Extraction()
        norm = _normalise(text)
        for clause in CLAUSE_SPLIT.split(norm):
            clause = clause.strip(" ,")
            if not clause:
                continue
            taken: List[Tuple[int, int]] = []
            found = []
            for pattern, phrase, key in self.patterns:
                for m in pattern.finditer(clause):
                    if any(m.start() < te and m.end() > ts for ts, te in taken):
                        continue
                    if phrase.isdigit() and not re.search(r"(fever|temperature|degree|°)", clause):
                        continue  # "104" only means fever when temperature is mentioned
                    taken.append((m.start(), m.end()))
                    found.append((m.start(), m.end(), m.group(0), key, False))
            for s, e, gram, key in self._fuzzy(clause, taken):
                found.append((s, e, gram, key, True))
            for s, e, surface, key, fuzzy in sorted(found):
                neg = self._negated(clause, s)
                ex.mentions.append(Mention(key, surface, neg, fuzzy))
                (ex.negated if neg else ex.reported).add(key)

        m = DURATION_RE.search(norm)
        if m:
            qty = m.group(1).lower()
            n = int(qty) if qty.isdigit() else WORD_NUM.get(qty, 1)
            unit = m.group(2).lower().rstrip("s")
            ex.duration_days = n * UNIT_DAYS[unit]
        # clinical duration rule: a cough lasting 2+ weeks is a chronic cough
        if ex.duration_days and ex.duration_days >= 14 and "cough" in self.kb.expand(ex.reported):
            if "chronic_cough" not in ex.reported:
                ex.reported.add("chronic_cough")
                ex.notes.append(f"Cough for about {ex.duration_days} days counts as a chronic cough (2+ weeks).")
        ex.reported -= ex.negated
        ex.present = self.kb.expand(ex.reported) - ex.negated
        return ex


_default: Optional[SymptomExtractor] = None


def extract_symptoms(text: str) -> Extraction:
    global _default
    if _default is None:
        _default = SymptomExtractor()
    return _default.extract(text)
