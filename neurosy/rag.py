"""Retrieval-augmented explanations over the medical knowledge base.

Indexing: every disease is split into section chunks (overview, symptoms,
precautions, when to see a doctor, red flags).

Retrieval: TF-IDF by default (no downloads). Set NEUROSY_EMBEDDINGS=1 with
sentence-transformers installed to use dense embeddings (all-MiniLM-L6-v2),
with FAISS if it is available.

Generation: an extractive, template-based answer built only from retrieved
chunks, with numbered citations. Optionally, an OpenAI-compatible LLM
(NEUROSY_LLM_API_KEY / OPENAI_API_KEY) rewrites it, and a grounding check
falls back to the extractive answer if the LLM says things the sources don't
support.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge import KnowledgeBase, load_kb


@dataclass
class Chunk:
    id: str
    disease: str
    section: str
    title: str
    text: str


def build_chunks(kb: KnowledgeBase) -> List[Chunk]:
    chunks = []
    for key, d in kb.diseases.items():
        typical = sorted(d.symptoms, key=d.symptoms.get, reverse=True)
        sections = {
            "overview": f"{d.name} (ICD-10 {d.icd10}, {d.specialty}). {d.description}",
            "symptoms": f"Typical symptoms of {d.name}: " + ", ".join(kb.label(s) for s in typical) + ".",
            "precautions": f"What to do for {d.name}: " + " ".join(d.precautions),
            "see_doctor": f"When to see a doctor for {d.name}: {d.see_doctor}",
            "red_flags": f"Warning signs with {d.name}: " + "; ".join(d.red_flags) + ".",
        }
        for sec, text in sections.items():
            chunks.append(Chunk(f"{key}:{sec}", key, sec, f"{d.name} - {sec.replace('_', ' ')}", text))
    return chunks


class Retriever:
    def __init__(self, chunks: List[Chunk], use_embeddings: Optional[bool] = None):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        want_dense = use_embeddings if use_embeddings is not None else os.environ.get("NEUROSY_EMBEDDINGS") == "1"
        self.backend = "tfidf"
        if want_dense:
            try:
                from sentence_transformers import SentenceTransformer

                self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.matrix = self.encoder.encode(texts, normalize_embeddings=True)
                self.backend = "embeddings"
                try:
                    import faiss

                    self.index = faiss.IndexFlatIP(self.matrix.shape[1])
                    self.index.add(np.asarray(self.matrix, dtype="float32"))
                    self.backend = "embeddings+faiss"
                except ImportError:
                    self.index = None
            except Exception:
                self.backend = "tfidf"
        if self.backend == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform(texts)

    def _scores(self, query: str) -> np.ndarray:
        if self.backend == "tfidf":
            return cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        q = self.encoder.encode([query], normalize_embeddings=True)
        return (self.matrix @ q[0]).astype(float)

    def search(self, query: str, k: int = 4, disease: Optional[str] = None, boost: float = 0.3) -> List[tuple]:
        scores = self._scores(query)
        if disease:
            scores = scores + np.array([boost if c.disease == disease else 0.0 for c in self.chunks])
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


_SENT = re.compile(r"(?<=[.!?])\s+")


def grounded(answer: str, sources: List[Chunk], min_overlap: float = 0.5) -> bool:
    """Every sentence must share at least ``min_overlap`` of its content words with the sources."""
    vocab = set(re.findall(r"[a-z]{4,}", " ".join(c.text.lower() for c in sources)))
    for sent in _SENT.split(answer.strip()):
        words = set(re.findall(r"[a-z]{4,}", re.sub(r"\[\d+\]", "", sent.lower())))
        if len(words) >= 3 and len(words & vocab) / len(words) < min_overlap:
            return False
    return True


class Explainer:
    def __init__(self, kb: Optional[KnowledgeBase] = None, retriever: Optional[Retriever] = None):
        self.kb = kb or load_kb()
        self.retriever = retriever or Retriever(build_chunks(self.kb))

    SECTION_ORDER = ["overview", "symptoms", "precautions", "see_doctor", "red_flags"]

    def retrieve(self, disease_key: str, symptoms: List[str], user_text: str = "", k: int = 5):
        name = self.kb.diseases[disease_key].name
        query = f"{name} " + " ".join(self.kb.label(s) for s in symptoms) + " " + user_text
        hits = self.retriever.search(query, k=k + 3, disease=disease_key)
        # keep the disease's own sections first, then any related chunk
        own = sorted([h for h in hits if h[0].disease == disease_key],
                     key=lambda h: self.SECTION_ORDER.index(h[0].section))
        other = [h for h in hits if h[0].disease != disease_key]
        return (own + other)[:k]

    def extractive(self, disease_key: str, hits) -> str:
        d = self.kb.diseases[disease_key]
        idx = {h[0].section: i + 1 for i, h in enumerate(hits) if h[0].disease == disease_key}
        parts = []
        if "overview" in idx:
            parts.append(f"{d.description} [{idx['overview']}]")
        if "precautions" in idx:
            parts.append("What helps: " + " ".join(d.precautions[:3]) + f" [{idx['precautions']}]")
        if "see_doctor" in idx:
            parts.append(f"{d.see_doctor} [{idx['see_doctor']}]")
        if "red_flags" in idx:
            parts.append("Watch for: " + "; ".join(d.red_flags) + f". [{idx['red_flags']}]")
        if not parts:
            parts = [f"{h[0].text} [{i + 1}]" for i, h in enumerate(hits[:2])]
        return "\n\n".join(parts)

    def _llm(self, disease_key: str, hits, user_text: str) -> Optional[str]:
        key = os.environ.get("NEUROSY_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        import requests

        base = os.environ.get("NEUROSY_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.environ.get("NEUROSY_LLM_MODEL", "gpt-4o-mini")
        context = "\n".join(f"[{i + 1}] {h[0].text}" for i, h in enumerate(hits))
        prompt = (f"Patient description: {user_text}\nCandidate condition: {self.kb.diseases[disease_key].name}\n\n"
                  f"Sources:\n{context}\n\nUsing ONLY the sources, explain the condition, what helps and when "
                  "to see a doctor in under 120 words. Cite sources like [1]. Do not add facts that are not in the sources.")
        try:
            r = requests.post(f"{base}/chat/completions", timeout=20,
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": model, "temperature": 0,
                                    "messages": [{"role": "system", "content": "You are a careful medical information assistant."},
                                                 {"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    def explain(self, disease_key: str, symptoms: List[str], user_text: str = "") -> dict:
        hits = self.retrieve(disease_key, symptoms, user_text)
        sources = [h[0] for h in hits]
        answer, mode = self.extractive(disease_key, hits), "extractive"
        llm = self._llm(disease_key, hits, user_text)
        if llm:
            if grounded(llm, sources):
                answer, mode = llm, "llm (grounding check passed)"
            else:
                mode = "extractive (LLM answer failed the grounding check)"
        return {"answer": answer, "mode": mode, "retriever": self.retriever.backend,
                "sources": [{"n": i + 1, "id": c.id, "title": c.title, "text": c.text, "score": round(s, 3)}
                            for i, (c, s) in enumerate(hits)]}
