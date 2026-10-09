"""Export the trained model and indexes for the browser demo in web/.

Writes web/model/bundle.json with:
  * the knowledge base
  * the RF + DT ensemble as plain arrays (every tree: feature, threshold, children,
    leaf class distributions), so the browser reproduces predict_proba exactly
  * the TF-IDF vocabulary, idf weights and the normalised chunk vectors for RAG
  * the bundled OpenStreetMap snapshot

Run:  python scripts/export_web.py      (CI runs it before deploying to GitHub Pages)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

from neurosy.knowledge import KB_PATH, load_kb  # noqa: E402
from neurosy.model import load_or_train  # noqa: E402
from neurosy.rag import Explainer  # noqa: E402

OUT = ROOT / "web" / "model" / "bundle.json"


def export_tree(est, n_classes):
    t = est.tree_
    leaves = {}
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            v = t.value[i][0]
            v = v / v.sum()
            nz = np.nonzero(v > 0)[0]
            leaves[str(i)] = [[int(c), round(float(v[c]), 6)] for c in nz]
    return {"f": t.feature.tolist(), "t": [round(float(x), 4) for x in t.threshold],
            "l": t.children_left.tolist(), "r": t.children_right.tolist(), "v": leaves}


def main():
    kb = load_kb()
    clf = load_or_train()
    model = clf.model
    rf, dt = model.named_estimators_["rf"], model.named_estimators_["dt"]
    n_classes = len(clf.classes)
    forest = {
        "classes": clf.classes, "features": clf.features, "weights": [float(w) for w in model.weights],
        "rf": [export_tree(e, n_classes) for e in rf.estimators_], "dt": export_tree(dt, n_classes),
        "meta": clf.meta,
    }
    ex = Explainer(kb)
    r = ex.retriever
    vec = r.vectorizer
    vocab = {k: int(v) for k, v in vec.vocabulary_.items()}
    m = r.matrix.tocsr()
    chunks = [{"id": c.id, "disease": c.disease, "section": c.section, "title": c.title, "text": c.text,
               "vec": [[int(j), round(float(x), 6)] for j, x in zip(m[i].indices, m[i].data)]}
              for i, c in enumerate(r.chunks)]
    bundle = {
        "kb": json.loads(Path(KB_PATH).read_text()),
        "model": forest,
        "tfidf": {"vocabulary": vocab, "idf": [round(float(x), 6) for x in vec.idf_],
                  "stop_words": sorted(ENGLISH_STOP_WORDS), "chunks": chunks},
        "osm_snapshot": json.loads((ROOT / "data" / "osm_providers_chennai_srm.json").read_text()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, separators=(",", ":")))
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB, {len(forest['rf'])} RF trees)")


if __name__ == "__main__":
    main()
