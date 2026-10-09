"""Check that the browser engine (web/engine.js) gives the same answers as the Python engine.

Runs every vignette (plus a few edge cases) through both and compares extracted
symptoms, ranked diagnoses, fused scores, rule status, abstention and triage.
Requires Node 18+ and web/model/bundle.json (python scripts/export_web.py).

Run:  python scripts/check_web_parity.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neurosy.pipeline import NeuroSyRAG  # noqa: E402

EXTRA = [
    "itchy", "no fever but bad headche", "chest pain and short of breath, feeling confused",
    "not sneezing, no runny nose, just a sore throat and a fever", "I have diarrea and vomitting since yesterday",
    "cough for three weeks, night sweats", "fever, rash, joint pain, no headache", "hello there",
]
NODE = """
import { readFileSync } from "node:fs";
import { createEngine } from "./web/engine.js";
const eng = createEngine(JSON.parse(readFileSync("web/model/bundle.json", "utf8")));
const texts = JSON.parse(readFileSync(0, "utf8"));
console.log(JSON.stringify(texts.map((t) => eng.diagnose(t))));
"""


def summary(r):
    return {
        "present": sorted(r["symptoms"]["present"]), "negated": sorted(r["symptoms"]["negated"]),
        "abstained": r["abstained"], "triage": r["triage"]["level"],
        "preds": [(p["disease"], p["status"]) for p in r["predictions"]],
        "finals": [p["final"] for p in r["predictions"]],
        "answer": (r["explanation"] or {}).get("answer"),
    }


def main():
    texts = [v["text"] for f in ("vignettes.json", "vignettes_holdout.json")
             for v in json.loads((ROOT / "data" / f).read_text())] + EXTRA
    eng = NeuroSyRAG()
    py = [summary(eng.diagnose(t, find_providers=False)) for t in texts]
    script = ROOT / "_parity.mjs"
    script.write_text(NODE)
    try:
        out = subprocess.run(["node", str(script)], input=json.dumps(texts), cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    finally:
        script.unlink()
    js = [summary(r) for r in json.loads(out)]
    bad = 0
    for t, a, b in zip(texts, py, js):
        same = (a["present"] == b["present"] and a["negated"] == b["negated"] and a["abstained"] == b["abstained"]
                and a["triage"] == b["triage"] and a["preds"] == b["preds"] and a["answer"] == b["answer"]
                and all(abs(x - y) < 1e-3 for x, y in zip(a["finals"], b["finals"])))
        if not same:
            bad += 1
            print(f"DIFF: {t!r}")
            for k in a:
                if a[k] != b[k]:
                    print(f"   {k}: py={a[k]!r}\n   {' ' * len(k)}  js={b[k]!r}")
    print(f"{len(texts) - bad}/{len(texts)} inputs give identical results in Python and in the browser engine")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
