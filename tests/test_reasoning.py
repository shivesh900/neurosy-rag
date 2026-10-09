from neurosy.knowledge import load_kb
from neurosy.symbolic import NeuroSymbolicReasoner


def test_rule_rejects_disease_without_required_symptom():
    kb = load_kb()
    r = NeuroSymbolicReasoner(kb)
    present = kb.expand({"joint_pain", "rash", "pain_behind_eyes"})
    # pretend the ML model is sure it's dengue, but the patient said "no fever"
    ml = {k: 0.0 for k in kb.diseases}
    ml["dengue"] = 0.9
    out = r.reason(ml, present, present, negated={"fever"})
    assert all(c.disease != "dengue" for c in out["candidates"])
    rejected = {c.disease: c for c in out["rejected"]}
    assert "dengue" in rejected and "R2" in rejected["dengue"].reasons[0]


def test_red_flags_raise_triage():
    kb = load_kb()
    t = NeuroSymbolicReasoner(kb).triage(kb.expand({"chest_pain", "shortness_of_breath"}), None, None)
    assert t.level == "emergency"


def test_end_to_end_dengue(engine):
    r = engine.diagnose("high fever for 3 days, severe headache, pain behind my eyes, joint pain and a rash",
                        find_providers=False)
    assert not r["abstained"]
    assert r["predictions"][0]["disease"] == "dengue"
    assert r["triage"]["level"] in {"urgent", "emergency"}
    assert r["explanation"]["sources"]
    assert "[1]" in r["explanation"]["answer"]  # answers cite retrieved sources


def test_abstains_on_single_vague_symptom(engine):
    r = engine.diagnose("itchy", find_providers=False)
    assert r["abstained"]
    assert r["predictions"] == [] or r["abstain_reason"]


def test_vignette_accuracy_floor(engine):
    import json
    from pathlib import Path

    vig = json.loads((Path(__file__).resolve().parent.parent / "data" / "vignettes.json").read_text())
    hits = 0
    for v in vig:
        preds = engine.diagnose(v["text"], find_providers=False)["predictions"]
        hits += bool(preds) and preds[0]["disease"] == v["expected"]
    assert hits / len(vig) >= 0.8
