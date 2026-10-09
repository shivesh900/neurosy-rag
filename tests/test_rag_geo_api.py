from neurosy.geo import haversine_km, rank_providers, load_snapshot
from neurosy.rag import Explainer, grounded


def test_retrieval_returns_own_sections_first():
    ex = Explainer()
    hits = ex.retrieve("malaria", ["chills", "sweating"], "fever with chills")
    assert hits[0][0].disease == "malaria"
    assert {h[0].section for h in hits} >= {"overview", "see_doctor"}


def test_grounding_check_blocks_unsupported_text():
    ex = Explainer()
    sources = [h[0] for h in ex.retrieve("dengue", ["fever"])]
    assert grounded("Dengue is a mosquito-borne viral infection spread by Aedes mosquitoes.", sources)
    assert not grounded("Quantum vaccines cure dengue instantly using blockchain nanobots tonight.", sources)


def test_llm_answer_falls_back_when_ungrounded(monkeypatch):
    ex = Explainer()
    monkeypatch.setattr(ex, "_llm", lambda *a, **k: "Eat raw garlic and drink bleach; doctors are unnecessary.")
    out = ex.explain("dengue", ["fever"])
    assert out["mode"].startswith("extractive")


def test_haversine():
    assert abs(haversine_km((12.8231, 80.0442), (13.0827, 80.2707)) - 37.8) < 1.5  # SRM KTR -> Chennai centre


def test_provider_ranking_prefers_near_hospitals_for_high_risk():
    origin, providers = load_snapshot()
    ranked = rank_providers(providers, origin, "Infectious Diseases", "high", top_n=5)
    assert len(ranked) == 5
    assert ranked[0].score >= ranked[-1].score
    assert ranked[0].kind == "hospital"


def test_api(engine, monkeypatch):
    from fastapi.testclient import TestClient

    import neurosy.api as api

    monkeypatch.setattr(api, "get_engine", lambda: engine)
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/diagnose", json={"text": "burning when I pee and peeing often", "find_providers": False})
    assert r.status_code == 200
    assert r.json()["predictions"][0]["disease"] == "uti"
