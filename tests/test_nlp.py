import pytest

from neurosy.nlp import extract_symptoms


@pytest.mark.parametrize("text,present,negated", [
    ("I have a high fever and a terrible headache", {"high_fever", "fever", "severe_headache", "headache"}, set()),
    ("no fever but bad headche", {"headache"}, {"fever"}),                       # negation + typo
    ("not sneezing, no runny nose, just a sore throat", {"sore_throat"}, {"sneezing", "runny_nose"}),
    ("my eyes have turned yellow and urine is dark", {"yellow_skin", "dark_urine"}, set()),  # inversion
    ("I have diarrea and vomitting", {"diarrhea", "vomiting"}, set()),
])
def test_extraction(text, present, negated):
    ex = extract_symptoms(text)
    assert present <= ex.present
    assert negated == ex.negated


def test_duration_rule_makes_chronic_cough():
    ex = extract_symptoms("cough for three weeks and night sweats")
    assert ex.duration_days == 21
    assert "chronic_cough" in ex.present


def test_no_false_fuzzy_match():
    assert "swollen_lymph_nodes" not in extract_symptoms("the toe is red, swollen and painful").present
