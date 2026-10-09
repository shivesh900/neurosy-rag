"""Streamlit demo:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from neurosy import DISCLAIMER
from neurosy.pipeline import NeuroSyRAG

st.set_page_config(page_title="NeuroSy-RAG", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading knowledge base and model…")
def engine():
    return NeuroSyRAG()


EXAMPLES = [
    "I've had a high fever for 3 days, a terrible headache, pain behind my eyes and aching joints",
    "burning when I pee and I need to pee often, no fever",
    "cough for 3 weeks with phlegm, night sweats and I've lost weight",
    "chest pain and short of breath, feeling confused",
    "itchy",
]
LEVEL = {"self-care": ("🟢", "success"), "see a doctor": ("🟡", "info"), "urgent": ("🟠", "warning"),
         "emergency": ("🔴", "error")}

st.title("🩺 NeuroSy-RAG")
st.caption("Hybrid neuro-symbolic symptom checker: NLP extraction → RF + DT ensemble → knowledge-graph rule "
           "validation → RAG explanation → nearby care ranking")
st.warning(DISCLAIMER, icon="⚠️")

with st.sidebar:
    st.header("Try an example")
    for ex in EXAMPLES:
        if st.button(ex[:60] + ("…" if len(ex) > 60 else ""), use_container_width=True):
            st.session_state["text"] = ex
    st.divider()
    place = st.text_input("Find care near (optional)", placeholder="e.g. Tambaram, Chennai")
    offline = st.checkbox("Use the offline map snapshot (SRM KTR)", value=not place)
    st.divider()
    eng = engine()
    st.caption(f"Model: RF+DT soft-voting ensemble · {eng.classifier.meta.get('n_classes')} conditions · "
               f"held-out accuracy {eng.classifier.meta.get('holdout_accuracy')} (synthetic data)")
    st.caption(f"Retriever: {eng.explainer.retriever.backend}")

text = st.text_area("Describe your symptoms in plain English", key="text", height=100,
                    placeholder="e.g. fever with chills and sweating for 2 days, headache, no cough")
go = st.button("Analyse", type="primary")

if go and text.strip():
    r = eng.diagnose(text, place=place or None, offline=offline)
    sy = r["symptoms"]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("1 · Symptoms understood (NLP)")
        st.write(" ".join(f"`{s}`" for s in sy["present_labels"]) or "_none recognised_")
        if sy["negated_labels"]:
            st.write("Denied: " + " ".join(f"~~{s}~~" for s in sy["negated_labels"]))
        for n in sy["notes"]:
            st.caption("ℹ️ " + n)
    with c2:
        icon, kind = LEVEL[r["triage"]["level"]]
        getattr(st, kind)(f"{icon} **Triage: {r['triage']['level'].upper()}**\n\n" +
                          "\n".join("- " + x for x in r["triage"]["reasons"]))

    st.subheader("2 · Neuro-symbolic diagnosis")
    if r["abstained"]:
        st.info(f"**No confident suggestion.** {r['abstain_reason']} Please describe more symptoms or see a doctor.")
        if r["ml_only_top"]:
            st.caption(f"(The ML model alone would have guessed **{r['ml_only_top']}**. The rule layer blocked that guess.)")
    else:
        df = pd.DataFrame([{"Condition": p["name"], "Final score": p["final"], "ML probability": p["ml_prob"],
                            "Knowledge-graph score": p["kg_score"], "Status": p["status"]} for p in r["predictions"]])
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.2f")
                                    for c in ("Final score", "ML probability", "Knowledge-graph score")})
        top = r["predictions"][0]
        with st.expander(f"Why {top['name']}? (rule trace)", expanded=True):
            for reason in top["reasons"]:
                st.write("• " + reason)
            st.write("Matched: " + ", ".join(top["matched"]))
            if top["missing_hallmarks"]:
                st.write("Typical but not reported: " + ", ".join(top["missing_hallmarks"]))
    if r["rejected_by_rules"]:
        with st.expander(f"Rejected by clinical rules ({len(r['rejected_by_rules'])})"):
            for c in r["rejected_by_rules"]:
                st.write(f"• **{c['name']}** (ML {c['ml_prob']:.2f}): " + " ".join(c["reasons"][:1]))

    if r["explanation"]:
        st.subheader("3 · What it is and what to do (RAG)")
        st.markdown(r["explanation"]["answer"].replace("\n\n", "\n\n"))
        with st.expander("Retrieved sources"):
            for s in r["explanation"]["sources"]:
                st.markdown(f"**[{s['n']}] {s['title']}** (score {s['score']})  \n{s['text']}")
        st.caption(f"Generation mode: {r['explanation']['mode']}")

    if r["care"] and r["care"].get("providers"):
        st.subheader("4 · Nearby care, ranked")
        st.caption(f"Near {r['care']['origin']['label']} · {r['care']['source']} · {r['care']['attribution']}")
        prov = r["care"]["providers"]
        st.dataframe(pd.DataFrame([{"Provider": p["name"], "Score": p["score"], "Distance (km)": p["distance_km"],
                                    "Why": p["why"], "Map": p["maps_url"]} for p in prov]),
                     hide_index=True, use_container_width=True,
                     column_config={"Map": st.column_config.LinkColumn("Map", display_text="open"),
                                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f")})
        st.map(pd.DataFrame([{"lat": p["lat"], "lon": p["lon"]} for p in prov]), size=60)
    st.caption(f"Processed in {r['latency_ms']} ms")
