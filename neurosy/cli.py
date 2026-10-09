"""Command line interface.

    python -m neurosy.cli "high fever for 3 days, pain behind my eyes, joint pain and a rash"
    python -m neurosy.cli "burning when i pee and peeing a lot" --place "Guindy, Chennai"
"""
from __future__ import annotations

import argparse
import json
import textwrap

from .pipeline import NeuroSyRAG

LEVEL_ICON = {"self-care": "[ok]", "see a doctor": "[doctor]", "urgent": "[URGENT]", "emergency": "[EMERGENCY]"}


def render(r: dict) -> str:
    w = lambda s, i="  ": textwrap.fill(s, 96, initial_indent=i, subsequent_indent=i)  # noqa: E731
    lines = ["", "Symptoms understood: " + (", ".join(r["symptoms"]["present_labels"]) or "none")]
    if r["symptoms"]["negated_labels"]:
        lines.append("Denied: " + ", ".join(r["symptoms"]["negated_labels"]))
    for n in r["symptoms"]["notes"]:
        lines.append("Note: " + n)
    lines.append("")
    if r["abstained"]:
        lines.append("No confident suggestion: " + (r["abstain_reason"] or ""))
    else:
        lines.append(f"{'Condition':<34}{'Final':>7}{'ML':>7}{'KG':>7}  Status")
        for p in r["predictions"]:
            lines.append(f"{p['name'][:33]:<34}{p['final']:>7.2f}{p['ml_prob']:>7.2f}{p['kg_score']:>7.2f}  {p['status']}")
        lines.append("")
        lines.append("Why (neuro-symbolic checks on the top suggestion):")
        for reason in r["predictions"][0]["reasons"]:
            lines.append(w("- " + reason))
    if r["rejected_by_rules"]:
        lines.append("Rejected by clinical rules:")
        for c in r["rejected_by_rules"]:
            lines.append(w(f"- {c['name']} (ML {c['ml_prob']:.2f}): " + " ".join(c["reasons"][:1])))
    lines.append("")
    lines.append(f"Triage: {LEVEL_ICON[r['triage']['level']]} {r['triage']['level']}")
    for reason in r["triage"]["reasons"]:
        lines.append(w("- " + reason))
    if r["explanation"]:
        lines.append("")
        lines.append(f"About it (retrieved from the knowledge base, {r['explanation']['retriever']}):")
        for para in r["explanation"]["answer"].split("\n\n"):
            lines.append(w(para))
        lines.append("  Sources: " + "; ".join(f"[{s['n']}] {s['title']}" for s in r["explanation"]["sources"]))
    if r["care"] and r["care"].get("providers"):
        lines.append("")
        lines.append(f"Nearby care near {r['care']['origin']['label']} ({r['care']['source']}):")
        for p in r["care"]["providers"]:
            lines.append(w(f"- {p['name']}: score {p['score']:.2f} ({p['why']})"))
        lines.append("  " + r["care"]["attribution"])
    lines += ["", w(r["disclaimer"], ""), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NeuroSy-RAG symptom checker (educational).")
    ap.add_argument("text", help="describe the symptoms in plain English")
    ap.add_argument("--place", help="city or area to find nearby care, e.g. 'Tambaram, Chennai'")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--offline-care", action="store_true", help="use the bundled OSM snapshot for providers")
    ap.add_argument("--json", action="store_true", help="print the full JSON result")
    a = ap.parse_args(argv)
    r = NeuroSyRAG().diagnose(a.text, place=a.place, lat=a.lat, lon=a.lon, offline=a.offline_care)
    print(json.dumps(r, indent=2) if a.json else render(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
