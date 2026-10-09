import { createEngine, DISCLAIMER } from "./engine.js";

const $ = (id) => document.getElementById(id);
const EXAMPLES = [
  "High fever for 3 days, terrible headache, pain behind my eyes and aching joints",
  "Fever with shivering chills every evening then heavy sweating, headache, no cough",
  "Burning when I pee and I need to pee often, no fever",
  "Cough for 3 weeks with phlegm, night sweats and I've lost weight",
  "Chest pain and short of breath, feeling confused",
  "itchy",
];
const LEVEL = { "self-care": "ok", "see a doctor": "doc", urgent: "urgent", emergency: "emergency" };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let engine, last, map, layer;

$("disclaimer").textContent = DISCLAIMER;
EXAMPLES.forEach((t) => {
  const b = document.createElement("button");
  b.className = "chip-btn"; b.textContent = t.length > 48 ? t.slice(0, 46) + "…" : t;
  b.onclick = () => { $("text").value = t; run(); };
  $("examples").appendChild(b);
});

const bar = (v) => `<div class="bar"><div style="width:${Math.max(2, v * 100)}%"></div></div>`;

function render(r) {
  last = r;
  $("results").hidden = false;
  const s = r.symptoms;
  $("symptoms").innerHTML = s.present_labels.length ? s.present_labels.map((x) => `<span class="chip ok">${esc(x)}</span>`).join("")
    : `<span class="muted">No symptoms recognised. Try describing them differently.</span>`;
  $("negated").innerHTML = s.negated_labels.map((x) => `<span class="chip neg">no ${esc(x)}</span>`).join("");
  $("notes").textContent = s.notes.join(" ");
  const t = r.triage;
  $("triage").className = `card triage ${LEVEL[t.level]}`;
  $("triage").innerHTML = `<div class="tlabel">Triage</div><div class="tlevel">${esc(t.level.toUpperCase())}</div><ul>${t.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;

  if (r.abstained) {
    $("preds").innerHTML = `<div class="abstain"><b>No confident suggestion.</b> ${esc(r.abstain_reason || "")}<br/>Describe more symptoms, or see a doctor.` +
      (r.ml_only_top ? `<div class="muted small">The ML model alone would have guessed <b>${esc(r.ml_only_top)}</b>. The rule layer blocked that guess.</div>` : "") + `</div>`;
    $("trace").hidden = true;
  } else {
    $("preds").innerHTML = `<table><thead><tr><th>Condition</th><th>Final</th><th>ML</th><th>Knowledge graph</th><th></th></tr></thead><tbody>` +
      r.predictions.map((p, i) => `<tr class="${i ? "" : "top"}"><td><b>${esc(p.name)}</b></td><td>${bar(p.final)}<span>${p.final.toFixed(2)}</span></td>` +
        `<td>${bar(p.ml_prob)}<span>${p.ml_prob.toFixed(2)}</span></td><td>${bar(p.kg_score)}<span>${p.kg_score.toFixed(2)}</span></td>` +
        `<td><span class="status ${p.status}">${p.status}</span></td></tr>`).join("") + `</tbody></table>`;
    const top = r.predictions[0];
    $("trace").hidden = false;
    $("reasons").innerHTML = top.reasons.map((x) => `<li>${esc(x)}</li>`).join("") +
      `<li>Matched: ${esc(top.matched.join(", "))}</li>` +
      (top.missing_hallmarks.length ? `<li>Typical but not reported: ${esc(top.missing_hallmarks.join(", "))}</li>` : "");
  }
  $("rejected").hidden = !r.rejected_by_rules.length;
  $("rejected").querySelector("summary").textContent = `Rejected by clinical rules (${r.rejected_by_rules.length})`;
  $("rejectedList").innerHTML = r.rejected_by_rules.map((c) => `<li><b>${esc(c.name)}</b> (ML ${c.ml_prob.toFixed(2)}): ${esc(c.reasons[0])}</li>`).join("");

  $("ragCard").hidden = !r.explanation;
  if (r.explanation) {
    $("answer").innerHTML = r.explanation.answer.split("\n\n").map((p) => `<p>${esc(p).replace(/\[(\d)\]/g, "<sup>[$1]</sup>")}</p>`).join("");
    $("sources").innerHTML = r.explanation.sources.map((s) => `<li><b>${esc(s.title)}</b> <span class="muted small">score ${s.score}</span><br/>${esc(s.text)}</li>`).join("");
  }
  $("latency").textContent = `Analysed in ${r.latency_ms} ms, entirely in your browser`;
  care({ sample: true });
}

async function care(opts = {}) {
  if (!last) return;
  const d = last.top_disease;
  $("careStatus").textContent = opts.sample ? "Loading the bundled sample…" : "Searching OpenStreetMap live (can take ~10 s)…";
  const res = await engine.findCare({ ...opts, specialty: d ? d.specialty : "General Medicine", risk: d ? d.risk_level : "medium",
    place: opts.sample ? undefined : opts.place, lat: opts.lat, lon: opts.lon });
  $("careStatus").innerHTML = `Near <b>${esc(res.origin.label)}</b> · ${esc(res.source)}` +
    (d ? ` · ranked for <b>${esc(d.specialty)}</b>, ${esc(d.risk_level)} risk` : "");
  $("providers").innerHTML = res.providers.map((p) => `<li><a href="${p.maps_url}" target="_blank" rel="noreferrer"><b>${esc(p.name)}</b></a>` +
    ` <span class="score">${p.score.toFixed(2)}</span><br/><span class="muted small">${esc(p.why)}</span></li>`).join("") || "<li>No providers found.</li>";
  if (window.L) {
    if (!map) {
      map = L.map("map", { scrollWheelZoom: false });
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "© OpenStreetMap contributors" }).addTo(map);
    }
    if (layer) layer.remove();
    layer = L.layerGroup().addTo(map);
    L.circleMarker([res.origin.lat, res.origin.lon], { radius: 7, color: "#0f766e", fillOpacity: 1 }).bindTooltip("You").addTo(layer);
    const pts = [[res.origin.lat, res.origin.lon]];
    res.providers.forEach((p, i) => { L.marker([p.lat, p.lon]).bindPopup(`${i + 1}. ${esc(p.name)}`).addTo(layer); pts.push([p.lat, p.lon]); });
    map.fitBounds(pts, { padding: [24, 24], maxZoom: 15 });
    setTimeout(() => map.invalidateSize(), 50);
  }
}

function run() {
  const text = $("text").value.trim();
  if (!text || !engine) return;
  render(engine.diagnose(text));
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

$("go").onclick = run;
$("text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); });
$("careBtn").onclick = () => $("place").value.trim() && care({ place: $("place").value.trim() });
$("place").addEventListener("keydown", (e) => { if (e.key === "Enter") $("careBtn").click(); });
$("sampleBtn").onclick = () => care({ sample: true });
$("geoBtn").onclick = () => navigator.geolocation?.getCurrentPosition(
  (pos) => care({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
  () => { $("careStatus").textContent = "Location permission denied. Type a place instead."; });

(async () => {
  const t0 = performance.now();
  $("status").textContent = "Downloading the model (≈0.3 MB)…";
  const bundle = await (await fetch("model/bundle.json")).json();
  engine = createEngine(bundle);
  $("nDis").textContent = engine.nDiseases;
  $("go").disabled = false; $("go").textContent = "Analyse";
  $("status").textContent = `Model ready in ${Math.round(performance.now() - t0)} ms · ${engine.nDiseases} conditions · RF+DT held-out accuracy ${engine.meta.holdout_accuracy} (synthetic data)`;
  const q = new URLSearchParams(location.search).get("q");
  if (q) { $("text").value = q; run(); }
})();
