// NeuroSy-RAG in the browser: a faithful JavaScript port of neurosy/ (nlp, model, symbolic, rag, geo).
// Runs the exported Python model (scripts/export_web.py) and is checked against the Python engine
// by scripts/check_web_parity.py. Works in browsers and in Node 18+.

// ------------------------------------------------------------------ difflib (Python port)
function longestMatch(a, b, b2j, alo, ahi, blo, bhi) {
  let besti = alo, bestj = blo, bestsize = 0, j2len = new Map();
  for (let i = alo; i < ahi; i++) {
    const newj2len = new Map();
    for (const j of b2j.get(a[i]) || []) {
      if (j < blo) continue;
      if (j >= bhi) break;
      const k = (j2len.get(j - 1) || 0) + 1;
      newj2len.set(j, k);
      if (k > bestsize) { besti = i - k + 1; bestj = j - k + 1; bestsize = k; }
    }
    j2len = newj2len;
  }
  while (besti > alo && bestj > blo && a[besti - 1] === b[bestj - 1]) { besti--; bestj--; bestsize++; }
  while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] === b[bestj + bestsize]) bestsize++;
  return [besti, bestj, bestsize];
}
export function ratio(a, b) {
  const b2j = new Map();
  [...b].forEach((ch, j) => { if (!b2j.has(ch)) b2j.set(ch, []); b2j.get(ch).push(j); });
  let matches = 0;
  const queue = [[0, a.length, 0, b.length]];
  while (queue.length) {
    const [alo, ahi, blo, bhi] = queue.pop();
    const [i, j, k] = longestMatch(a, b, b2j, alo, ahi, blo, bhi);
    if (k) {
      matches += k;
      if (alo < i && blo < j) queue.push([alo, i, blo, j]);
      if (i + k < ahi && j + k < bhi) queue.push([i + k, ahi, j + k, bhi]);
    }
  }
  const t = a.length + b.length;
  return t ? (2 * matches) / t : 1;
}
function closeMatches(word, possibilities, n, cutoff) {
  const res = [];
  for (const x of possibilities) { const r = ratio(x, word); if (r >= cutoff) res.push([r, x]); }
  res.sort((p, q) => q[0] - p[0] || (q[1] > p[1] ? 1 : q[1] < p[1] ? -1 : 0));
  return res.slice(0, n).map((p) => p[1]);
}

const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

// ------------------------------------------------------------------ knowledge base
class KB {
  constructor(raw) {
    this.symptoms = raw.symptoms;
    this.diseases = raw.diseases;
    this.keys = Object.keys(raw.symptoms).sort(cmp);
    this.childrenOf = {};
    for (const [child, info] of Object.entries(this.symptoms))
      for (const p of info.implies || []) (this.childrenOf[p] ||= new Set()).add(child);
  }
  label(s) { return this.symptoms[s]?.label ?? s.replace(/_/g, " "); }
  expand(list) {
    const out = new Set(list), stack = [...out];
    while (stack.length) for (const p of this.symptoms[stack.pop()]?.implies || []) if (!out.has(p)) { out.add(p); stack.push(p); }
    return out;
  }
  children(s) { return this.childrenOf[s] || new Set(); }
}

// ------------------------------------------------------------------ NLP
const NEGATION_CUES = ["no", "not", "without", "never", "denies", "deny", "dont have", "don't have", "do not have",
  "didn't have", "did not have", "haven't had", "free of", "absence of", "negative for", "nor", "none"];
const NEG_RES = NEGATION_CUES.map((c) => new RegExp(`(?<![a-z'])${esc(c)}(?![a-z'])`));
const SCOPE_BREAKERS = /\b(but|however|although|though|except|yet|just|only|and i have|and have|i have|i do have|there is)\b/g;
const CLAUSE_SPLIT = /[.;!?\n]|,\s*(?=but\b)|\bbut\b/;
const WORD_NUM = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  "a couple of": 2, "a few": 3, several: 4, a: 1, an: 1 };
const DURATION_RE = /(?:for|since|past|last)\s+(?:the\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a couple of|a few|several|a|an)\s*(day|days|week|weeks|month|months)/i;
const UNIT_DAYS = { day: 1, week: 7, month: 30 };
const BODY = "(eyes?|skin|urine|joints?|toes?|knees?|wrists?|fingers?|throat|nose|head|chest|stomach|neck|heart)";
const STATE = "(yellow|dark|red|swollen|painful|sore|stiff|blocked|itchy|tight|pounding|racing|hot|dry|scaly|aching)";
const INVERSION = new RegExp(`\\b${BODY}\\s+(?:is|are|was|were|feels?|felt|looks?|has|have|got|gets|turned|became|keeps? getting)` +
  `(?:\\s+(?:turned|become|been|gone|getting|so|very|really|quite|all|a bit))*\\s+${STATE}\\b`, "g");
const DETERMINERS = /(?<=\s)(my|the|both|his|her|your|their|our)\s+/g;

function normalise(text) {
  text = text.toLowerCase().replace(/\u2019/g, "'");
  text = text.replace(/[^a-z0-9'\s.,;!?/-]/g, " ").replace(/[ \t]+/g, " ");
  text = text.replace(INVERSION, (m, body, state) => `${m}, ${state} ${body}`);
  return text.replace(DETERMINERS, " ").replace(/ {2,}/g, " ");
}

class SymptomExtractor {
  constructor(kb, cutoff = 0.86) {
    this.kb = kb; this.cutoff = cutoff;
    const set = new Map();
    for (const [key, info] of Object.entries(kb.symptoms))
      for (const syn of new Set([...info.synonyms, info.label])) {
        const p = (" " + syn.toLowerCase()).replace(DETERMINERS, " ").replace(/ {2,}/g, " ").trim();
        set.set(`${p}\u0000${key}`, [p, key]);
      }
    this.phrases = [...set.values()].sort((x, y) => y[0].length - x[0].length || cmp(x[0], y[0]) || cmp(x[1], y[1]));
    this.patterns = this.phrases.map(([p, k]) => [new RegExp(`(?<![a-z0-9])${esc(p)}(?![a-z0-9])`, "g"), p, k]);
    this.fuzzy = new Map();
    for (const [p, k] of this.phrases) if (p.length >= 6 && !/^\d+$/.test(p)) this.fuzzy.set(p, k);
    this.fuzzyKeys = [...this.fuzzy.keys()];
  }
  negated(clause, start) {
    let before = clause.slice(0, start), last = null;
    for (const m of before.matchAll(SCOPE_BREAKERS)) last = m;
    if (last) before = before.slice(last.index + last[0].length);
    const window = (before.match(/[a-z']+/g) || []).slice(-6).join(" ");
    return NEG_RES.some((re) => re.test(window));
  }
  fuzzyHits(clause, taken) {
    const hits = [], toks = [...clause.matchAll(/[a-z']+/g)].map((m) => [m.index, m.index + m[0].length]);
    for (const n of [3, 2, 1]) {
      for (let i = 0; i + n <= toks.length; i++) {
        const s = toks[i][0], e = toks[i + n - 1][1];
        if (taken.some(([ts, te]) => s < te && e > ts)) continue;
        const gram = clause.slice(s, e);
        if (gram.length < 6) continue;
        const gw = gram.split(/\s+/);
        for (const cand of closeMatches(gram, this.fuzzyKeys, 3, this.cutoff)) {
          const cw = cand.split(/\s+/);
          if (cw.length === gw.length && gw.every((w, j) => ratio(w, cw[j]) >= 0.8)) {
            hits.push([s, e, gram, this.fuzzy.get(cand)]); taken.push([s, e]); break;
          }
        }
      }
    }
    return hits;
  }
  extract(text) {
    const reported = new Set(), negated = new Set(), mentions = [], notes = [];
    const norm = normalise(text);
    for (let clause of norm.split(CLAUSE_SPLIT)) {
      clause = clause.replace(/^[ ,]+|[ ,]+$/g, "");
      if (!clause) continue;
      const taken = [], found = [];
      for (const [re, phrase, key] of this.patterns) {
        re.lastIndex = 0;
        for (const m of clause.matchAll(re)) {
          const s = m.index, e = s + m[0].length;
          if (taken.some(([ts, te]) => s < te && e > ts)) continue;
          if (/^\d+$/.test(phrase) && !/(fever|temperature|degree|°)/.test(clause)) continue;
          taken.push([s, e]); found.push([s, e, m[0], key, false]);
        }
      }
      for (const [s, e, g, k] of this.fuzzyHits(clause, taken)) found.push([s, e, g, k, true]);
      found.sort((x, y) => x[0] - y[0] || x[1] - y[1] || cmp(x[2], y[2]) || cmp(x[3], y[3]));
      for (const [s, , surface, key, fuzzy] of found) {
        const neg = this.negated(clause, s);
        mentions.push({ symptom: key, text: surface, negated: neg, fuzzy });
        (neg ? negated : reported).add(key);
      }
    }
    let duration = null;
    const m = norm.match(DURATION_RE);
    if (m) {
      const q = m[1].toLowerCase();
      const n = /^\d+$/.test(q) ? parseInt(q, 10) : (WORD_NUM[q] ?? 1);
      duration = n * UNIT_DAYS[m[2].toLowerCase().replace(/s$/, "")];
    }
    if (duration && duration >= 14 && this.kb.expand(reported).has("cough") && !reported.has("chronic_cough")) {
      reported.add("chronic_cough");
      notes.push(`Cough for about ${duration} days counts as a chronic cough (2+ weeks).`);
    }
    for (const s of negated) reported.delete(s);
    const present = new Set([...this.kb.expand(reported)].filter((s) => !negated.has(s)));
    return { present, reported, negated, mentions, duration_days: duration, notes };
  }
}

// ------------------------------------------------------------------ RF + DT ensemble
function treeProba(t, x, nClasses) {
  let i = 0;
  while (t.l[i] !== -1) i = x[t.f[i]] <= t.t[i] ? t.l[i] : t.r[i];
  const out = new Float64Array(nClasses);
  for (const [c, p] of t.v[i]) out[c] = p;
  return out;
}
class Classifier {
  constructor(m) { Object.assign(this, m); this.n = m.classes.length; }
  predictProba(symptoms) {
    const x = this.features.map((f) => (symptoms.has(f) ? 1 : 0));
    const rf = new Float64Array(this.n);
    for (const t of this.rf) { const p = treeProba(t, x, this.n); for (let c = 0; c < this.n; c++) rf[c] += p[c]; }
    const dt = treeProba(this.dt, x, this.n);
    const [w1, w2] = this.weights, out = {};
    this.classes.forEach((c, i) => { out[c] = (w1 * (rf[i] / this.rf.length) + w2 * dt[i]) / (w1 + w2); });
    return out;
  }
}

// ------------------------------------------------------------------ symbolic
const URGENCY = ["self-care", "see a doctor", "urgent", "emergency"];
class Reasoner {
  constructor(kb, mlWeight = 0.55, abstainBelow = 0.3) { this.kb = kb; this.w = mlWeight; this.abstainBelow = abstainBelow; }
  coverage(key, present, reported) {
    const d = this.kb.diseases[key];
    const syms = Object.entries(d.symptoms);
    const total = syms.reduce((a, [, w]) => a + w, 0);
    let got = 0; const matched = [];
    const parentHit = (s) => [...this.kb.expand([s])].some((p) => p !== s && present.has(p));
    for (const [s, w] of syms) {
      if (present.has(s)) { got += w; matched.push(s); } else if (parentHit(s)) { got += 0.6 * w; matched.push(s); }
    }
    const support = total ? got / total : 0;
    const profile = this.kb.expand(Object.keys(d.symptoms));
    const dset = new Set(Object.keys(d.symptoms));
    const explained = [...reported].filter((s) => profile.has(s) || [...this.kb.children(s)].some((c) => dset.has(c)));
    const explains = reported.size ? explained.length / reported.size : 0;
    const hallmarks = syms.filter(([s, w]) => w >= 0.85 && !present.has(s) && !parentHit(s)).map(([s]) => s);
    return [support, explains, matched, hallmarks];
  }
  check(c, present, reported, negated) {
    const d = this.kb.diseases[c.disease], lab = (s) => this.kb.label(s);
    for (const g of d.requires) {
      if (!g.some((s) => present.has(s))) {
        const denied = g.filter((s) => negated.has(s));
        const names = g.map(lab).join(" or ");
        c.reasons.push(denied.length && denied.length === g.length
          ? `R2 contradiction: ${d.name} needs ${names}, which you said you don't have.`
          : `R1 necessary finding missing: ${d.name} needs ${names}.`);
        c.status = "rejected";
      }
    }
    const [support, explains, matched, hallmarks] = this.coverage(c.disease, present, reported);
    Object.assign(c, { kg_support: support, kg_explains: explains, kg_score: 0.6 * support + 0.4 * explains,
      matched: matched.map(lab), missing_hallmarks: hallmarks.map(lab) });
    const deniedH = Object.entries(d.symptoms).filter(([s, w]) => w >= 0.85 && negated.has(s)).map(([s]) => s);
    const penalty = 0.7 ** deniedH.length;
    if (deniedH.length) c.reasons.push("R2 denied hallmark symptom: " + deniedH.map(lab).join(", ") + ".");
    if (c.status === "rejected") { c.final = 0; return c; }
    c.final = (this.w * c.ml_prob + (1 - this.w) * c.kg_score) * penalty;
    if (c.kg_score < 0.35) {
      c.status = "adjusted";
      c.reasons.push(`R3 weak knowledge-graph support (${c.kg_score.toFixed(2)}): few of this disease's typical symptoms were reported.`);
    } else {
      c.reasons.push(`R3 knowledge graph supports it: ${matched.length} typical symptoms matched (coverage ${Math.round(support * 100)}%, explains ${Math.round(explains * 100)}% of your symptoms).`);
    }
    return c;
  }
  reason(ml, present, reported, negated, topK = 3, pool = 8) {
    const keys = Object.keys(ml);
    const mlRank = keys.map((k, i) => [k, i]).sort((a, b) => ml[b[0]] - ml[a[0]] || a[1] - b[1]).map((p) => p[0]);
    const poolKeys = [...new Set([...mlRank.slice(0, pool),
      ...Object.keys(this.kb.diseases).filter((k) => this.coverage(k, present, reported)[0] >= 0.5)])];
    const cands = poolKeys.map((k) => this.check({ disease: k, name: this.kb.diseases[k].name, ml_prob: ml[k] || 0,
      kg_support: 0, kg_explains: 0, kg_score: 0, final: 0, status: "validated", reasons: [], matched: [], missing_hallmarks: [] },
      present, reported, negated));
    cands.sort((a, b) => b.final - a.final);
    const accepted = cands.filter((c) => c.status !== "rejected");
    const rejected = cands.filter((c) => c.status === "rejected" && c.ml_prob >= 0.05);
    if (accepted.length && accepted[0].disease !== mlRank[0])
      accepted[0].reasons.unshift(`Ranked first after symbolic validation (the ML model alone preferred ${this.kb.diseases[mlRank[0]].name}).`);
    const abstain = !accepted.length || accepted[0].final < this.abstainBelow || reported.size < 2;
    let why = null;
    if (!reported.size) why = "No recognisable symptoms were found in the text.";
    else if (reported.size < 2) why = "Only one symptom was given. That's not enough to suggest a condition safely.";
    else if (!accepted.length) why = "Every candidate the model proposed failed a clinical rule.";
    else if (accepted[0].final < this.abstainBelow) why = `The best candidate's combined confidence (${accepted[0].final.toFixed(2)}) is below the ${this.abstainBelow} safety threshold.`;
    return { candidates: accepted.slice(0, topK), rejected, abstain, abstain_reason: why, ml_top: mlRank[0] };
  }
  triage(p, top, duration) {
    let level = "self-care"; const reasons = [];
    const bump = (n, why) => { reasons.push(why); if (URGENCY.indexOf(n) > URGENCY.indexOf(level)) level = n; };
    const has = (...s) => s.every((x) => p.has(x));
    if (p.has("confusion")) bump("emergency", "Confusion or disorientation is a red-flag symptom.");
    if (p.has("coughing_blood")) bump("emergency", "Coughing up blood needs immediate assessment.");
    if (has("chest_pain", "shortness_of_breath")) bump("emergency", "Chest pain with breathlessness can be a heart or lung emergency.");
    if (has("nosebleed_or_gum_bleeding", "fever")) bump("emergency", "Bleeding with fever can be a warning sign of severe dengue.");
    if (p.has("lower_right_abdominal_pain") && (p.has("fever") || p.has("vomiting"))) bump("emergency", "Lower-right abdominal pain with fever or vomiting can mean appendicitis.");
    if (has("severe_headache", "neck_pain", "fever")) bump("emergency", "Severe headache with neck stiffness and fever can mean meningitis.");
    if (p.has("shortness_of_breath")) bump("urgent", "Breathlessness should be checked by a doctor soon.");
    if (p.has("yellow_skin")) bump("urgent", "Jaundice (yellow skin or eyes) needs liver tests.");
    if (p.has("high_fever")) bump("see a doctor", "High fever should be checked by a doctor within 24 hours.");
    if (duration && duration >= 14) bump("see a doctor", `Symptoms for ${duration} days need a medical review.`);
    if (top) {
      const risk = this.kb.diseases[top.disease].risk_level;
      if (risk === "high") bump("urgent", `The leading candidate (${top.name}) is a high-risk condition.`);
      else if (risk === "medium") bump("see a doctor", `${top.name} usually needs a doctor's diagnosis.`);
    }
    if (!reasons.length) reasons.push("No red-flag symptoms were detected.");
    return { level, reasons };
  }
}

// ------------------------------------------------------------------ RAG (TF-IDF over exported chunk vectors)
const SECTION_ORDER = ["overview", "symptoms", "precautions", "see_doctor", "red_flags"];
class Explainer {
  constructor(kb, tfidf) {
    this.kb = kb; this.vocab = tfidf.vocabulary; this.idf = tfidf.idf; this.stop = new Set(tfidf.stop_words);
    this.chunks = tfidf.chunks.map((c) => ({ ...c, vecMap: new Map(c.vec) }));
  }
  embed(q) {
    const toks = (q.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || []).filter((t) => !this.stop.has(t));
    const grams = [...toks];
    for (let i = 0; i + 1 < toks.length; i++) grams.push(`${toks[i]} ${toks[i + 1]}`);
    const tf = new Map();
    for (const g of grams) if (g in this.vocab) tf.set(this.vocab[g], (tf.get(this.vocab[g]) || 0) + 1);
    const v = new Map(); let norm = 0;
    for (const [j, c] of tf) { const w = (1 + Math.log(c)) * this.idf[j]; v.set(j, w); norm += w * w; }
    norm = Math.sqrt(norm) || 1;
    for (const [j, w] of v) v.set(j, w / norm);
    return v;
  }
  search(q, k, disease, boost = 0.3) {
    const v = this.embed(q);
    const scored = this.chunks.map((c, i) => {
      let s = 0; for (const [j, w] of v) s += w * (c.vecMap.get(j) || 0);
      return [c, s + (c.disease === disease ? boost : 0), i];
    });
    scored.sort((a, b) => b[1] - a[1] || a[2] - b[2]);
    return scored.slice(0, k);
  }
  retrieve(key, symptoms, text = "", k = 5) {
    const q = `${this.kb.diseases[key].name} ` + symptoms.map((s) => this.kb.label(s)).join(" ") + " " + text;
    const hits = this.search(q, k + 3, key);
    const own = hits.filter((h) => h[0].disease === key).sort((a, b) => SECTION_ORDER.indexOf(a[0].section) - SECTION_ORDER.indexOf(b[0].section));
    return [...own, ...hits.filter((h) => h[0].disease !== key)].slice(0, k);
  }
  explain(key, symptoms, text) {
    const d = this.kb.diseases[key], hits = this.retrieve(key, symptoms, text);
    const idx = {}; hits.forEach((h, i) => { if (h[0].disease === key) idx[h[0].section] = i + 1; });
    const parts = [];
    if (idx.overview) parts.push(`${d.description} [${idx.overview}]`);
    if (idx.precautions) parts.push("What helps: " + d.precautions.slice(0, 3).join(" ") + ` [${idx.precautions}]`);
    if (idx.see_doctor) parts.push(`${d.see_doctor} [${idx.see_doctor}]`);
    if (idx.red_flags) parts.push("Watch for: " + d.red_flags.join("; ") + `. [${idx.red_flags}]`);
    return { answer: parts.join("\n\n"), mode: "extractive", retriever: "tfidf",
      sources: hits.map(([c, s], i) => ({ n: i + 1, id: c.id, title: c.title, text: c.text, score: Math.round(s * 1000) / 1000 })) };
  }
}

// ------------------------------------------------------------------ geo (OpenStreetMap)
const SPECIALTY_TERMS = {
  "Infectious Diseases": ["infectious", "general", "internal", "fever"], "General Medicine": ["general", "internal", "family"],
  Pulmonology: ["pulmonology", "chest", "respiratory", "tb"], ENT: ["otolaryngology", "ent", "ear"],
  Ophthalmology: ["ophthalmology", "eye"], Gastroenterology: ["gastroenterology", "gastro", "digestive", "liver"],
  "General Surgery": ["surgery", "general"], Urology: ["urology", "kidney", "nephrology"],
  Cardiology: ["cardiology", "heart", "cardiac"], Endocrinology: ["endocrinology", "diabetes", "thyroid"],
  Hematology: ["haematology", "hematology", "blood"], Neurology: ["neurology", "neuro"],
  Rheumatology: ["rheumatology", "arthritis", "ortho"], Orthopedics: ["orthopaedics", "orthopedics", "ortho", "physiotherapy", "spine"],
  Dermatology: ["dermatology", "skin"], Psychiatry: ["psychiatry", "mental", "counselling"],
};
const RISK_EW = { high: 1.0, medium: 0.5, low: 0.15 };
export function haversineKm([la1, lo1], [la2, lo2]) {
  const r = (x) => (x * Math.PI) / 180;
  const h = Math.sin(r(la2 - la1) / 2) ** 2 + Math.cos(r(la1)) * Math.cos(r(la2)) * Math.sin(r(lo2 - lo1) / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}
export function parseElements(els) {
  return els.map((el) => {
    const t = el.tags || {}, lat = el.lat ?? el.center?.lat, lon = el.lon ?? el.center?.lon, name = t.name || t["name:en"];
    if (lat == null || !name) return null;
    return { name, lat: +lat, lon: +lon, kind: t.amenity || t.healthcare || "clinic", emergency: t.emergency === "yes",
      speciality: t["healthcare:speciality"] || "", phone: !!(t.phone || t["contact:phone"]),
      website: !!(t.website || t["contact:website"]), opening_hours: t.opening_hours || "" };
  }).filter(Boolean);
}
export function rankProviders(providers, origin, specialty = "General Medicine", risk = "low", topN = 5) {
  const terms = SPECIALTY_TERMS[specialty] || ["general"], ew = RISK_EW[risk] ?? 0.3, seen = new Set(), out = [];
  for (const p0 of providers) {
    const key = `${p0.name.toLowerCase()}|${p0.lat.toFixed(3)}|${p0.lon.toFixed(3)}`;
    if (seen.has(key)) continue; seen.add(key);
    const p = { ...p0 };
    p.distance_km = haversineKm(origin, [p.lat, p.lon]);
    const prox = Math.exp(-p.distance_km / 4), hay = `${p.speciality} ${p.name}`.toLowerCase();
    const spec = terms.some((t) => hay.includes(t)) ? 1 : p.kind === "hospital" ? 0.5 : 0.2;
    const em = (p.emergency ? 1 : p.kind === "hospital" ? 0.4 : 0) * ew + (1 - ew) * 0.5;
    const fit = risk === "high" ? (p.kind === "hospital" ? 1 : 0.4) : (p.kind !== "hospital" ? 0.8 : 0.7);
    const q = ((p.opening_hours ? 1 : 0) + (p.phone ? 1 : 0) + (p.website ? 1 : 0)) / 3;
    p.score = 0.4 * prox + 0.25 * spec + 0.15 * em + 0.1 * fit + 0.1 * q;
    const why = [`${p.distance_km.toFixed(1)} km away`];
    if (spec === 1) why.push(`matches ${specialty}`);
    if (p.emergency) why.push("has emergency department");
    why.push(p.kind);
    p.why = why.join(", ");
    p.maps_url = `https://www.openstreetmap.org/?mlat=${p.lat}&mlon=${p.lon}#map=17/${p.lat}/${p.lon}`;
    out.push(p);
  }
  return out.sort((a, b) => b.score - a.score).slice(0, topN);
}
// Public Overpass mirrors that send CORS headers. The main instance is busy at peak times (504/429),
// so each is tried in turn; if all fail, findCare() falls back to the bundled snapshot and says so.
const OVERPASS = ["https://overpass-api.de/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
  "https://overpass.private.coffee/api/interpreter"];
export async function fetchProviders(lat, lon, radius = 6000) {
  const around = `(around:${radius},${lat},${lon})`;
  const q = `[out:json][timeout:20];(nwr["amenity"~"^(hospital|clinic|doctors)$"]${around};` +
            `nwr["healthcare"~"^(hospital|clinic|doctor)$"]${around};);out center tags 150;`;
  let last;
  for (const url of OVERPASS) {
    try {
      const ctl = new AbortController(); const timer = setTimeout(() => ctl.abort(), 25000);
      const r = await fetch(url, { method: "POST", body: new URLSearchParams({ data: q }), signal: ctl.signal });
      clearTimeout(timer);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return parseElements((await r.json()).elements || []);
    } catch (e) { last = e.name === "AbortError" ? new Error("timed out") : e; }
  }
  throw new Error(`the public OpenStreetMap servers are busy right now (${last?.message})`);
}
export async function geocode(place) {
  const r = await fetch(`https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(place)}`);
  const js = await r.json();
  return js.length ? [+js[0].lat, +js[0].lon, js[0].display_name] : null;
}

// ------------------------------------------------------------------ pipeline
export const DISCLAIMER = "NeuroSy-RAG is an educational project, not a medical device. It can be wrong. " +
  "It does not replace a doctor. In an emergency call 108 (India) or your local emergency number.";

const r4 = (x) => Math.round(x * 1e4) / 1e4;
export function createEngine(bundle) {
  const kb = new KB(bundle.kb), extractor = new SymptomExtractor(kb), clf = new Classifier(bundle.model);
  const reasoner = new Reasoner(kb), explainer = new Explainer(kb, bundle.tfidf);
  const snapshot = { center: bundle.osm_snapshot.center, providers: parseElements(bundle.osm_snapshot.elements) };
  const fmt = (c) => ({ ...c, ml_prob: r4(c.ml_prob), kg_support: r4(c.kg_support), kg_explains: r4(c.kg_explains),
    kg_score: r4(c.kg_score), final: r4(c.final) });

  function diagnose(text) {
    const t0 = (globalThis.performance || Date).now();
    const ex = extractor.extract(text);
    const ml = clf.predictProba(ex.present);
    const res = reasoner.reason(ml, ex.present, ex.reported, ex.negated);
    const top = res.abstain ? null : res.candidates[0];
    const sorted = (s) => [...s].sort(cmp);
    return {
      input: text,
      symptoms: { present: sorted(ex.present), reported: sorted(ex.reported), negated: sorted(ex.negated),
        present_labels: sorted(ex.reported).map((s) => kb.label(s)), negated_labels: sorted(ex.negated).map((s) => kb.label(s)),
        duration_days: ex.duration_days, notes: ex.notes, mentions: ex.mentions },
      abstained: res.abstain, abstain_reason: res.abstain_reason,
      predictions: res.candidates.map(fmt), rejected_by_rules: res.rejected.map(fmt),
      ml_only_top: res.ml_top ? kb.diseases[res.ml_top].name : null,
      triage: reasoner.triage(ex.present, top, ex.duration_days),
      explanation: top ? explainer.explain(top.disease, sorted(ex.reported), text) : null,
      top_disease: top ? kb.diseases[top.disease] : null,
      disclaimer: DISCLAIMER,
      latency_ms: Math.round(((globalThis.performance || Date).now() - t0) * 10) / 10,
    };
  }

  async function findCare({ place, lat, lon, specialty = "General Medicine", risk = "medium" } = {}) {
    let label = place || (lat != null ? `${lat.toFixed(4)}, ${lon.toFixed(4)}` : null), source = "OpenStreetMap (live)", providers;
    try {
      if (lat == null && place) {
        const g = await geocode(place);
        if (!g) throw new Error(`could not find "${place}"`);
        [lat, lon, label] = g;
      }
      if (lat == null) throw new Error("no location given");
      providers = await fetchProviders(lat, lon);
      if (!providers.length) providers = await fetchProviders(lat, lon, 15000);
    } catch (e) {
      [lat, lon] = snapshot.center; providers = snapshot.providers;
      label = "SRM Kattankulathur, Chennai (sample)";
      source = e.message === "no location given" ? "bundled OpenStreetMap snapshot"
        : `live search unavailable: ${e.message}. Showing the bundled OpenStreetMap snapshot instead`;
    }
    return { origin: { lat, lon, label }, source, attribution: "Map data © OpenStreetMap contributors, ODbL",
      providers: rankProviders(providers, [lat, lon], specialty, risk) };
  }

  return { kb, diagnose, findCare, meta: bundle.model.meta, nDiseases: Object.keys(kb.diseases).length };
}
