"use client";

import { useMemo, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";

type ModelName = "ExtraTreesRegressor" | "CatBoostRegressor" | "SVR";
type ModelPrediction = { pIC50: number; IC50_nM: number };
type ADResult = {
  knn: { distance: number; threshold: number; inside: boolean };
  bounding_box: { outside_feature_count: number; n_features: number; inside: boolean };
  leverage: { value: number; threshold: number; inside: boolean };
  inside_count: number;
  inside_all_three: boolean;
};
type PredictionResult = {
  input_name: string;
  input_format: string;
  canonical_smiles: string;
  status: string;
  predictions: Record<ModelName, ModelPrediction>;
  applicability_domain: ADResult;
};
type Failure = { input_name: string; record?: number | null; reason: string; features?: string[] };
type PredictionResponse = { results: PredictionResult[]; failures: Failure[]; n_parsed: number; n_scored: number };
type ShapFeature = { feature: string; value: number; shap_value: number; direction: string };
type MorganLegend = { feature: string; shap_value: number; direction: string; color: string; atom_indices: number[]; bond_indices: number[] };
type ModelExplanation = {
  base_value: number;
  prediction: number;
  shap_sum: number;
  additivity_error: number;
  features: ShapFeature[];
  top_features: ShapFeature[];
  morgan_svg: string | null;
  morgan_legend: MorganLegend[];
};
type ExplanationResponse = { canonical_smiles: string; models: Partial<Record<Exclude<ModelName, "SVR">, ModelExplanation>> };
type InputMode = "smiles" | "file";

const MODEL_LABELS: Record<ModelName, string> = {
  ExtraTreesRegressor: "ExtraTrees",
  CatBoostRegressor: "CatBoost",
  SVR: "SVR",
};
const MODEL_ORDER: ModelName[] = ["ExtraTreesRegressor", "CatBoostRegressor", "SVR"];
const AD_LABELS = ["5-nearest neighbors", "Training range", "Leverage"] as const;

function apiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL || "").trim().replace(/\/+$/, "");
}

async function readError(response: Response): Promise<string> {
  const raw = await response.text();
  try {
    const parsed = JSON.parse(raw) as { detail?: string | Array<{ msg?: string }> };
    if (typeof parsed.detail === "string") return parsed.detail;
    if (Array.isArray(parsed.detail)) return parsed.detail.map((item) => item.msg || "Request validation failed").join("; ");
  } catch {
    // Keep the plain-text backend response below.
  }
  return raw || `Request failed (${response.status}).`;
}

function csvCell(value: unknown): string {
  let text = value == null ? "" : String(value);
  if (/^[=+@\-]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}

function exportCsv(rows: PredictionResult[], failures: Failure[]) {
  const columns = [
    "input_name", "input_format", "canonical_smiles", "ExtraTrees_pIC50", "ExtraTrees_IC50_nM",
    "CatBoost_pIC50", "CatBoost_IC50_nM", "SVR_pIC50", "SVR_IC50_nM",
    "knn_distance", "knn_threshold", "knn_inside", "features_outside_training_range", "bounding_box_inside",
    "leverage", "leverage_threshold", "leverage_inside", "inside_count_of_3", "inside_all_three", "status",
  ];
  const lines = [columns.map(csvCell).join(",")];
  for (const r of rows) {
    const ad = r.applicability_domain;
    const values = [
      r.input_name, r.input_format, r.canonical_smiles,
      r.predictions.ExtraTreesRegressor?.pIC50, r.predictions.ExtraTreesRegressor?.IC50_nM,
      r.predictions.CatBoostRegressor?.pIC50, r.predictions.CatBoostRegressor?.IC50_nM,
      r.predictions.SVR?.pIC50, r.predictions.SVR?.IC50_nM,
      ad.knn.distance, ad.knn.threshold, ad.knn.inside,
      ad.bounding_box.outside_feature_count, ad.bounding_box.inside,
      ad.leverage.value, ad.leverage.threshold, ad.leverage.inside,
      ad.inside_count, ad.inside_all_three, r.status,
    ];
    lines.push(values.map(csvCell).join(","));
  }
  if (failures.length) {
    lines.push("");
    lines.push(["input_name", "record", "reason"].map(csvCell).join(","));
    for (const f of failures) lines.push([f.input_name, f.record, f.reason].map(csvCell).join(","));
  }
  const blob = new Blob(["\ufeff" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "brd4_predictions_and_applicability_domain.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function fmt(value: number | undefined, digits = 3): string {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

function Waterfall({ explanation }: { explanation: ModelExplanation }) {
  const bars = useMemo(() => {
    const ranked = [...explanation.features].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
    const shown = ranked.slice(0, 12);
    const residual = ranked.slice(12).reduce((sum, row) => sum + row.shap_value, 0);
    const items = [...shown];
    if (ranked.length > shown.length) items.push({ feature: "Other features", value: 0, shap_value: residual, direction: residual >= 0 ? "increases" : "decreases" });
    let cursor = explanation.base_value;
    return items.map((item) => {
      const start = cursor;
      cursor += item.shap_value;
      return { ...item, start, end: cursor };
    });
  }, [explanation]);

  const width = 820;
  const labelW = 190;
  const plotW = 535;
  const rowH = 32;
  const height = 54 + bars.length * rowH;
  const allVals = [explanation.base_value, explanation.prediction, ...bars.flatMap((b) => [b.start, b.end])];
  let min = Math.min(...allVals), max = Math.max(...allVals);
  if (max - min < 0.02) { min -= 0.05; max += 0.05; }
  const x = (v: number) => labelW + ((v - min) / (max - min)) * plotW;

  return (
    <div className="waterfall-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Local SHAP waterfall">
        <line x1={x(explanation.base_value)} x2={x(explanation.base_value)} y1="25" y2={height - 12} className="wf-base" />
        <text x={labelW} y="16" className="wf-axis">Base {fmt(explanation.base_value)}</text>
        <text x={labelW + plotW} y="16" textAnchor="end" className="wf-axis">Prediction {fmt(explanation.prediction)}</text>
        {bars.map((bar, index) => {
          const y = 30 + index * rowH;
          const left = Math.min(x(bar.start), x(bar.end));
          const barWidth = Math.max(2, Math.abs(x(bar.end) - x(bar.start)));
          const positive = bar.shap_value >= 0;
          return (
            <g key={`${bar.feature}-${index}`}>
              <text x={labelW - 10} y={y + 15} textAnchor="end" className="wf-feature">
                <title>{bar.feature}</title>
                {bar.feature.length > 27 ? `${bar.feature.slice(0, 25)}…` : bar.feature}
              </text>
              <rect x={left} y={y + 3} width={barWidth} height="18" rx="4" fill={positive ? "#0b8a75" : "#d95b52"} />
              <text x={labelW + plotW + 9} y={y + 16} className="wf-value">{bar.shap_value >= 0 ? "+" : ""}{fmt(bar.shap_value)}</text>
            </g>
          );
        })}
        <line x1={labelW} x2={labelW + plotW} y1={height - 7} y2={height - 7} className="wf-axis-line" />
        <text x={labelW} y={height - 1} className="wf-axis">{fmt(min)}</text>
        <text x={labelW + plotW} y={height - 1} textAnchor="end" className="wf-axis">{fmt(max)}</text>
      </svg>
      <div className="waterfall-key"><span><i className="positive-dot" /> raises predicted pIC₅₀</span><span><i className="negative-dot" /> lowers predicted pIC₅₀</span></div>
    </div>
  );
}

function DomainRow({ label, inside, value }: { label: string; inside: boolean; value: string }) {
  return (
    <div className="domain-row">
      <div><span className={`status-dot ${inside ? "is-in" : "is-out"}`} />{label}</div>
      <span className="domain-value">{value}</span>
      <span className={`domain-status ${inside ? "is-in" : "is-out"}`}>{inside ? "Inside" : "Outside"}</span>
    </div>
  );
}

export default function Home() {
  const [mode, setMode] = useState<InputMode>("smiles");
  const [smilesText, setSmilesText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [explainTarget, setExplainTarget] = useState<PredictionResult | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [explainModel, setExplainModel] = useState<"ExtraTreesRegressor" | "CatBoostRegressor">("ExtraTreesRegressor");
  const [explainBusy, setExplainBusy] = useState(false);
  const [showFailures, setShowFailures] = useState(true);

  async function predict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(""); setResponse(null); setExplanation(null);
    if (!apiBase()) { setError("Set NEXT_PUBLIC_API_URL in the Vercel project environment variables."); return; }
    if (mode === "smiles" && !smilesText.trim()) { setError("Enter at least one SMILES string."); return; }
    if (mode === "file" && !file) { setError("Choose an SDF or MOL2 file first."); return; }
    setBusy(true); setBusyLabel("Standardizing molecules and calculating the fixed features…");
    try {
      let res: Response;
      if (mode === "smiles") {
        res = await fetch(`${apiBase()}/api/predict/smiles`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles_text: smilesText }), signal: AbortSignal.timeout(180000),
        });
      } else {
        const form = new FormData();
        form.append("file", file as File);
        res = await fetch(`${apiBase()}/api/predict/file`, { method: "POST", body: form, signal: AbortSignal.timeout(180000) });
      }
      if (!res.ok) throw new Error(await readError(res));
      const data = await res.json() as PredictionResponse;
      setResponse(data);
      if (!data.n_scored && data.failures.length) setError("No molecule could be scored. Review the record-level failures below.");
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : "Prediction request failed.";
      setError(message.includes("timed out") ? "The prediction request timed out. Try a smaller batch; the backend may be waking up." : message);
    } finally {
      setBusy(false); setBusyLabel("");
    }
  }

  async function loadExplanation(result: PredictionResult, model: "ExtraTreesRegressor" | "CatBoostRegressor") {
    setError(""); setExplainBusy(true);
    try {
      const res = await fetch(`${apiBase()}/api/explain/smiles`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: result.canonical_smiles, model }), signal: AbortSignal.timeout(180000),
      });
      if (!res.ok) throw new Error(await readError(res));
      const data = await res.json() as ExplanationResponse;
      setExplanation((previous) => previous?.canonical_smiles === data.canonical_smiles
        ? { ...previous, models: { ...previous.models, ...data.models } }
        : data);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Could not create the local explanation.");
    } finally {
      setExplainBusy(false);
    }
  }

  async function explain(result: PredictionResult) {
    const firstModel = "ExtraTreesRegressor" as const;
    setExplainTarget(result);
    setExplanation(null);
    setExplainModel(firstModel);
    await loadExplanation(result, firstModel);
  }

  function selectExplanationModel(model: "ExtraTreesRegressor" | "CatBoostRegressor") {
    setExplainModel(model);
    if (explainTarget && !explanation?.models[model] && !explainBusy) {
      void loadExplanation(explainTarget, model);
    }
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] || null;
    setFile(selected);
  }

  const activeExplanation = explanation?.models[explainModel];

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="BRD4 Predictor home"><span className="brand-mark">B4</span><span>BRD4 <b>Predictor</b></span></a>
        <div className="topbar-right"><span className="live-dot" /> Ligand-based QSAR <span className="topbar-divider" /> Stage 05 fixed features</div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <span className="eyebrow">BRD4 BROMODOMAIN · pIC₅₀ ESTIMATION</span>
          <h1>Explore a molecule’s<br /><em>predicted activity.</em></h1>
          <p>Run three saved QSAR models, check how each molecule compares with the training chemistry, and inspect model explanations for tree models.</p>
          <div className="hero-tags"><span>ExtraTrees</span><span>CatBoost</span><span>SVR</span><span>3 applicability checks</span></div>
        </div>
        <div className="hero-side">
          <div className="hero-stat"><strong>250</strong><span>fixed model features</span></div>
          <div className="hero-stat"><strong>3</strong><span>training-set AD methods</span></div>
          <div className="hero-note"><span className="note-icon">i</span><span>Ligand-based predictions support prioritization; they do not establish binding pose or mechanism.</span></div>
        </div>
      </section>

      <section className="workspace-grid">
        <form className="input-card" onSubmit={predict}>
          <div className="section-heading"><div><span className="step-label">01 / INPUT</span><h2>Submit molecules</h2></div><span className="secure-tag">Processed in memory</span></div>
          <div className="input-tabs" role="tablist" aria-label="Molecule input type">
            <button type="button" className={mode === "smiles" ? "active" : ""} onClick={() => setMode("smiles")}>Paste SMILES</button>
            <button type="button" className={mode === "file" ? "active" : ""} onClick={() => setMode("file")}>Upload SDF / MOL2</button>
          </div>
          {mode === "smiles" ? (
            <label className="field-label" htmlFor="smiles-input">One molecule per line <span>Optional name after a space</span>
              <textarea id="smiles-input" value={smilesText} onChange={(e) => setSmilesText(e.target.value)} placeholder={"CCO ethanol\nCC(=O)O acetic_acid\nCN1C=NC2=C1N=CN2 caffeine"} spellCheck={false} />
            </label>
          ) : (
            <label className={`upload-box ${file ? "has-file" : ""}`} htmlFor="molecule-file">
              <input id="molecule-file" type="file" accept=".sdf,.mol2" onChange={onFileChange} />
              <span className="upload-icon">↑</span><strong>{file ? file.name : "Choose a structure file"}</strong>
              <span>{file ? `${(file.size / 1024).toFixed(1)} KB · ${file.name.toLowerCase().endsWith(".sdf") ? "SDF" : "MOL2"}` : "SDF or MOL2 · up to 20 MB · max 250 molecules"}</span>
            </label>
          )}
          <div className="feature-note"><span>✓</span><p>Uses the saved Morgan settings, Mordred 2D and MolLogP feature list. <b>No 3D descriptors, conformer search or imputation.</b></p></div>
          <button className="primary-button" type="submit" disabled={busy}>{busy ? <><span className="spinner" />{busyLabel}</> : <>Predict activity <span aria-hidden="true">→</span></>}</button>
          {error && <div className="error-banner" role="alert">{error}</div>}
        </form>

        <aside className="guide-card">
          <span className="step-label">READING THE RESULTS</span>
          <h3>Prediction is one part of the answer.</h3>
          <div className="guide-line"><span className="guide-number">1</span><p><b>Compare models.</b> View pIC₅₀ estimates separately; disagreement is a signal to inspect the molecule.</p></div>
          <div className="guide-line"><span className="guide-number">2</span><p><b>Check applicability.</b> See all three training-domain methods and their measured values.</p></div>
          <div className="guide-line"><span className="guide-number">3</span><p><b>Explain on demand.</b> Local SHAP is available for ExtraTrees and CatBoost, not SVR.</p></div>
          <div className="guide-caution"><strong>Interpret carefully</strong><span>SHAP values describe how the fitted model used features. They are not causal evidence of activity.</span></div>
        </aside>
      </section>

      {response && <section className="results-section" aria-live="polite">
        <div className="results-heading">
          <div><span className="step-label">02 / RESULTS</span><h2>Prediction summary</h2><p>{response.n_scored} scored of {response.n_parsed} parsed molecule{response.n_parsed === 1 ? "" : "s"}.</p></div>
          <button className="secondary-button" type="button" onClick={() => exportCsv(response.results, response.failures)} disabled={!response.results.length}>↓ Export CSV</button>
        </div>
        {response.results.map((result, index) => {
          const ad = result.applicability_domain;
          const values = [
            [AD_LABELS[0], ad.knn.inside, `${fmt(ad.knn.distance)} / ${fmt(ad.knn.threshold)}`],
            [AD_LABELS[1], ad.bounding_box.inside, `${ad.bounding_box.outside_feature_count} of ${ad.bounding_box.n_features} features out of range`],
            [AD_LABELS[2], ad.leverage.inside, `${fmt(ad.leverage.value)} / ${fmt(ad.leverage.threshold)}`],
          ] as const;
          return <article className="molecule-card" key={`${result.canonical_smiles}-${index}`}>
            <div className="molecule-head"><div><span className="molecule-index">MOLECULE {String(index + 1).padStart(2, "0")}</span><h3>{result.input_name}</h3><code>{result.canonical_smiles}</code></div>
              <div className={`consensus-badge ${ad.inside_all_three ? "inside" : "caution"}`}><strong>{ad.inside_count}/3</strong><span>{ad.inside_all_three ? "Inside all AD checks" : "Review domain flags"}</span></div>
            </div>
            <div className="prediction-grid">{MODEL_ORDER.map((model) => {
              const prediction = result.predictions[model];
              return <div className="prediction-tile" key={model}><span>{MODEL_LABELS[model]}</span><strong>{fmt(prediction?.pIC50)}</strong><small>pIC₅₀</small><div>≈ {fmt(prediction?.IC50_nM, 1)} nM</div></div>;
            })}</div>
            <div className="ad-panel"><div className="ad-title"><div><span className="step-label">CHEMICAL APPLICABILITY</span><h4>Training-domain checks</h4></div><span className={`domain-overall ${ad.inside_all_three ? "is-in" : "is-out"}`}>{ad.inside_count} of 3 inside</span></div>
              <div className="domain-grid">{values.map(([label, inside, value]) => <DomainRow key={label} label={label} inside={inside} value={value} />)}</div>
              <p className="ad-footnote">These are complementary screening rules, not a probability of prediction accuracy or a prediction interval.</p>
            </div>
            <div className="molecule-actions"><span>Compare the three estimates and review domain flags before using the prediction.</span><button className="explain-button" type="button" onClick={() => explain(result)} disabled={explainBusy}>{explainBusy && explainTarget?.canonical_smiles === result.canonical_smiles ? "Calculating SHAP…" : "Explain with SHAP"} <span>↗</span></button></div>
          </article>;
        })}
        {response.failures.length > 0 && <div className="failures-card"><button type="button" className="failure-toggle" onClick={() => setShowFailures((v) => !v)}><span><span className="warning-mark">!</span> {response.failures.length} record{response.failures.length === 1 ? "" : "s"} not scored</span><span>{showFailures ? "Hide −" : "Review +"}</span></button>
          {showFailures && <div className="failure-list">{response.failures.map((f, i) => <div className="failure-row" key={`${f.input_name}-${i}`}><b>{f.input_name}</b><span>{f.reason}</span></div>)}</div>}
        </div>}
      </section>}

      {explainTarget && <div className="modal-backdrop" role="presentation" onMouseDown={(e) => { if (e.target === e.currentTarget) { setExplainTarget(null); setExplanation(null); } }}>
        <section className="explain-modal" role="dialog" aria-modal="true" aria-labelledby="explain-title">
          <div className="modal-header"><div><span className="step-label">LOCAL MODEL EXPLANATION</span><h2 id="explain-title">{explainTarget.input_name}</h2><code>{explainTarget.canonical_smiles}</code></div><button className="close-button" onClick={() => { setExplainTarget(null); setExplanation(null); }} aria-label="Close explanation">×</button></div>
          {explainBusy && <div className="explain-loading"><span className="spinner dark" />Calculating TreeSHAP contributions and Morgan environments…</div>}
          {explanation && <>
            <div className="explain-model-tabs">{(["ExtraTreesRegressor", "CatBoostRegressor"] as const).map((model) => <button type="button" key={model} className={explainModel === model ? "active" : ""} disabled={explainBusy} onClick={() => selectExplanationModel(model)}>{MODEL_LABELS[model]}</button>)}</div>
            {activeExplanation && <>
              <div className="explain-stat-row"><div><span>SHAP baseline</span><b>{fmt(activeExplanation.base_value)}</b></div><div><span>Model prediction</span><b>{fmt(activeExplanation.prediction)}</b></div><div><span>Additivity check</span><b>{fmt(activeExplanation.additivity_error, 5)}</b></div></div>
              <div className="explain-grid">
                <div className="explain-panel"><div className="panel-title"><span className="step-label">CONTRIBUTION WATERFALL</span><h3>Feature contributions to pIC₅₀</h3><p>Positive contributions raise this model’s prediction; negative contributions lower it.</p></div><Waterfall explanation={activeExplanation} /></div>
                <div className="explain-panel"><div className="panel-title"><span className="step-label">MORGAN ENVIRONMENTS</span><h3>Highlighted atom environments</h3><p>Color legend links mapped Morgan bits to their signed SHAP contribution.</p></div>
                  {activeExplanation.morgan_svg ? <img className="molecule-svg" src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(activeExplanation.morgan_svg)}`} alt="Molecule with Morgan atom environments highlighted" /> : <div className="empty-figure">No Morgan bit environment was selected for this explanation.</div>}
                  <div className="morgan-legend">{activeExplanation.morgan_legend.map((item) => <div className="legend-item" key={item.feature}><i style={{ backgroundColor: item.color }} /><span><b>{item.feature}</b><small>{item.shap_value >= 0 ? "+" : ""}{fmt(item.shap_value)} pIC₅₀ · {item.direction}</small></span></div>)}</div>
                </div>
              </div>
              <div className="top-features"><div className="panel-title"><span className="step-label">LARGEST LOCAL TERMS</span><h3>Feature-level contributions</h3></div><div className="feature-table"><div className="feature-table-head"><span>Feature</span><span>Value</span><span>SHAP (pIC₅₀)</span></div>{activeExplanation.top_features.map((f) => <div className="feature-table-row" key={f.feature}><span>{f.feature}</span><span>{fmt(f.value, 4)}</span><b className={f.shap_value >= 0 ? "shap-positive" : "shap-negative"}>{f.shap_value >= 0 ? "+" : ""}{fmt(f.shap_value, 4)}</b></div>)}</div></div>
              <p className="shap-disclaimer">Model explanation only: these attributions describe the fitted model’s prediction, not causal molecular effects or protein contacts. SVR SHAP is not provided.</p>
            </>}
          </>}
        </section>
      </div>}

      <footer className="page-footer"><span>BRD4 QSAR Predictor</span><span>Ligand-based estimate · Review training-domain flags · No assay result is generated</span></footer>
    </main>
  );
}
