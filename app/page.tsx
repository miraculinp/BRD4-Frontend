"use client";

import { useMemo, useState } from "react";
import type { ChangeEvent } from "react";

type ResultRow = Record<string, string | number | boolean | null>;

type ApiResponse = {
  results: ResultRow[];
  failures: ResultRow[];
  n_parsed: number;
  n_scored: number;
};

type Mode = "smiles" | "file";

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ??
  (process.env.NODE_ENV === "development"
    ? "http://localhost:8000"
    : "")
).replace(/\/+$/, "");

function csvCell(value: unknown): string {
  let text = value == null ? "" : String(value);

  // Prevent spreadsheet formula execution from untrusted text fields.
  if (typeof value === "string" && /^[\s]*[=+\-@\t\r]/.test(text)) {
    text = `'${text}`;
  }

  return `"${text.replaceAll('"', '""')}"`;
}

function exportCsv(rows: ResultRow[], failures: ResultRow[]): void {
  // Explicit annotation fixes the TypeScript index-signature error.
  const all: ResultRow[] = [
    ...rows.map(
      (row): ResultRow => ({ ...row, record_type: "prediction" })
    ),
    ...failures.map(
      (row): ResultRow => ({ ...row, record_type: "not_scored" })
    ),
  ];

  if (!all.length) return;

  const headers: string[] = [
    ...new Set(all.flatMap((row) => Object.keys(row))),
  ];

  const content = [
    headers.map(csvCell).join(","),
    ...all.map((row) =>
      headers.map((key) => csvCell(row[key])).join(",")
    ),
  ].join("\r\n");

  const url = URL.createObjectURL(
    new Blob(["\uFEFF", content], {
      type: "text/csv;charset=utf-8",
    })
  );

  const link = document.createElement("a");
  link.href = url;
  link.download = "brd4_qsar_predictions_and_applicability_domain.csv";
  link.click();
  URL.revokeObjectURL(url);
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("smiles");
  const [smiles, setSmiles] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const canSubmit =
    mode === "smiles" ? smiles.trim().length > 0 : file !== null;

  const rows = response?.results ?? [];
  const failures = response?.failures ?? [];

  const columns = useMemo(
    () => [
      "input_name",
      "canonical_smiles",
      "predicted_pIC50_ExtraTreesRegressor",
      "predicted_pIC50_CatBoostRegressor",
      "predicted_pIC50_SVR",
      "mean_5nn_distance",
      "5nn_threshold",
      "inside_5nn_AD",
      "features_outside_training_range",
      "leverage",
      "leverage_threshold_h_star",
      "inside_leverage_AD",
      "inside_bounding_box_AD",
      "applicability_status",
    ],
    []
  );

  function updateFile(event: ChangeEvent<HTMLInputElement>): void {
    setFile(event.target.files?.[0] ?? null);
    setResponse(null);
    setError("");
  }

  async function submit(): Promise<void> {
    setBusy(true);
    setError("");
    setResponse(null);

    try {
      if (!API_URL) {
        throw new Error(
          "The prediction API URL is not configured for this deployment."
        );
      }

      let responseRaw: Response;

      if (mode === "smiles") {
        responseRaw = await fetch(`${API_URL}/api/predict/smiles`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles_text: smiles }),
        });
      } else {
        if (!file) {
          throw new Error("Choose an SDF or MOL2 file first.");
        }

        const body = new FormData();
        body.append("file", file);

        responseRaw = await fetch(`${API_URL}/api/predict/file`, {
          method: "POST",
          body,
        });
      }

      const data = await responseRaw.json();

      if (!responseRaw.ok) {
        throw new Error(
          data.detail ?? `Request failed (${responseRaw.status}).`
        );
      }

      setResponse(data as ApiResponse);
    } catch (caught: unknown) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Prediction request failed."
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand-mark">B</div>
        <div>
          <div className="brand">BRD4 QSAR</div>
          <div className="brand-sub">Ligand activity predictor</div>
        </div>
        <span className="top-pill">Research tool</span>
      </header>

      <section className="hero">
        <div className="eyebrow">MOLECULE IN · MODEL OUTPUT</div>
        <h1>
          Estimate BRD4 activity.
          <br />
          <em>Check model coverage.</em>
        </h1>
        <p>
          Upload structures or paste SMILES to get pIC50 estimates from three
          saved models, plus three training-based applicability-domain checks.
        </p>
      </section>

      <section className="workspace">
        <div className="input-panel panel">
          <div className="panel-heading">
            <div>
              <span className="step">01</span>
              <h2>Provide molecules</h2>
            </div>
            <span className="small-note">Up to 100 per request</span>
          </div>

          <div className="tabs" role="tablist">
            <button
              className={mode === "smiles" ? "tab active" : "tab"}
              onClick={() => {
                setMode("smiles");
                setResponse(null);
                setError("");
              }}
            >
              Paste SMILES
            </button>
            <button
              className={mode === "file" ? "tab active" : "tab"}
              onClick={() => {
                setMode("file");
                setResponse(null);
                setError("");
              }}
            >
              Upload SDF / MOL2
            </button>
          </div>

          {mode === "smiles" ? (
            <>
              <label className="field-label" htmlFor="smiles">
                One molecule per line
              </label>
              <textarea
                id="smiles"
                value={smiles}
                onChange={(event) => setSmiles(event.target.value)}
                placeholder={"CCO Ethanol\nCC(=O)O Acetic_acid"}
              />
              <p className="hint">
                Put an optional molecule name after each SMILES, separated by a
                space.
              </p>
            </>
          ) : (
            <label className="dropzone">
              <input
                type="file"
                accept=".sdf,.mol2"
                onChange={updateFile}
              />
              <span className="upload-icon">↑</span>
              <strong>{file ? file.name : "Choose a structure file"}</strong>
              <span>
                {file
                  ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
                  : "SDF or MOL2 · maximum 20 MB"}
              </span>
            </label>
          )}

          <button
            className="primary"
            onClick={submit}
            disabled={!canSubmit || busy}
          >
            {busy ? (
              <>
                <span className="spinner" />
                Calculating features and predictions…
              </>
            ) : (
              "Predict activity and check domain →"
            )}
          </button>

          {error && <div className="error">{error}</div>}

          <div className="feature-note">
            <b>Fixed model input</b>
            <span>250 features · 103 Morgan · 146 Mordred 2D · MolLogP</span>
            <span>No 3D descriptors, conformer generation, or imputation.</span>
          </div>
        </div>

        <div className="method-panel panel">
          <div className="panel-heading">
            <div>
              <span className="step">02</span>
              <h2>Applicability domain</h2>
            </div>
          </div>

          <p className="method-intro">
            Each molecule is checked against the training compounds using the
            same three methods used in the study.
          </p>

          <div className="method">
            <span className="method-number">A</span>
            <div>
              <b>5-nearest neighbours</b>
              <p>Standardized descriptor distance vs. training mean + 2 SD.</p>
            </div>
          </div>

          <div className="method">
            <span className="method-number">B</span>
            <div>
              <b>Feature range</b>
              <p>
                Checks whether any of the 250 features exceed training min/max.
              </p>
            </div>
          </div>

          <div className="method">
            <span className="method-number">C</span>
            <div>
              <b>Leverage</b>
              <p>Hat value compared with h* = 3p/n.</p>
            </div>
          </div>

          <div className="method-foot">
            “Inside” means the molecule passes all three checks. It is not a
            probability or a guarantee of accuracy.
          </div>
        </div>
      </section>

      {response && (
        <section className="results panel">
          <div className="results-head">
            <div>
              <span className="step">03</span>
              <h2>Prediction results</h2>
              <p>
                {response.n_scored} scored of {response.n_parsed} parsed
                molecules
              </p>
            </div>
            <button
              className="secondary"
              disabled={!rows.length && !failures.length}
              onClick={() => exportCsv(rows, failures)}
            >
              ↓ Export CSV
            </button>
          </div>

          {rows.length > 0 ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {columns.map((column) => (
                      <th key={column}>{column.replaceAll("_", " ")}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, index) => (
                    <tr key={`${row.input_name}-${index}`}>
                      {columns.map((column) => (
                        <td key={column}>
                          {column === "applicability_status" ? (
                            <span
                              className={
                                row.inside_all_three_AD
                                  ? "status inside"
                                  : "status outside"
                              }
                            >
                              {String(row[column] ?? "")}
                            </span>
                          ) : typeof row[column] === "number" ? (
                            Number(row[column]).toFixed(3)
                          ) : (
                            String(row[column] ?? "—")
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="empty">No valid molecules were available to score.</p>
          )}

          {failures.length > 0 && (
            <details className="failures">
              <summary>{failures.length} record(s) not scored</summary>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Molecule</th>
                      <th>Record</th>
                      <th>Reason</th>
                      <th>Feature details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {failures.map((row, index) => (
                      <tr key={index}>
                        <td>{String(row.input_name ?? "—")}</td>
                        <td>{String(row.record ?? "—")}</td>
                        <td>{String(row.reason ?? "—")}</td>
                        <td>{String(row.feature_issue ?? "—")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </section>
      )}

      <footer>
        <span>BRD4 ligand-based QSAR</span>
        <span>
          Predicted pIC50 is computational; applicability-domain status reflects
          training-space coverage.
        </span>
      </footer>
    </main>
  );
}