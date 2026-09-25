# Space-Grade Semiconductor Burn-In Anomaly & Drift Prediction System

An end-to-end Machine Learning and Statistical Screening System for Environmental Stress Screening (ESS) Burn-In testing ($125^\circ\text{C}$) of mission-critical electronic components (satellites, aerospace payloads, deep-space probes).

---

## The Core Challenge: The "Latent Defect" Problem

In high-reliability sectors, semiconductor components undergo thermal burn-in (e.g. $125^\circ\text{C}$ for 168 hours). Traditional screening relies on **static parametric limits** (e.g. Leakage Current $I_{\text{leak}} \le 50\,\mu\text{A}$).

However, **latent defects escape static screening**:
1. **The In-Spec Distribution Outlier:** If a wafer lot average is $10\,\mu\text{A}$ ($\sigma = 1.5\,\mu\text{A}$), a part measuring $45\,\mu\text{A}$ passes the static $50\,\mu\text{A}$ limit, but is a **$>17\sigma$ anomaly** harboring fatal crystal or gate oxide defects. In flight, it fails catastrophically.
2. **The Accelerated Drifter (TDDB / Electromigration):** A part starts at a normal $10\,\mu\text{A}$ at $0\text{h}$, but exhibits accelerated non-linear drift between $0\text{h}$ and $24\text{h}$. By $168\text{h}$, it causes payload loss. Waiting 168h for every lot consumes immense thermal chamber capacity.

---

## System Architecture

```
                       [ Physics-Informed Burn-In Data Generator ]
                      (Simulates 0h, 24h, 96h, 168h across Wafer Lots)
                                            │
                                            ▼
                             [ Data Preprocessing & Features ]
                          (Lot normalization, deltas, Arrhenius rate)
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
      [ MODULE A: Dynamic Screening ]                 [ MODULE B: Early Drift Predictor ]
    * Statistical DPAT (AEC-Q001)                   * Gradient Boosted Regressor & Bayesian Ridge
    * Isolation Forest & Robust Mahalanobis         * 0h + 24h inputs -> Forecast 168h value
    * Cost-sensitive threshold (High Recall)        * Safety slope violation -> Early Rejection Flag
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                             [ MODULE C: QA Inspector Explainability ]
                           * SHAP feature attribution per component
                           * Plain-English inspector audit justification card
                                            │
                                            ▼
                         [ Streamlit QA Mission-Assurance Dashboard ]
                           * Interactive lot screening & early triage
                           * Trajectory drift forecast curves
                           * Audit card & SHAP waterfall plots
```

---

## Modules Breakdown

### Module A: Dynamic Outlier Detection (AEC-Q001 DPAT + ML)
* **Statistical DPAT:** Computes robust location (median) and spread (normalized IQR):
  $$\sigma_{\text{robust}} = \frac{Q_3 - Q_1}{1.349}$$
  $$\text{DPAT}_{\text{upper}} = \text{Median} + k_{\sigma} \times \sigma_{\text{robust}}$$
* **Multivariate ML:** Isolation Forest trained on lot-relative Z-scores across leakage, $I_{ddq}$, and propagation delay ($t_{pd}$).
* **Cost-Sensitive Thresholding:** Optimized with $\text{Cost}(FN) = 10 \times \text{Cost}(FP)$ to eliminate fatal field escapes.

### Module B: Time-Series Early Drift Predictor (24h -> 168h)
* **Features:** Engineered degradation rate $\Delta_{24-0}$, percentage drift, and lot-relative drift velocities.
* **Forecast Engine:** LightGBM regressor predicting hidden $V_{168\text{h}}$ values with high accuracy.
* **Safety Slope Rejection:** Calculates drift velocity:
  $$\text{Slope}_{\text{forecast}} = \frac{\widehat{V}_{168\text{h}} - V_{0\text{h}}}{168.0}$$
  If $\text{Slope} > \text{Slope}_{\text{safe}}$ or $\widehat{V}_{168\text{h}} > \text{Limit}_{\text{datasheet}}$, the part is rejected at **$t = 24\text{h}$**, saving **144 chamber hours** per rejected component.

### Module C: QA Inspector Explainability Station
* Generates signed QA Audit Certificates with:
  * Screening verdict: `QUALIFIED FOR SPACE FLIGHT` vs `REJECT / NON-FLIGHT GRADE`
  * Actionable guidance: `PROCEED TO CONFORMAL COATING` vs `SCRAP / DOWNGRADE TO COMMERCIAL`
  * Engineering failure mode diagnosis (TDDB, In-spec lot outlier, Gross defect)
  * Feature attribution risk breakdown (SHAP values).

---

## Quickstart Guide

### 1. Run the Evaluation Benchmark
Runs the complete test suite comparing Static vs Dynamic screening, computing Recall, Cost-Penalized Score, MAE, and chamber hours saved:
```powershell
python evaluate.py
```

### 2. Launch the Web Dashboards

#### Option A: FastAPI Web App (Vercel-Native)
```powershell
python app.py
```
Open `http://localhost:8000` to view the interactive single-page dashboard.

#### Option B: Streamlit Web Dashboard
```powershell
streamlit run streamlit_app.py
```
Open `http://localhost:8501` to explore lot distributions, trajectories, and inspect individual components.

---

## Deployment

* **Deploy to Vercel**: Connect your GitHub repository to [Vercel](https://vercel.com). Vercel will automatically detect `vercel.json` and `app.py` and deploy the serverless web app instantly.
* **Deploy to Streamlit Community Cloud**: Connect the repository to [share.streamlit.io](https://share.streamlit.io) and set the main file path to `streamlit_app.py`.
