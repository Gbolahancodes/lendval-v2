# CreditIQ Nigeria — Thin-File Credit Scoring Engine

ML-powered credit scoring for Nigerian digital lenders. Scores "thin-file" borrowers
using everyday financial behaviour (airtime, mobile money, bill payments) and explains
every decision in plain English for loan officers and regulators.

---

## What This System Does

| Problem | Solution |
|---|---|
| Millions of Nigerians have no formal credit history | Alt-data scoring from mobile behaviour proxies |
| Black-box models are a CBN regulatory liability | SHAP reason codes on every decision |
| Bias against informal sector workers | Fairlearn audit + equalized-odds monitoring |
| Models go stale after macro shocks | Evidently PSI drift detection |
| Loan officers can't interrogate the model | GPT-4o tool-calling assistant |

---

## Architecture

```
┌────────────────────────────────────────────────────┐
│                   React Frontend                    │
│  Applicant Builder · SHAP Plot · Chat · Fairness   │
└───────────────────┬────────────────────────────────┘
                    │ REST API
┌───────────────────▼────────────────────────────────┐
│              FastAPI Backend (Python)               │
│  /predict  /assistant/chat  /fairness  /drift      │
└───┬──────────────┬─────────────┬──────────────┬────┘
    │              │             │              │
  ONNX         LightGBM       SHAP         OpenAI
  Runtime      Booster       Explainer     GPT-4o
    │              │             │
    └──────────────▼─────────────┘
              ML Pipeline
    ┌─────────────────────────────┐
    │  Home Credit Dataset (Kaggle)│
    │  + Nigerian Alt-Data Inject  │
    │  + Fairlearn Bias Audit      │
    │  + MLflow Experiment Track   │
    │  + Evidently Drift Monitor   │
    └─────────────────────────────┘
```

---

## Prerequisites

- Python 3.11+
- Node.js 20+ and pnpm
- Docker + Docker Compose (for production deployment)
- Kaggle account (free) for dataset download
- OpenAI API key (for the loan officer assistant)

---

## Step 1 — Clone and Install

```bash
git clone <your-repo>
cd creditiq-nigeria

# Frontend
pnpm install

# Backend training environment (heavier, for ML work)
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-train.txt
```

---

## Step 2 — Download the Kaggle Dataset

The Home Credit Default Risk dataset is free and publicly available.

### Option A — Kaggle CLI (recommended)

1. Go to https://www.kaggle.com/account and create an API token
2. Place the downloaded `kaggle.json` in `~/.kaggle/kaggle.json`
3. Accept the competition rules at https://www.kaggle.com/c/home-credit-default-risk

```bash
# From the backend/ directory
mkdir -p data/raw
kaggle competitions download -c home-credit-default-risk -p data/raw/
```

### Option B — Manual download

1. Go to https://www.kaggle.com/competitions/home-credit-default-risk/data
2. Download `application_train.csv` and place it in `backend/data/raw/`

---

## Step 3 — Generate Nigerian Dataset

This script reads `application_train.csv`, rescales amounts to Naira,
assigns Nigerian employment types and states, and injects four synthetic
alternative-data features correlated with the real default label.

```bash
cd backend
python training/generate_nigerian_data.py
```

Output: `backend/data/processed/nigerian_credit_dataset.parquet`

The script prints Pearson correlations for each injected feature so you can
verify the signal is real and statistically documented:

```
Feature                            : r (expected)
airtime_topup_frequency            : r ≈ -0.18
mobile_money_velocity              : r ≈ -0.21
bill_payment_regularity            : r ≈ -0.29
balance_stability_score            : r ≈ -0.24
```

For quick testing on a subset:

```bash
python training/generate_nigerian_data.py --sample 50000
```

---

## Step 4 — Train the Models

```bash
cd backend
python training/train.py
```

This will:
1. Engineer 16 features from the Nigerian dataset
2. Train a logistic regression baseline (for AUC comparison)
3. Train LightGBM with early stopping
4. Compute SHAP values and feature importance
5. Run Fairlearn demographic parity + equalized odds audit
6. Log everything to MLflow (`backend/mlruns/`)
7. Save `models/credit_model.txt` and `models/shap_background.npy`

Expected metrics (full 300k dataset):
- LightGBM ROC-AUC: ~0.76–0.79
- Logistic Regression ROC-AUC: ~0.67–0.70
- AUC lift: ~+0.09–0.12

To skip MLflow (faster):
```bash
python training/train.py --no-mlflow
```

---

## Step 5 — Export to ONNX

```bash
cd backend
python training/export_onnx.py
```

Output: `backend/models/credit_model.onnx`

ONNX enables faster CPU inference in production and removes the LightGBM
dependency from the production Docker image (only `onnxruntime` needed).

---

## Step 6 — Run Drift Simulation (optional)

Simulates a macro economic shock (post-election naira depreciation) and
generates an Evidently HTML drift report.

```bash
cd backend
python training/drift_simulation.py
```

Open `backend/reports/drift_report.html` in a browser to see the interactive report.

---

## Step 7 — Run the Backend Locally

```bash
cd backend
pip install -r requirements.txt   # Lighter, no training deps
export OPENAI_API_KEY=sk-...
export MODEL_DIR=./models
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Test a prediction:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Emeka Nwosu",
    "age": 28,
    "state": "Lagos",
    "gender": "male",
    "employment_type": "gig_worker",
    "monthly_income": 92000,
    "loan_amount": 400000,
    "loan_tenor_months": 9,
    "existing_dti": 0.52,
    "airtime_topup_frequency": 14,
    "mobile_money_velocity": 9,
    "bill_payment_regularity": 67,
    "balance_stability_score": 58
  }'
```

---

## Step 8 — Run the Frontend

```bash
# From the project root
pnpm dev
```

Open http://localhost:5173

The dashboard is a single applicant scorer. When the FastAPI backend is running
it scores each applicant against the real LightGBM/ONNX model; if the backend is
not reachable it automatically falls back to an equivalent in-browser model, so
the app always returns a decision. The result panel labels which model produced
the score.

To point the frontend at the backend, copy `.env.example` to `.env` (or set the
variable inline). Note the `/api/v1` path — that is where the `/predict` route
lives:

```bash
VITE_API_URL=http://localhost:8000/api/v1 pnpm dev
```

---

## Step 9 — Deploy with Docker

### Set up environment variables

```bash
cp .env.example .env
# Edit .env and set:
#   OPENAI_API_KEY=sk-...
```

### Build and run

```bash
docker compose up --build
```

Services:
- **Frontend**: http://localhost:5173
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

---

## MLflow Experiment Tracking

After training, view experiment results:

```bash
cd backend
mlflow ui --port 5001
```

Open http://localhost:5001 to compare runs, view SHAP importance charts,
and download artifacts.

---

## Project Structure

```
creditiq-nigeria/
├── src/                           # React frontend (Vite + Tailwind)
│   ├── App.tsx                    # Main shell — 4 tabs
│   ├── index.css                  # Fonts + design tokens
│   ├── lib/scoring.ts             # Client-side mock scoring engine
│   └── components/
│       ├── ApplicantBuilder.tsx   # Applicant form + risk output
│       ├── RiskGauge.tsx          # SVG gauge + decision badge
│       ├── ShapForceplot.tsx      # SHAP feature attribution bars
│       ├── LoanOfficerChat.tsx    # AI assistant chat UI
│       ├── FairnessTab.tsx        # Bias audit + counterfactuals
│       └── DriftMonitor.tsx       # PSI time series + alerts
│
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app + CORS
│   │   ├── models.py              # Pydantic request/response models
│   │   ├── dependencies.py        # ONNX + explainer loading
│   │   ├── ml/
│   │   │   ├── features.py        # Feature engineering pipeline
│   │   │   ├── explainer.py       # SHAP wrapper + reason codes
│   │   │   └── bias_audit.py      # Fairlearn metrics + PSI
│   │   └── routes/
│   │       ├── predict.py         # POST /predict
│   │       └── assistant.py       # POST /assistant/chat (GPT-4o)
│   ├── training/
│   │   ├── generate_nigerian_data.py  # Kaggle → Nigerian dataset
│   │   ├── train.py                   # LightGBM + SHAP + Fairlearn
│   │   ├── export_onnx.py             # LightGBM → ONNX
│   │   └── drift_simulation.py        # Evidently drift report
│   ├── requirements.txt           # Production deps
│   ├── requirements-train.txt     # Training deps (heavier)
│   └── Dockerfile
│
├── docker-compose.yml
└── README.md
```

---

## API Reference

### `POST /api/v1/predict`

Score a single applicant.

**Request body**: `ApplicantRequest` (see `backend/app/models.py`)

**Response**:
```json
{
  "applicant_id": "uuid",
  "risk_score": 67.3,
  "decision": "DECLINE",
  "default_probability": 0.673,
  "shap_features": [...],
  "reason_codes": [
    "Post-disbursement DTI of 58% exceeds the 45% policy threshold",
    "Monthly repayment burden of 31% of income leaves insufficient buffer",
    "Bill payment regularity of 67% indicates 33% missed/late payments"
  ],
  "confidence_interval": [63.1, 71.5],
  "model_version": "ng_credit_v2.4_lgbm"
}
```

### `POST /api/v1/assistant/chat`

LLM loan officer assistant with tool-calling.

**Request body**:
```json
{
  "messages": [{"role": "user", "content": "Why was applicant #482 declined?"}],
  "applicant_context": { ... }
}
```

### `GET /api/v1/fairness`

Pre-computed fairness audit metrics.

### `GET /api/v1/drift`

Evidently PSI drift report for key features.

---

## Regulatory Compliance

This system is designed to comply with:

- **CBN Consumer Protection Regulations 2022** — every adverse action notice includes 3 plain-English reason codes
- **NDPC Act 2023** — no personally identifiable data is stored; applicant IDs are UUIDs
- **CBN Fair Lending principles** — demographic parity and equalized odds audits run quarterly; results are logged to MLflow

---

## Customisation for Production

### Connect real mobile data

Replace the synthetic alt-data columns in `generate_nigerian_data.py` with real data from:
- **MTN/Airtel API** — airtime top-up frequency, balance queries
- **Interswitch / Paystack webhooks** — bill payment events
- **Opay / PalmPay / Moniepoint API** — mobile money transaction velocity

### Add more employment cohorts

Edit `EMPLOYMENT_ENCODING` in `backend/app/ml/features.py` and retrain.

### Sub-models by segment

For the informal trader segment (highest default rate, most false negatives),
consider training a dedicated LightGBM model on that cohort only,
weighting alt-data features more heavily. Pass `employment_type` as a routing
key in the `/predict` endpoint to dispatch to the right model.

### Integrating with credit bureaus

CRC Credit Bureau and FirstCentral both offer REST APIs. Add bureau score as a
feature in `features.py` and retrain — expect an AUC gain of 0.03–0.06.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | For assistant | GPT-4o API key |
| `MODEL_DIR` | Yes | Path to trained model artifacts |
| `MLFLOW_TRACKING_URI` | Optional | MLflow server URI |
| `VITE_API_URL` | Frontend | Backend API base URL |

---

## Common Issues

**`credit_model.onnx not found`**
Run training steps 3–5 first. The model files are not committed to git.

**`ImportError: No module named 'fairlearn'`**
Install training deps: `pip install -r requirements-train.txt`

**SHAP takes too long**
In `train.py`, reduce the SHAP sample: `X_test.values[:100]` instead of `[:500]`.

**Kaggle download fails**
Make sure you accepted the competition terms on the Kaggle website before running the CLI.

**Frontend shows wrong API URL**
Set `VITE_API_URL=http://localhost:8000` before running `pnpm dev`.

---

## Citation

Dataset: Home Credit Default Risk
https://www.kaggle.com/competitions/home-credit-default-risk

Nigerian alternative-data features are synthetic, generated with
statistically documented correlations to the real default label.
See `training/generate_nigerian_data.py` for the exact generation procedure.
