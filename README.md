# VeriClaim — AI-Powered Health Insurance Fraud Detection

> **Eduinx AI Product Management Capstone | Team 5 | June 2026**  
> Sultan · Vighnesh · Abhishek

---

## What Is VeriClaim?

VeriClaim is a B2B SaaS engine that scores every health insurance claim for Fraud, Waste & Abuse (FWA) risk using XGBoost + SHAP explainability. India loses ₹8,000–10,000 crore annually to health insurance fraud. VeriClaim stops it at the gate — not after the money is gone.

**Core principle:** VeriClaim never auto-rejects a claim. Every adverse decision is made by a human investigator with a documented, explainable rationale.

---

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vericlaim.streamlit.app)

<img width="982" height="524" alt="image" src="https://github.com/user-attachments/assets/bb3fc4d8-7b8d-4f83-8a1a-552c7626a051" />

<img width="872" height="522" alt="image" src="https://github.com/user-attachments/assets/f808f35f-122a-46f2-8772-e0e2bcd68675" />

<img width="959" height="490" alt="image" src="https://github.com/user-attachments/assets/8bea93cb-7a85-4579-8a9b-7d5db243ae99" />

<img width="901" height="544" alt="image" src="https://github.com/user-attachments/assets/3c640759-016b-437d-a039-cfcf777b4f03" />

---

## The Problem

| Actor | Fraud Pattern | Est. Share of FWA |
|---|---|---|
| Provider / Hospital | Upcoding, phantom procedures, inflated room rent | 60–70% |
| Policyholder | Pre-existing concealment, claim exaggeration | 20–25% |
| Intermediary / Agent | Ghost policies, collusion rings | 10–15% |

Current defences (rule-based threshold systems + manual review) are:
- Gamed — fraudsters stay below thresholds
- Noisy — too many false positives overwhelm investigators
- Too late — fraud caught only after payment

---

## How VeriClaim Works

```
Claim Submitted
      ↓
Feature Extraction (40+ engineered features)
      ↓
┌─────────────────┬──────────────────┐
│   Rules Engine  │   ML Scorer      │
│  (deterministic)│  (XGBoost V0)    │
└─────────────────┴──────────────────┘
      ↓
Risk Score 0–100 + SHAP Explanation
      ↓
Confidence-Based Routing
  ├── Score < 30, Confidence > 80% → Auto-Clear (same day)
  ├── Score 31–70 → Soft Review (investigator, 4 hrs)
  └── Score 71–100 → Full Investigation (senior, 24 hrs)
      ↓
Investigator Decision (HUMAN ALWAYS DECIDES)
      ↓
Decision logged → Retraining pipeline (data flywheel)
```

**Core guardrail:** No path auto-rejects. Auto-apply is used ONLY to clear low-risk claims — never to deny.

---

## MVP Features

| Screen | What It Does |
|---|---|
| 🏠 Investigator Queue | All claims ranked by risk score, colour-coded pills, one-line SHAP reason per claim |
| 🔍 Claim Deep Dive | Risk gauge, SHAP waterfall chart, peer comparison, approve/escalate/override actions |
| 📁 Batch Upload | Upload any CSV → score all claims → download results |
| 📊 Management Dashboard | KPI cards, routing breakdown, high-risk provider table, kill switch, export |

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| ML Model | XGBoost + SMOTE (class imbalance) |
| Explainability | SHAP |
| Data Processing | pandas + numpy |
| Visualisation | plotly |
| Dataset | Kaggle — Healthcare Provider Fraud Detection (558K claims, Medicare) |
| Deployment | Streamlit Community Cloud |

---

## Run Locally

### Step 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/vericlaim.git
cd vericlaim
```

### Step 2 — Download the dataset
Go to: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis

Download and extract all 4 CSV files into a `data/` folder:
```
vericlaim/
└── data/
    ├── Train-1542865627584.csv
    ├── Train_Beneficiarydata-1542865627584.csv
    ├── Train_Inpatientdata-1542865627584.csv
    └── Train_Outpatientdata-1542865627584.csv
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the model (run once, ~3 minutes)
```bash
python train_model.py
```
This creates a `model/` folder with the trained XGBoost model, feature columns, and scored providers.

### Step 5 — Launch the app
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

---

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub (include the `model/` folder after training)
2. Go to https://share.streamlit.io
3. New app → select your repo → select `app.py` → Deploy
4. Live URL in ~2 minutes

---

## Project Structure

```
vericlaim/
├── app.py                  ← Main Streamlit application (4 screens)
├── train_model.py          ← Model training script (run once)
├── requirements.txt        ← Python dependencies
├── runtime.txt             ← Python version for Streamlit Cloud
├── README.md               ← This file
├── .streamlit/
│   └── config.toml         ← Streamlit theme config
├── data/                   ← Kaggle CSV files (download separately)
│   └── .gitkeep
└── model/                  ← Generated after running train_model.py
    ├── vericlaim_model.pkl
    ├── feature_cols.pkl
    ├── metrics.pkl
    └── scored_providers.csv
```

---

## ROI at a Glance

| Metric | Figure |
|---|---|
| Annual claims (mid-sized TPA) | 20 lakh |
| Total claims pool | ₹5,000 crore |
| FWA leakage at 8% | ₹400 crore |
| Additional fraud caught (+25%) | ₹10 crore |
| VeriClaim cost (₹20/claim) | ₹4 crore |
| **Net annual gain** | **₹6 crore** |
| **ROI** | **~150%** |
| **Payback period** | **< 5 months** |

---

## Data Sources

| Dataset | Use | Link |
|---|---|---|
| Kaggle Medicare Fraud Detection | Primary ML training (558K claims, fraud labels) | [kaggle.com/rohitrox](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) |
| IRDAI Annual Report FY25 | Market sizing (₹1.27L Cr GWP) | [irdai.gov.in](https://irdai.gov.in/annual-reports) |
| BCG–Medi Assist FWA Study | 8–10% leakage estimate | Industry literature |
| WHO ICD-10 | Clinical consistency checks | [icd.who.int](https://icd.who.int/browse10) |

---

## Capstone Deliverables Covered

| # | Deliverable | Status |
|---|---|---|
| 1 | Discovery Report — Problem, Personas, JTBD, Market, Competitors | ✅ |
| 2 | AI Strategy — Vision, OKRs, SCORE, Roadmap | ✅ |
| 3 | Data Strategy — Inventory, DRL, Bias, DPDP Compliance | ✅ |
| 4 | Model Selection — XGBoost, Guardrails, Precision/Recall | ✅ |
| 5 | UX Design & HITL — 4 paths, escalation ladder | ✅ |
| 6 | MVP Prototype — Working Streamlit app | ✅ |
| 7 | Monitoring & Drift — Adversarial retraining, thresholds | ✅ |
| 8 | Ethics & Safety — DPDP, never-auto-reject, fairness audits | ✅ |
| 9 | Pricing & GTM — Hybrid pricing, phased rollout, ROI | ✅ |

---

## Ethics & Safety

- **Never auto-reject** — architectural constraint, not a guideline
- **Protected attributes excluded** as model inputs (gender, religion, pincode)
- **Monitored as outcomes** — fairness audits across geography, hospital tier
- **DPDP Act 2023 compliant** — data minimisation, purpose limitation, audit trail
- **Kill switch** — instant revert to full manual mode
- **HITL mandatory** — every adverse decision is human-made and documented

---

## Team

**Team 5 — Eduinx AI Product Management Cohort 2025**

| Member | Role |
|---|---|
| Vighnesh M V | AI PM — Strategy, GTM, Ethics |
| Abhishek | AI PM — Data Strategy, Model Selection |
| Sultan | AI PM — UX Design, Monitoring |

---

## Disclaimer

> MVP metrics are illustrative / simulated for demonstration. The model is trained on US Medicare public data (Kaggle) and proves the method only. Production accuracy for Indian claims requires client-data retraining (Stage 2). This is not a production-ready fraud detection system.

---

*VeriClaim — Team 5 — Eduinx AIPM Capstone — June 2026*
