"""
VeriClaim — Model Training Script
Run this ONCE to train the model and save model.pkl
Command: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

print("=" * 60)
print("VeriClaim — Model Training")
print("=" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────────
print("\n[1/6] Loading data...")

DATA_DIR = "data"

try:
    train_labels = pd.read_csv(os.path.join(DATA_DIR, "Train-1542865627584.csv"))
    beneficiary  = pd.read_csv(os.path.join(DATA_DIR, "Train_Beneficiarydata-1542865627584.csv"))
    inpatient    = pd.read_csv(os.path.join(DATA_DIR, "Train_Inpatientdata-1542865627584.csv"))
    outpatient   = pd.read_csv(os.path.join(DATA_DIR, "Train_Outpatientdata-1542865627584.csv"))
    print("  ✓ All 4 files loaded successfully")
except FileNotFoundError as e:
    print(f"\n  ✗ ERROR: {e}")
    print("  Make sure your CSV files are inside the 'data' folder.")
    print("  Expected files:")
    print("    data/Train-1542865627584.csv")
    print("    data/Train_Beneficiarydata-1542865627584.csv")
    print("    data/Train_Inpatientdata-1542865627584.csv")
    print("    data/Train_Outpatientdata-1542865627584.csv")
    exit(1)

# ── 2. PREPROCESS LABELS ──────────────────────────────────────────
print("\n[2/6] Preprocessing labels...")
train_labels["PotentialFraud"] = (train_labels["PotentialFraud"] == "Yes").astype(int)
print(f"  ✓ Fraud: {train_labels['PotentialFraud'].sum()} | Genuine: {(train_labels['PotentialFraud']==0).sum()}")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────
print("\n[3/6] Engineering features...")

# --- Inpatient features ---
inpatient["ClaimDuration"] = (
    pd.to_datetime(inpatient["DischargeDt"], errors="coerce") -
    pd.to_datetime(inpatient["AdmissionDt"], errors="coerce")
).dt.days.fillna(0)

inpatient["InscClaimAmtReimbursed"] = pd.to_numeric(
    inpatient["InscClaimAmtReimbursed"], errors="coerce").fillna(0)
inpatient["DeductibleAmtPaid"] = pd.to_numeric(
    inpatient["DeductibleAmtPaid"], errors="coerce").fillna(0)
inpatient["IPAnnualReimbursementAmt"] = pd.to_numeric(
    inpatient.get("IPAnnualReimbursementAmt", pd.Series([0]*len(inpatient))),
    errors="coerce").fillna(0)

# Aggregate inpatient by provider
ip_agg = inpatient.groupby("Provider").agg(
    IP_TotalClaims            = ("ClaimID", "count"),
    IP_AvgClaimAmt            = ("InscClaimAmtReimbursed", "mean"),
    IP_TotalClaimAmt          = ("InscClaimAmtReimbursed", "sum"),
    IP_AvgClaimDuration       = ("ClaimDuration", "mean"),
    IP_MaxClaimDuration       = ("ClaimDuration", "max"),
    IP_UniquePatients         = ("BeneID", "nunique"),
    IP_UniqueAdmittingDoctors = ("AttendingPhysician", "nunique"),
    IP_AvgDeductible          = ("DeductibleAmtPaid", "mean"),
).reset_index()

# --- Outpatient features ---
outpatient["InscClaimAmtReimbursed"] = pd.to_numeric(
    outpatient["InscClaimAmtReimbursed"], errors="coerce").fillna(0)
outpatient["DeductibleAmtPaid"] = pd.to_numeric(
    outpatient["DeductibleAmtPaid"], errors="coerce").fillna(0)

op_agg = outpatient.groupby("Provider").agg(
    OP_TotalClaims    = ("ClaimID", "count"),
    OP_AvgClaimAmt    = ("InscClaimAmtReimbursed", "mean"),
    OP_TotalClaimAmt  = ("InscClaimAmtReimbursed", "sum"),
    OP_UniquePatients = ("BeneID", "nunique"),
    OP_AvgDeductible  = ("DeductibleAmtPaid", "mean"),
).reset_index()

# --- Beneficiary features ---
bene_chronic_cols = [c for c in beneficiary.columns if "ChronicCond" in c]
beneficiary["TotalChronicConditions"] = beneficiary[bene_chronic_cols].apply(
    lambda row: (row == 1).sum(), axis=1)
beneficiary["DOB"] = pd.to_datetime(beneficiary["DOB"], errors="coerce")
beneficiary["Age"] = ((pd.Timestamp("2009-12-31") - beneficiary["DOB"]).dt.days / 365).fillna(0)
beneficiary["IPAnnualReimbursementAmt"] = pd.to_numeric(
    beneficiary.get("IPAnnualReimbursementAmt", pd.Series([0]*len(beneficiary))),
    errors="coerce").fillna(0)
beneficiary["OPAnnualReimbursementAmt"] = pd.to_numeric(
    beneficiary.get("OPAnnualReimbursementAmt", pd.Series([0]*len(beneficiary))),
    errors="coerce").fillna(0)

# Map beneficiary features back to inpatient then to provider
ip_with_bene = inpatient.merge(
    beneficiary[["BeneID","TotalChronicConditions","Age","IPAnnualReimbursementAmt","OPAnnualReimbursementAmt"]],
    on="BeneID", how="left"
)
bene_agg = ip_with_bene.groupby("Provider").agg(
    Bene_AvgAge              = ("Age", "mean"),
    Bene_AvgChronicCond      = ("TotalChronicConditions", "mean"),
    Bene_AvgIPReimbursement  = ("IPAnnualReimbursementAmt", "mean"),
    Bene_AvgOPReimbursement  = ("OPAnnualReimbursementAmt", "mean"),
).reset_index()

# --- Merge all features ---
df = train_labels.merge(ip_agg,   on="Provider", how="left")
df = df.merge(op_agg,   on="Provider", how="left")
df = df.merge(bene_agg, on="Provider", how="left")

# Derived ratio features
df["IP_ClaimPerPatient"] = (df["IP_TotalClaims"] / df["IP_UniquePatients"].replace(0,1)).fillna(0)
df["OP_ClaimPerPatient"] = (df["OP_TotalClaims"] / df["OP_UniquePatients"].replace(0,1)).fillna(0)
df["IP_OP_ClaimRatio"]   = (df["IP_TotalClaims"] / (df["OP_TotalClaims"].replace(0,1))).fillna(0)
df["AvgClaimAmtRatio"]   = (df["IP_AvgClaimAmt"]  / (df["OP_AvgClaimAmt"].replace(0,1))).fillna(0)

df = df.fillna(0)
print(f"  ✓ Feature matrix: {df.shape[0]} providers × {df.shape[1]} columns")

# ── 4. TRAIN MODEL ────────────────────────────────────────────────
print("\n[4/6] Training XGBoost model...")

FEATURE_COLS = [
    "IP_TotalClaims","IP_AvgClaimAmt","IP_TotalClaimAmt",
    "IP_AvgClaimDuration","IP_MaxClaimDuration",
    "IP_UniquePatients","IP_UniqueAdmittingDoctors","IP_AvgDeductible",
    "OP_TotalClaims","OP_AvgClaimAmt","OP_TotalClaimAmt",
    "OP_UniquePatients","OP_AvgDeductible",
    "Bene_AvgAge","Bene_AvgChronicCond",
    "Bene_AvgIPReimbursement","Bene_AvgOPReimbursement",
    "IP_ClaimPerPatient","OP_ClaimPerPatient",
    "IP_OP_ClaimRatio","AvgClaimAmtRatio",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

X = df[FEATURE_COLS].values
y = df["PotentialFraud"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE to handle class imbalance
print("  Applying SMOTE for class imbalance...")
sm = SMOTE(random_state=42, k_neighbors=min(5, (y_train==1).sum()-1))
try:
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
except Exception:
    X_train_res, y_train_res = X_train, y_train
    print("  SMOTE skipped (too few minority samples) — using raw data")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=(y_train==0).sum() / max((y_train==1).sum(), 1),
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_res, y_train_res, verbose=False)
print("  ✓ Model trained")

# ── 5. EVALUATE ───────────────────────────────────────────────────
print("\n[5/6] Evaluating model...")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]
f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"  F1 Score : {f1:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Genuine","Fraud"]))

# ── 6. SAVE ARTEFACTS ─────────────────────────────────────────────
print("\n[6/6] Saving model artefacts...")
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/vericlaim_model.pkl")
joblib.dump(FEATURE_COLS, "model/feature_cols.pkl")

metrics = {"f1": round(f1,4), "auc": round(auc,4)}
joblib.dump(metrics, "model/metrics.pkl")

# Save a sample of the processed data for the demo queue
sample_df = df.copy()
sample_df["FraudProbability"] = model.predict_proba(df[FEATURE_COLS].values)[:,1]
sample_df["RiskScore"] = (sample_df["FraudProbability"] * 100).round(1)
sample_df.to_csv("model/scored_providers.csv", index=False)

print("  ✓ model/vericlaim_model.pkl")
print("  ✓ model/feature_cols.pkl")
print("  ✓ model/metrics.pkl")
print("  ✓ model/scored_providers.csv")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE")
print(f"   F1: {f1:.4f}  |  AUC: {auc:.4f}")
print("   Now run:  streamlit run app.py")
print("=" * 60)
