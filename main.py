import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 1. Initialize FastAPI
app = FastAPI(title="Credit Decisioning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Input Schema (Including Q1-Q15 as integers 1, 2, 3)
class CustomerRequest(BaseModel):
    age: int
    employment_status: str
    household_dependents: int
    marital_status: str
    city: str
    monthly_income: float
    credit_history_type: str
    Total_Debits: float
    Total_Credits: float
    outstanding_liabilities: float
    loan_amount: float
    loan_purpose: str
    # Psychometric Inputs (Numeric 1, 2, or 3)
    Q1: int; Q2: int; Q3: int; Q4: int; Q5: int
    Q6: int; Q7: int; Q8: int; Q9: int; Q10: int
    Q11: int; Q12: int; Q13: int; Q14: int; Q15: int

# 3. Load Model Artifacts
try:
    # Note: Using the model we just trained in the previous step
    model = joblib.load("risk_scoring_model.joblib")
    label_mapping = joblib.load("label_mapping.joblib")
    print("✓ Hybrid ML Model loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    label_mapping = None

# --- Helper Functions for Feature Engineering ---

def age_score(age):
    if age < 22: return 0.1
    elif age <= 25: return 0.4
    elif age <= 30: return 0.7
    elif age <= 35: return 1.0
    elif age <= 55: return 0.6
    else: return 0.5

def dependent_score(n):
    if n == 0: return 1.0
    elif n <= 2: return 0.7
    elif n <= 4: return 0.5
    else: return 0.3

def city_score(city):
    tier1 = ['Karachi', 'Lahore', 'Islamabad']
    if city in tier1: return 1.0
    return 0.5

def squash(x, midpoint=0.75, steepness=6):
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

# Psychometric Mapping (1=A, 2=B, 3=C)
psych_mapping = {
    'Q1': {1: 0.3, 2: 0.7, 3: 1.0}, 'Q2': {1: 1.0, 2: 0.5, 3: 0.0},
    'Q3': {1: 1.0, 2: 0.6, 3: 0.2}, 'Q4': {1: 1.0, 2: 0.5, 3: 0.1},
    'Q5': {1: 0.2, 2: 0.6, 3: 1.0}, 'Q6': {1: 1.0, 2: 0.5, 3: 0.0},
    'Q7': {1: 0.1, 2: 1.0, 3: 0.5}, 'Q8': {1: 1.0, 2: 0.4, 3: 0.0},
    'Q9': {1: 1.0, 2: 0.5, 3: 0.0}, 'Q10': {1: 1.0, 2: 0.1, 3: 0.4},
    'Q11': {1: 1.0, 2: 0.6, 3: 0.1}, 'Q12': {1: 1.0, 2: 0.7, 3: 0.3},
    'Q13': {1: 1.0, 2: 0.5, 3: 0.0}, 'Q14': {1: 1.0, 2: 0.6, 3: 0.1},
    'Q15': {1: 1.0, 2: 0.7, 3: 0.2}
}

# 4. API Endpoints
@app.get("/")
def health():
    return {"status": "online", "model_ready": model is not None}

@app.post("/predict")
def predict(request: CustomerRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert Request to Dict and then DataFrame
        data = request.dict()
        df = pd.DataFrame([data])

        # A. Financial Engineering
        yearly_inc = data['monthly_income'] * 12
        dti = data['outstanding_liabilities'] / (yearly_inc + 1)
        new_dti = (data['outstanding_liabilities'] + data['loan_amount']) / (yearly_inc + 1)
        spend_ratio = data['Total_Debits'] / (data['Total_Credits'] + 1)

        # B. Life Stability Engineering
        emp_map = {'Salaried': 1.0, 'Pensioner': 0.9, 'Self-Employed': 0.5}
        stab_base = (
            0.30 * age_score(data['age']) +
            0.40 * emp_map.get(data['employment_status'], 0.5) +
            0.20 * dependent_score(data['household_dependents']) +
            0.10 * city_score(data['city'])
        )
        life_stab_adj = squash(stab_base)

        # C. Psychometric Engineering
        q_scores = []
        for i in range(1, 16):
            q_val = data[f'Q{i}']
            q_score = psych_mapping[f'Q{i}'].get(q_val, 0.5)
            q_scores.append(q_score)
        
        # Pillar Calculations
        c_score = np.mean([q_scores[0], q_scores[1], q_scores[13], q_scores[14]])
        i_score = np.mean([q_scores[4], q_scores[5], q_scores[6]])
        int_score = np.mean([q_scores[7], q_scores[8], q_scores[9]])
        loc_score = np.mean([q_scores[2], q_scores[3]])
        risk_app_score = np.mean([q_scores[10], q_scores[11], q_scores[12]])
        psych_idx = np.mean(q_scores)

        # --- GATEKEEPER RULES ---
        if data['age'] < 22:
            return {"Decision": "Decline", "Reason": "Underage"}
        if dti > 10.0:
            return {"Decision": "Decline", "Reason": "Existing Debt too High"}

        # --- ML INFERENCE ---
        # Must match the 'features' list from train.py exactly
        features_for_model = np.array([[
            dti, spend_ratio, life_stab_adj, psych_idx,
            c_score, i_score, int_score, loc_score, risk_app_score
        ]])

        pred_idx = model.predict(features_for_model)[0]
        probs = model.predict_proba(features_for_model)[0]
        
        # Inverse mapping to get string label
        inv_map = {v: k for k, v in label_mapping.items()}
        risk_label = inv_map.get(pred_idx, "Unknown")

        # Business Logic Decision
        decision = "Approve"
        if risk_label in ["High", "Very High"] or new_dti > 1.5:
            decision = "Review" if risk_label == "High" else "Decline"

        return {
            "Risk_Label": risk_label,
            "Credit_Score": int(300 + (550 * (1 - probs[-1]))), # Simple score based on High Risk Prob
            "Decision": decision,
            "DTI": round(new_dti, 2),
            "Psychometric_Stability": round(psych_idx, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
