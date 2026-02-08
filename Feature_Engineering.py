import pandas as pd
import numpy as np
import os

# Define your file path
DATA_PATH = "training_feature_processed_data.csv"

def prepare_environment(file_path):
    if os.path.exists(file_path):
        print(f"Found existing data at {file_path}. Deleting to prevent skew...")
        os.remove(file_path)
    else:
        print("No existing data found. Starting fresh.")

# Execute the cleanup
prepare_environment(DATA_PATH)

# 1. LOAD DATA
try:
    # Loading the raw data
    df = pd.read_csv('training_data.csv', index_col=False)
    print("✓ Data loaded successfully. Shape:", df.shape)
except FileNotFoundError:
    print("Error: CSV file not found. Check the filename in your repository.")
    exit()

# 2. RATIO CALCULATIONS
df['yearly_income'] = df['monthly_income'] * 12

# Debt to Income Ratio (DTI)
df['debt_to_income_ratio'] = df['outstanding_liabilities'] / (df['yearly_income'] + 1)

# Spend to Income Ratio 
df['spend_to_income'] = df['Total_Debits'] / (df['Total_Credits'] + 1)

# 3. SCORING FUNCTIONS
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
    tier2 = ['Faisalabad', 'Multan', 'Peshawar']
    if city in tier1: return 1.0
    elif city in tier2: return 0.8
    else: return 0.4

def instability_penalty(row):
    penalty = 0
    if row['age'] < 30 and row['household_dependents'] >= 3:
        penalty += 0.10
    if row['employment_status'] in ['Self-Employed', 'Pensioner'] and row['household_dependents'] >= 4:
        penalty += 0.10
    if row['age'] > 55 and row['employment_status'] not in ['Salaried', 'Pensioner']:
        penalty += 0.05
    return penalty

def squash(x, midpoint=0.75, steepness=6):
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

# --- NEW: PSYCHOMETRIC SCORING LOGIC ---
# Mapping 1.0(a), 2.0(b), 3.0(c) to Risk Scores (1.0 is lowest risk)
psych_mapping = {
    'Q1': {1.0: 0.2, 2.0: 0.7, 3.0: 1.0},
    'Q2': {1.0: 1.0, 2.0: 0.5, 3.0: 0.0},
    'Q3': {1.0: 1.0, 2.0: 0.6, 3.0: 0.2},
    'Q4': {1.0: 1.0, 2.0: 0.5, 3.0: 0.1},
    'Q5': {1.0: 0.2, 2.0: 0.6, 3.0: 1.0},
    'Q6': {1.0: 1.0, 2.0: 0.5, 3.0: 0.0},
    'Q7': {1.0: 0.1, 2.0: 1.0, 3.0: 0.5},
    'Q8': {1.0: 1.0, 2.0: 0.4, 3.0: 0.0},
    'Q9': {1.0: 1.0, 2.0: 0.5, 3.0: 0.0},
    'Q10': {1.0: 1.0, 2.0: 0.1, 3.0: 0.4},
    'Q11': {1.0: 1.0, 2.0: 0.6, 3.0: 0.1},
    'Q12': {1.0: 1.0, 2.0: 0.7, 3.0: 0.3},
    'Q13': {1.0: 1.0, 2.0: 0.5, 3.0: 0.0},
    'Q14': {1.0: 1.0, 2.0: 0.6, 3.0: 0.1},
    'Q15': {1.0: 1.0, 2.0: 0.7, 3.0: 0.2}
}

# Apply mapping to create individual Q scores
for q in range(1, 16):
    col = f'Q{q}'
    df[f'{col}_score'] = df[col].map(psych_mapping[col])

# Calculate Pillar Averages
df['conscientiousness_score'] = df[['Q1_score', 'Q2_score', 'Q14_score', 'Q15_score']].mean(axis=1)
df['impulsivity_score'] = df[['Q5_score', 'Q6_score', 'Q7_score']].mean(axis=1)
df['integrity_score'] = df[['Q8_score', 'Q9_score', 'Q10_score']].mean(axis=1)
df['locus_of_control_score'] = df[['Q3_score', 'Q4_score']].mean(axis=1)
df['risk_appetite_score'] = df[['Q11_score', 'Q12_score', 'Q13_score']].mean(axis=1)

# Combined Psychometric Index (Neutral 0.5 for Thick/Thin File users)
df['psych_index'] = df[['conscientiousness_score', 'impulsivity_score', 'integrity_score', 
                        'locus_of_control_score', 'risk_appetite_score']].mean(axis=1).fillna(0.5)

# 4. APPLY LIFE STABILITY SCORING
employment_map = {'Salaried': 1.0, 'Pensioner': 0.9, 'Self-Employed': 0.5}

base_score = (
    0.30 * df['age'].apply(age_score) +
    0.40 * df['employment_status'].map(employment_map).fillna(0.5) +
    0.20 * df['household_dependents'].apply(dependent_score) +
    0.05 * df['marital_status'].map({'Married': 1.0, 'Single': 0.8}).fillna(0.8) +
    0.10 * df['city'].apply(city_score)
)

df['life_stability_score'] = (base_score - df.apply(instability_penalty, axis=1)).clip(0, 1)

# Normalization
df['life_stability_score_adj'] = squash(df['life_stability_score'])
min_val, max_val = df['life_stability_score_adj'].min(), df['life_stability_score_adj'].max()
df['life_stability_score_adj'] = (df['life_stability_score_adj'] - min_val) / (max_val - min_val)

# 5. FINAL RISK MODELING (Integrated Psychometrics)
# Adjusted Weights: 30% DTI, 25% Spend Ratio, 20% Life Stability, 25% Psychometric
df['base_risk_score'] = (
    0.30 * df['debt_to_income_ratio'].clip(0, 5) + 
    0.25 * df['spend_to_income'].clip(0, 2) + 
    0.20 * (1 - df['life_stability_score_adj']) +
    0.25 * (1 - df['psych_index'])
)

def final_risk_label(score):
    if score >= 2.4: return 'Very High'
    elif score >= 1.5: return 'High'
    elif score >= 1.0: return 'Medium'
    else: return 'Low'

df['final_risk_label'] = df['base_risk_score'].apply(final_risk_label)

print("\n--- Risk Label Distribution ---")
print(df['final_risk_label'].value_counts())

# 6. Save the processed data
output_filename = 'training_feature_processed_data.csv'

try:
    if os.path.exists(output_filename):
        os.remove(output_filename)
    
    df.to_csv(output_filename, index=False)
    print(f"\n✓ Success! File saved as {output_filename}")

except PermissionError:
    print(f"✗ Error: Please close '{output_filename}' and try again.")
except Exception as e:
    print(f"✗ An unexpected error occurred: {e}")
