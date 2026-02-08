import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. LOAD DATA
DATA_PATH = "training_feature_processed_data.csv"
df = pd.read_csv(DATA_PATH)

# 2. FEATURE SELECTION
# We select the engineered features that represent the "Profile" of the user
features = [
    'debt_to_income_ratio', 
    'spend_to_income', 
    'life_stability_score_adj',
    'psych_index',
    'conscientiousness_score', 
    'impulsivity_score', 
    'integrity_score', 
    'locus_of_control_score', 
    'risk_appetite_score'
]

X = df[features]
y = df['final_risk_label']

# 3. LABEL ENCODING
# Mapping categorical labels to numeric values
label_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Very High': 3
}
y_encoded = y.map(label_mapping)

# 4. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training on {len(X_train)} samples...")

# 5. INITIALIZE AND TRAIN MODEL
# Using Random Forest to capture interactions between Psychometrics and Finances
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced' # Handles any remaining skew in the distribution
)

model.fit(X_train, y_train)

# 6. EVALUATION
y_pred = model.predict(X_test)
print("\n--- Model Performance Report ---")
print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys())[:len(np.unique(y_test))]))

# Feature Importance
print("\n--- Top Features Influencing Risk ---")
importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
print(importance.sort_values(by='Importance', ascending=False))

# 7. SAVE MODEL AND MAPPING
model_filename = 'risk_scoring_model.joblib'
joblib.dump(model, model_filename)

# Save the label mapping separately so your API can use it
mapping_filename = 'label_mapping.joblib'
joblib.dump(label_mapping, mapping_filename)

print(f"\n✓ Success! Model saved as {model_filename}")
print(f"✓ Success! Label mapping saved as {mapping_filename}")
