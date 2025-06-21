# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Step 2: Load Dataset (replace with your dataset)
# Example dataset structure
data = pd.read_csv("credit_data.csv")  # Should contain columns like income, debts, payment_history, etc.

# Step 3: Data Preprocessing
data.dropna(inplace=True)  # Remove rows with missing values

# Optional: Feature Engineering
data["debt_to_income"] = data["debts"] / data["income"]
data["credit_utilization"] = data["used_credit"] / data["credit_limit"]

# Step 4: Define Features and Target
X = data[["income", "debts", "payment_history", "debt_to_income", "credit_utilization"]]  # customize as per your dataset
y = data["creditworthy"]  # Binary: 1 or 0

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Step 8: Evaluate Models
for name, model in models.items():
    print(f"\nModel: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    if y_proba is not None:
        print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
