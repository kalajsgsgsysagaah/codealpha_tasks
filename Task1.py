import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Step 1: Create Synthetic Dataset
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    "income": np.random.normal(50000, 15000, n_samples),
    "debts": np.random.normal(15000, 5000, n_samples),
    "payment_history": np.random.randint(0, 2, n_samples),  # 0 = bad, 1 = good
    "used_credit": np.random.uniform(1000, 15000, n_samples),
    "credit_limit": np.random.uniform(5000, 20000, n_samples)
})

# Target: creditworthy (1 = yes, 0 = no), based on some rule
data["creditworthy"] = (
    (data["income"] > 40000) &
    (data["debts"] < 20000) &
    (data["payment_history"] == 1)
).astype(int)

# Step 2: Feature Engineering
data["debt_to_income"] = data["debts"] / data["income"]
data["credit_utilization"] = data["used_credit"] / data["credit_limit"]

# Step 3: Define Features and Target
X = data[["income", "debts", "payment_history", "debt_to_income", "credit_utilization"]]
y = data["creditworthy"]

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 5: Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Model Training and Evaluation
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

