import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n_samples = 5000

# Generate synthetic SBI bank loan dataset
# Generate raw arrays first
annual_income = np.random.lognormal(mean=12.2, sigma=0.5, size=n_samples).astype(int)
annual_income = np.clip(annual_income, 200000, 2000000)

cibil_score = np.random.normal(700, 100, n_samples).astype(int)
cibil_score = np.clip(cibil_score, 300, 900)

existing_loan_amount = np.random.lognormal(mean=11.0, sigma=1.2, size=n_samples).astype(int)
existing_loan_amount = np.clip(existing_loan_amount, 0, 1000000)

existing_emi = np.random.lognormal(mean=9.0, sigma=0.8, size=n_samples).astype(int)
existing_emi = np.clip(existing_emi, 0, 50000)

requested_loan_amount = np.random.lognormal(mean=12.0, sigma=0.7, size=n_samples).astype(int)
requested_loan_amount = np.clip(requested_loan_amount, 50000, 2000000)

data = {
    "customer_id": range(100001, 100001 + n_samples),
    "age": np.random.randint(18, 71, n_samples),
    "annual_income": annual_income,
    "employment_type": np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
    "cibil_score": cibil_score,
    "existing_loan_amount": existing_loan_amount,
    "existing_emi": existing_emi,
    "requested_loan_amount": requested_loan_amount,
    "loan_tenure_months": np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], n_samples),
    "dependents": np.random.randint(0, 6, n_samples),
    "education": np.random.randint(1, 6, n_samples),
    "marital_status": np.random.randint(1, 4, n_samples),
}

df = pd.DataFrame(data)

# Generate target (default) based on risk factors
# Higher risk: low CIBIL, high EMI-to-income, high loan-to-income, unemployed, low age
emi_to_income = df["existing_emi"] / (df["annual_income"] / 12)
loan_to_income = df["requested_loan_amount"] / df["annual_income"]

risk_score = (
    (900 - df["cibil_score"]) / 600 * 0.35 +
    emi_to_income.clip(0, 1) * 0.25 +
    loan_to_income.clip(0, 2) / 2 * 0.20 +
    (df["employment_type"] == 2).astype(float) * 0.10 +
    (df["age"] < 25).astype(float) * 0.05 +
    np.random.normal(0, 0.1, n_samples)
)

default_prob = 1 / (1 + np.exp(-(risk_score - 0.5) * 8))
df["default"] = (np.random.random(n_samples) < default_prob).astype(int)

# Ensure some class balance
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['default'].value_counts()}")
print(f"Default rate: {df['default'].mean():.2%}")

# Save raw dataset
output_dir = Path("Credit Score/data")
output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_dir / "credit_data.csv", index=False)
print(f"\nSaved raw dataset to: {output_dir / 'credit_data.csv'}")
print(f"\nColumns: {list(df.columns)}")