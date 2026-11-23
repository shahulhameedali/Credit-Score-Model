
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Credit Score" / "data"

# Load cleaned dataset
df = pd.read_csv(DATA_DIR / "clean_credit_data.csv")

target = "default"

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# EMI to income ratio (monthly EMI / monthly income)
X["emi_to_income_ratio"] = X["existing_emi"] / (X["annual_income"] / 12)

# Loan to income ratio (requested loan / annual income)
X["loan_to_income_ratio"] = X["requested_loan_amount"] / X["annual_income"]

# Total debt burden (existing + requested loan)
X["total_debt_burden"] = X["existing_loan_amount"] + X["requested_loan_amount"]

# CIBIL risk tier: 300-550=1 (Poor), 550-700=2 (Fair), 700-900=3 (Good)
def cibil_tier(score):
    if score < 550:
        return 1
    elif score < 700:
        return 2
    else:
        return 3

X["cibil_risk_tier"] = X["cibil_score"].apply(cibil_tier)

# Save engineered dataset
engineered_df = pd.concat([X, y], axis=1)

engineered_df.to_csv(
    DATA_DIR / "engineered_credit_data.csv",
    index=False
)

print("Feature Engineering Completed!")
print("New dataset shape:", engineered_df.shape)
print("\nNew features:")
print([
    "emi_to_income_ratio",
    "loan_to_income_ratio",
    "total_debt_burden",
    "cibil_risk_tier"
])
