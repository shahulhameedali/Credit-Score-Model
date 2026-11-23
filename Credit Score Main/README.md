
# Credit Scoring Model – Credit Default Prediction

## Overview
A Machine Learning project that predicts the likelihood of credit card payment default using customer financial and payment history data.

## Objectives
- Predict customer credit default.
- Analyze financial and payment behavior.
- Compare multiple classification algorithms.
- Evaluate model performance using classification metrics.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree
3. Random Forest

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Features
- Exploratory Data Analysis
- Feature Engineering
- Model Comparison
- Credit Default Prediction
- Interactive Streamlit Dashboard

## Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch:

```bash
python -m streamlit run app.py
```

## Run the Vercel Version Locally

```bash
pip install -r requirements.txt
python -m uvicorn api.index:app --host 127.0.0.1 --port 8501
```

Open `http://127.0.0.1:8501`.

## Deploy to Vercel

1. Push the project to GitHub.
2. Import the repository at https://vercel.com/new.
3. Keep the detected framework as **Other**.
4. Deploy without setting a custom build command.

The root route is served by `api/index.py`, and predictions are available at
`POST /api/predict`. The model file must remain in
`Credit Score/model_outputs/random_forest_credit_model.joblib`.

## Disclaimer
This project is developed for educational purposes and is not an official credit approval system.
