from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title='CreditRisk AI | Premium Credit Dashboard',
    page_icon='❓',
    layout='wide',
    initial_sidebar_state='expanded',
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'Credit Score' / 'model_outputs' / 'random_forest_credit_model.joblib'
OUTPUT_DIR = BASE_DIR / 'Credit Score' / 'model_outputs'
DATA_PATH = BASE_DIR / 'Credit Score' / 'data' / 'engineered_credit_data.csv'
