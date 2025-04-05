import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulate_experiment import generate_experiment_data
from src.cuped import apply_cuped
from src.inference_methods import classical_t_test
from src.bayesian_ab import run_bayesian_ab_test

# Cache the data generation to speed up
@st.cache_data
def get_experiment_data(sample_size, treatment_effect, dropout_rate, use_cuped=True):
    df = generate_experiment_data(n_users=sample_size,
                                  treatment_effect=treatment_effect,
                                  dropout_rate=dropout_rate,
                                  stratify=True,
                                  seed=42)
    if use_cuped:
        df = apply_cuped(df)
    return df

# Streamlit UI Setup
st.set_page_config(page_title="Step-by-Step A/B Test Explorer", layout="wide")
st.title("🔍 Step-by-Step A/B Test Impact Explorer")

# Predefined steps
steps = [
    {"sample_size": 1000, "dropout_rate": 0.0, "treatment_effect": 0.02},
    {"sample_size": 5000, "dropout_rate": 0.1, "treatment_effect": 0.02},
    {"sample_size": 10000, "dropout_rate": 0.2, "treatment_effect": 0.02},
    {"sample_size": 20000, "dropout_rate": 0.1, "treatment_effect": 0.05}
]

# Stepper logic with buttons
if 'step' not in st.session_state:
    st.session_state.step = 0

# Buttons to navigate steps
col1, col2 = st.columns([1, 1])
with col1:
    if st.session_state.step > 0:
        if st.button("Previous Step"):
            st.session_state.step -= 1

with col2:
    if st.session_state.step < len(steps) - 1:
        if st.button("Next Step"):
            st.session_state.step += 1

# Get current step values
current = steps[st.session_state.step]
sample_size = current["sample_size"]
dropout_rate = current["dropout_rate"]
treatment_effect = current["treatment_effect"]
use_cuped = True

# Display current step details
st.subheader(f"Step {st.session_state.step + 1}: Sample Size = {sample_size}, Dropout Rate = {dropout_rate}, Treatment Effect = {treatment_effect}")

# Display loading message while processing
with st.spinner('Running the A/B test simulation...'):
    df = get_experiment_data(sample_size, treatment_effect, dropout_rate, use_cuped)
    metric = 'adjusted_metric' if use_cuped else 'post_metric'

    t_stat, p_val = classical_t_test(df, metric_col=metric)
    bayes = run_bayesian_ab_test(df, metric_col=metric)

# Display Results
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Metric Distribution by Group**")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=df, x=metric, hue="group", fill=True, ax=ax1)
    ax1.set_title("Metric Distribution")
    st.pyplot(fig1)

with col2:
    st.markdown("**Summary Statistics**")
    control = df[df.group == 'control'][metric].dropna()
    treatment = df[df.group == 'treatment'][metric].dropna()
    st.write(f"Mean (Control): {control.mean():.4f}")
    st.write(f"Mean (Treatment): {treatment.mean():.4f}")
    st.write(f"T-statistic: {t_stat:.4f}")
    st.write(f"P-value: {p_val:.4f}")
    st.write(f"Bayesian P(Treatment > Control): {bayes['prob_treatment_better']:.4f}")

st.markdown("---")
st.info("Use the 'Next Step' and 'Previous Step' buttons to view results for different experimental setups.")
