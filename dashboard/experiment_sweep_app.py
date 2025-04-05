# dashboard/experiment_sweep_app.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulate_experiment import generate_experiment_data
from src.cuped import apply_cuped
from src.inference_methods import classical_t_test
from src.bayesian_ab import run_bayesian_ab_test

# Config
st.set_page_config(page_title="Step-by-Step A/B Test Explorer", layout="wide")
st.title("🔍 Step-by-Step A/B Test Impact Explorer")

# Predefined steps
steps = [
    {"sample_size": 1000, "dropout_rate": 0.0, "treatment_effect": 0.02},
    {"sample_size": 5000, "dropout_rate": 0.1, "treatment_effect": 0.02},
    {"sample_size": 10000, "dropout_rate": 0.2, "treatment_effect": 0.02},
    {"sample_size": 20000, "dropout_rate": 0.1, "treatment_effect": 0.05}
]

# Stepper logic
step = st.number_input("Step #", min_value=0, max_value=len(steps) - 1, value=0, step=1, help="Cycle through predefined experimental setups")
current = steps[step]
sample_size = current["sample_size"]
dropout_rate = current["dropout_rate"]
treatment_effect = current["treatment_effect"]
use_cuped = True

# Generate and analyze
df = generate_experiment_data(n_users=sample_size,
                               treatment_effect=treatment_effect,
                               dropout_rate=dropout_rate,
                               stratify=True,
                               seed=42)
if use_cuped:
    df = apply_cuped(df)
    metric = 'adjusted_metric'
else:
    metric = 'post_metric'

t_stat, p_val = classical_t_test(df, metric_col=metric)
bayes = run_bayesian_ab_test(df, metric_col=metric)

st.subheader("📊 Generated Results")
st.write(f"**Step:** {step + 1} / {len(steps)}")
st.write(f"**Sample Size:** {sample_size}")
st.write(f"**Dropout Rate:** {dropout_rate}")
st.write(f"**Treatment Effect:** {treatment_effect}")
st.write(f"**Using CUPED:** {'Yes' if use_cuped else 'No'}")

# Metric Distributions
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Metric Distribution by Group**")
    fig1, ax1 = plt.subplots()
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
st.info("Use the step selector to view how changes in experimental setup affect your results.")
