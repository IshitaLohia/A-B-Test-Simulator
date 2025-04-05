# dashboard/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulate_experiment import generate_experiment_data
from src.cuped import apply_cuped
from src.inference_methods import classical_t_test
from src.bayesian_ab import run_bayesian_ab_test
from src.uplift_model import model_uplift_effects
from src.evaluation import summarize_results

# Streamlit UI Setup
st.set_page_config(page_title="A/B Test Simulator", page_icon="📊", layout="wide")
st.markdown("""
    <h2 style='text-align: center;'>📊 A/B Test Experimentation Simulator</h2>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("🧪 Experiment Controls")
    st.caption("Use these controls to simulate and visualize A/B test scenarios:")
    n_users = st.slider("👥 Sample Size (Number of Users)", 1000, 50000, 10000, step=1000,
                        help="Total number of users in your experiment.")
    treatment_effect = st.slider("🎯 Treatment Effect (%)", 0.0, 0.1, 0.02, step=0.005,
                                 help="Expected uplift due to treatment.")
    dropout_rate = st.slider("❌ Dropout Rate", 0.0, 0.5, 0.1, step=0.01,
                              help="Proportion of users lost post-assignment.")
    stratify = st.checkbox("📱 Stratified Assignment (by device)", value=True,
                           help="Controls whether random assignment is stratified by device type.")
    seed = st.number_input("🎲 Random Seed", min_value=0, value=42, step=1,
                           help="Set a seed to reproduce your results.")

# Generate and process data
df = generate_experiment_data(
    n_users=n_users,
    treatment_effect=treatment_effect,
    dropout_rate=dropout_rate,
    stratify=stratify,
    seed=seed
)
df = apply_cuped(df)
t_stat, p_val = classical_t_test(df)
bayes_results = run_bayesian_ab_test(df)
uplift_summary = model_uplift_effects(df)
summary = summarize_results(df, t_stat, p_val, bayes_results, uplift_summary)

# Layout Single Screen
st.subheader("📄 Results Summary")
st.text_area("Experiment Output", summary, height=500)

st.subheader("📈 Metric Distributions")
tab1, tab2 = st.tabs(["Raw Post Metric", "CUPED Adjusted"])

with tab1:
    fig1, ax1 = plt.subplots()
    sns.kdeplot(data=df, x="post_metric", hue="group", fill=True, ax=ax1)
    ax1.set_title("Post Metric Distribution by Group")
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots()
    sns.kdeplot(data=df.dropna(subset=['adjusted_metric']), x="adjusted_metric", hue="group", fill=True, ax=ax2)
    ax2.set_title("CUPED Adjusted Metric Distribution")
    st.pyplot(fig2)

# Hide Streamlit footer
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)
