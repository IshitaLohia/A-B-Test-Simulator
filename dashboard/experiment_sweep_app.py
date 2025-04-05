import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

# Dark mode toggle
mode = st.sidebar.radio("🌓 Select Theme", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("""
        <style>
            body {background-color: #111; color: #f0f0f0;}
            .block-container {background-color: #1c1c1c; color: #f0f0f0;}
        </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("""
    <h2 style='text-align: center; color: #333; margin-top: 1rem;'>📊 A/B Test Experimentation Simulator</h2>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("🧪 Experiment Controls")
    st.caption("Use these controls to simulate and visualize A/B test scenarios:")
    n_users = st.slider("👥 Sample Size (Number of Users)", 1000, 50000, 10000, step=1000)
    treatment_effect = st.slider("🎯 Treatment Effect (%)", 0.0, 0.1, 0.02, step=0.005)
    dropout_rate = st.slider("❌ Dropout Rate", 0.0, 0.5, 0.1, step=0.01)
    stratify = st.checkbox("📱 Stratified Assignment (by device)", value=True)
    seed = st.number_input("🎲 Random Seed", min_value=0, value=42, step=1)

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

# Simulation Results Logging
if 'simulations' not in st.session_state:
    st.session_state.simulations = []

st.session_state.simulations.append({
    "sample_size": n_users,
    "dropout_rate": dropout_rate,
    "treatment_effect": treatment_effect,
    "p_value": p_val,
    "bayes_prob_treatment_better": bayes_results['prob_treatment_better'],
    "summary": summary
})

# Layout Single Screen
st.subheader("📄 Results Summary")
st.text_area("Experiment Output", summary, height=500)

# Distributions Side-by-Side
st.subheader("📈 Metric Distributions with Confidence Intervals")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Raw Post Metric**")
    fig1, ax1 = plt.subplots()
    sns.kdeplot(data=df, x="post_metric", hue="group", fill=True, ax=ax1)
    mean_post = df.groupby("group")["post_metric"].mean()
    std_post = df.groupby("group")["post_metric"].std()
    for g in ["control", "treatment"]:
        ax1.axvline(mean_post[g], linestyle="--", label=f"{g} mean")
        ax1.axvline(mean_post[g] - std_post[g], color="gray", linestyle=":", alpha=0.5)
        ax1.axvline(mean_post[g] + std_post[g], color="gray", linestyle=":", alpha=0.5)
    ax1.set_title("Post Metric Distribution by Group")
    st.pyplot(fig1)

with col2:
    st.markdown("**CUPED Adjusted Metric**")
    fig2, ax2 = plt.subplots()
    adjusted = df.dropna(subset=['adjusted_metric'])
    sns.kdeplot(data=adjusted, x="adjusted_metric", hue="group", fill=True, ax=ax2)
    mean_adj = adjusted.groupby("group")["adjusted_metric"].mean()
    std_adj = adjusted.groupby("group")["adjusted_metric"].std()
    for g in ["control", "treatment"]:
        ax2.axvline(mean_adj[g], linestyle="--", label=f"{g} mean")
        ax2.axvline(mean_adj[g] - std_adj[g], color="gray", linestyle=":", alpha=0.5)
        ax2.axvline(mean_adj[g] + std_adj[g], color="gray", linestyle=":", alpha=0.5)
    ax2.set_title("CUPED Adjusted Metric Distribution")
    st.pyplot(fig2)

# Display Recorded Simulations
st.subheader("📋 Recorded Simulations")
simulation_df = pd.DataFrame(st.session_state.simulations)
st.dataframe(simulation_df)

# Optional Power Curve
with st.expander("📉 Show Power Curve Simulation"):
    import statsmodels.stats.power as smp
    st.markdown("Estimate the statistical power for detecting a given effect size.")
    effect_size = treatment_effect / df["post_metric"].std()
    sample_sizes = np.arange(500, 10001, 500)
    power = [smp.TTestIndPower().power(effect_size=effect_size, nobs1=n, alpha=0.05) for n in sample_sizes]
    fig3, ax3 = plt.subplots()
    ax3.plot(sample_sizes, power)
    ax3.axhline(0.8, color="red", linestyle="--")
    ax3.set_xlabel("Sample Size")
    ax3.set_ylabel("Power")
    ax3.set_title("Power Curve vs Sample Size")
    st.pyplot(fig3)

# Hide Streamlit footer
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)
