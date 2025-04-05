import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
from src.simulate_experiment import generate_experiment_data
from src.cuped import apply_cuped
from src.inference_methods import classical_t_test
from src.bayesian_ab import run_bayesian_ab_test
from src.uplift_model import model_uplift_effects
from src.evaluation import summarize_results

st.set_page_config(page_title="A/B Test Simulator", layout="wide")
st.title("📊 Meta-Style A/B Testing Simulator")

with st.sidebar:
    st.header("Experiment Controls")
    n_users = st.slider("Sample Size", min_value=500, max_value=50000, value=10000, step=500)
    treatment_effect = st.slider("Treatment Effect (%)", min_value=0.0, max_value=0.1, value=0.02, step=0.005)
    dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    stratify = st.checkbox("Stratify by Device", value=True)
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1)

# Run simulation and analysis
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

# Show summary
summary = summarize_results(df, t_stat, p_val, bayes_results, uplift_summary)
st.text_area("Results Summary", summary, height=500)

# Visualize distributions
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("Metric Distributions")
fig, ax = plt.subplots()
sns.kdeplot(data=df, x="post_metric", hue="group", fill=True, ax=ax)
ax.set_title("Post Metric Distribution by Group")
st.pyplot(fig)

# CUPED comparison
st.subheader("CUPED Adjusted Metric")
fig2, ax2 = plt.subplots()
sns.kdeplot(data=df.dropna(subset=['adjusted_metric']), x="adjusted_metric", hue="group", fill=True, ax=ax2)
ax2.set_title("CUPED Adjusted Metric Distribution")
st.pyplot(fig2)
