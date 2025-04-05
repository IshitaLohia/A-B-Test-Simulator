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
st.set_page_config(page_title="A/B Test Sweep Simulator", layout="wide")
st.title("📊 A/B Test Sweep: Effect of Sample Size, Dropout & Treatment Effect")

# Toggle for CUPED
use_cuped = st.checkbox("🧪 Use CUPED Adjustment", value=True)

# Sweep Parameters
sample_sizes = [1000, 5000, 10000, 20000, 50000]
dropout_rates = [0.0, 0.1, 0.2, 0.3]
treatment_effects = [0.0, 0.01, 0.02, 0.05, 0.1]

# Grid Sweep and Collect
results = []

for n_users in sample_sizes:
    for dropout in dropout_rates:
        for effect in treatment_effects:
            df = generate_experiment_data(n_users=n_users,
                                           treatment_effect=effect,
                                           dropout_rate=dropout,
                                           stratify=True,
                                           seed=42)
            if use_cuped:
                df = apply_cuped(df)
                metric = 'adjusted_metric'
            else:
                metric = 'post_metric'

            t_stat, p_val = classical_t_test(df, metric_col=metric)
            bayes = run_bayesian_ab_test(df, metric_col=metric)

            results.append({
                "sample_size": n_users,
                "dropout_rate": dropout,
                "treatment_effect": effect,
                "p_value": p_val,
                "bayes_prob_treatment_better": bayes['prob_treatment_better']
            })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Plots
st.markdown("### 🗺️ Heatmaps of P-Values and Bayesian Probabilities")

# Prepare heatmap data
for metric_name, value_col in [("P-Value", "p_value"), ("Bayesian P(T > C)", "bayes_prob_treatment_better")]:
    for effect in treatment_effects:
        heatmap_data = results_df[results_df['treatment_effect'] == effect].pivot(index='dropout_rate', columns='sample_size', values=value_col)
        fig, ax = plt.subplots()
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm" if value_col == "p_value" else "YlGnBu", ax=ax)
        ax.set_title(f"{metric_name} Heatmap (Effect = {effect:.2f})")
        st.pyplot(fig)


st.markdown("### 📉 P-Value vs Sample Size")
fig1, ax1 = plt.subplots()
sns.lineplot(data=results_df, x="sample_size", y="p_value", hue="treatment_effect", style="dropout_rate", ax=ax1)
ax1.axhline(0.05, linestyle="--", color="red")
ax1.set_title("Frequentist P-Values by Sample Size")
st.pyplot(fig1)

st.markdown("### 📈 Bayesian Probability Treatment > Control")
fig2, ax2 = plt.subplots()
sns.lineplot(data=results_df, x="sample_size", y="bayes_prob_treatment_better",
             hue="treatment_effect", style="dropout_rate", ax=ax2)
ax2.axhline(0.95, linestyle="--", color="green")
ax2.set_title("Bayesian Probability vs Sample Size")
st.pyplot(fig2)

st.markdown("---")
st.write("These sweeps demonstrate how increasing sample size, reducing dropout, and larger treatment effects contribute to statistical power and inference reliability.")
