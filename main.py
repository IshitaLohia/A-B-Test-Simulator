import pandas as pd
from src.simulate_experiment import generate_experiment_data
from src.cuped import apply_cuped
from src.inference_methods import classical_t_test
from src.bayesian_ab import run_bayesian_ab_test
from src.uplift_model import model_uplift_effects
from src.evaluation import summarize_results

# Step 1: Simulate Experiment Data
df = generate_experiment_data(
    n_users=10000,
    treatment_effect=0.02,
    dropout_rate=0.1,
    stratify=True,
    seed=42
)

# Step 2: Apply CUPED adjustment
df = apply_cuped(df)

# Step 3: Run Classical T-Test
t_stat, p_val = classical_t_test(df, metric_col='post_metric')

# Step 4: Run Bayesian A/B Inference
bayes_results = run_bayesian_ab_test(df)

# Step 5: Model Uplift / Heterogeneous Treatment Effects
uplift_summary = model_uplift_effects(df)

# Step 6: Summarize and Print Results
summary = summarize_results(df, t_stat, p_val, bayes_results, uplift_summary)
print(summary)

# Optionally save results
df.to_csv("/simulated_ab_test_results.csv", index=False)
