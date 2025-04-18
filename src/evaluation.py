def summarize_results(df, t_stat, p_val, bayes_results, uplift_summary):
    """
    Print and summarize key outputs from all methods, including practical significance.
    """
    practical_threshold = 0.015

    # Classical Effect Size
    classical_control = df[df['group'] == 'control']['post_metric'].dropna()
    classical_treatment = df[df['group'] == 'treatment']['post_metric'].dropna()
    classical_delta = classical_treatment.mean() - classical_control.mean()

    if abs(classical_delta) >= practical_threshold:
        classical_practical_msg = "The effect is practically significant (>= 1.5% difference)."
    else:
        classical_practical_msg = "The effect is small and may not be practically meaningful."

    # Bayesian Effect Size
    # Check for required keys in bayes_results
    bayes_delta = bayes_results.get('posterior_mean_treatment', 0) - bayes_results.get('posterior_mean_control', 0)
    if abs(bayes_delta) >= practical_threshold:
        bayes_practical_msg = "The effect is practically significant (>= 1.5% difference)."
    else:
        bayes_practical_msg = "The effect is small and may not be practically meaningful."

    # Uplift Summary
    uplift_auc = uplift_summary.get('uplift_auc', 'N/A')
    estimated_avg_uplift = uplift_summary.get('estimated_avg_uplift', 'N/A')
    example_uplift_scores = uplift_summary.get('example_uplift_scores', 'N/A')

    # Compile the full summary
    summary = f"""
==== EXPERIMENT RESULTS SUMMARY ====

Sample Size:
  - Total Users: {len(df)}
  - Control: {sum(df['group'] == 'control')}
  - Treatment: {sum(df['group'] == 'treatment')}

Classical A/B Test:
  - T-statistic: {t_stat:.4f}
  - P-value: {p_val:.4f}
  - Mean (Control): {classical_control.mean():.4f}
  - Mean (Treatment): {classical_treatment.mean():.4f}
  - Estimated Effect Size (Delta): {classical_delta:.4f}
  - Practical Significance: {classical_practical_msg}

Bayesian Inference:
  - Posterior Mean (Control): {bayes_results.get('posterior_mean_control', 'N/A'):.4f}
  - Posterior Mean (Treatment): {bayes_results.get('posterior_mean_treatment', 'N/A'):.4f}
  - Probability Treatment > Control: {bayes_results.get('prob_treatment_better', 'N/A'):.4f}
  - Estimated Effect Size (Delta): {bayes_delta:.4f}
  - Practical Significance: {bayes_practical_msg}

Uplift Modeling:
  - Uplift AUC: {uplift_auc:.4f}
  - Estimated Avg Uplift: {estimated_avg_uplift:.4f}
  - Sample Uplift Scores: {example_uplift_scores}

====================================
    """
    return summary
