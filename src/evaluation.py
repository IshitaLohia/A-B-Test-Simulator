
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
        classical_practical_msg = f"The observed difference of {classical_delta:.4f} is practically significant."
    else:
        classical_practical_msg = "The effect is small and may not be practically meaningful."


    # Bayesian Effect Size
    bayes_delta = bayes_results['posterior_mean_treatment'] - bayes_results['posterior_mean_control']
    if abs(bayes_delta) >= practical_threshold:
        bayes_practical_msg = f"The observed difference of {bayes_delta:.4f} is practically significant."
    else:
        bayes_practical_msg = "The effect is small and may not be practically meaningful."

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
  - Posterior Mean (Control): {bayes_results['posterior_mean_control']:.4f}
  - Posterior Mean (Treatment): {bayes_results['posterior_mean_treatment']:.4f}
  - Probability Treatment > Control: {bayes_results['prob_treatment_better']:.4f}
  - Estimated Effect Size (Delta): {bayes_delta:.4f}
  - Practical Significance: {bayes_practical_msg}

Uplift Modeling:
  - Uplift AUC: {uplift_summary['uplift_auc']:.4f}
  - Estimated Avg Uplift: {uplift_summary['estimated_avg_uplift']:.4f}
  - Sample Uplift Scores: {uplift_summary['example_uplift_scores']}

====================================
    """
    return summary
