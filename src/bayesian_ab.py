
import numpy as np
from scipy.stats import norm


def run_bayesian_ab_test(df, metric_col='post_metric', prior_mu=0, prior_sigma=1):
    """
    Simple Bayesian inference using normal-normal conjugate prior.
    Returns posterior means and probability that treatment > control.
    """
    df_valid = df[df[metric_col].notnull()]
    control = df_valid[df_valid['group'] == 'control'][metric_col]
    treatment = df_valid[df_valid['group'] == 'treatment'][metric_col]

    # Observed data
    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = np.mean(control), np.mean(treatment)
    var_c, var_t = np.var(control), np.var(treatment)

    # Posterior means and variances (normal-normal update)
    post_var_c = 1 / (n_c / var_c + 1 / (prior_sigma ** 2))
    post_var_t = 1 / (n_t / var_t + 1 / (prior_sigma ** 2))

    post_mean_c = post_var_c * (mean_c * n_c / var_c + prior_mu / (prior_sigma ** 2))
    post_mean_t = post_var_t * (mean_t * n_t / var_t + prior_mu / (prior_sigma ** 2))

    # Simulate posterior distributions
    samples_c = np.random.normal(post_mean_c, np.sqrt(post_var_c), 10000)
    samples_t = np.random.normal(post_mean_t, np.sqrt(post_var_t), 10000)

    diff = samples_t - samples_c
    prob_treatment_better = np.mean(diff > 0)

    return {
        'posterior_mean_control': post_mean_c,
        'posterior_mean_treatment': post_mean_t,
        'posterior_std_control': np.sqrt(post_var_c),
        'posterior_std_treatment': np.sqrt(post_var_t),
        'prob_treatment_better': prob_treatment_better
    }
