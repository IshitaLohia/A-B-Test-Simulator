
from scipy.stats import ttest_ind


def classical_t_test(df, metric_col='post_metric'):
    """
    Perform a two-sample t-test between control and treatment groups.
    Ignores rows with missing data in the metric.
    """
    df_valid = df[df[metric_col].notnull()]
    control = df_valid[df_valid['group'] == 'control'][metric_col]
    treatment = df_valid[df_valid['group'] == 'treatment'][metric_col]

    t_stat, p_val = ttest_ind(treatment, control, equal_var=False)
    return t_stat, p_val
