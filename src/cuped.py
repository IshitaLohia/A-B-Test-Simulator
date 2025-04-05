import numpy as np

def apply_cuped(df, pre_col='pre_metric', post_col='post_metric'):
    """
    Applies CUPED (Controlled Pre-Experiment Data) adjustment to reduce variance in post-treatment metrics.
    """
    pre = df[pre_col]
    post = df[post_col]

    # Drop missing post-metric rows
    valid_idx = post.notnull()
    pre = pre[valid_idx]
    post = post[valid_idx]

    # Compute theta (covariance / variance)
    cov = np.cov(post, pre)[0, 1]
    var = np.var(pre)
    theta = cov / var

    # Apply CUPED adjustment
    adjusted = post - theta * (pre - np.mean(pre))
    df.loc[valid_idx, 'adjusted_metric'] = adjusted

    return df
