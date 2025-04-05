import numpy as np
import pandas as pd

def generate_experiment_data(n_users=10000, treatment_effect=0.02, dropout_rate=0.1, stratify=True, seed=42):
    np.random.seed(seed)
    
    # Simulate user IDs and stratification variable
    user_id = np.arange(n_users)
    strata = np.random.choice(['mobile', 'desktop'], size=n_users)

    # Stratified or random assignment to control/treatment
    if stratify:
        group = []
        for s in strata:
            if s == 'mobile':
                group.append(np.random.choice(['control', 'treatment']))
            else:
                group.append(np.random.choice(['control', 'treatment']))
        group = np.array(group)
    else:
        group = np.random.choice(['control', 'treatment'], size=n_users)

    # Simulate pre-experiment metric
    pre_metric = np.random.normal(loc=1.0, scale=0.2, size=n_users)

    # Post metric with treatment effect
    post_metric = pre_metric + np.random.normal(loc=0, scale=0.1, size=n_users)
    post_metric += (group == 'treatment') * treatment_effect

    # Simulate missing data/dropout in post_metric
    mask = np.random.rand(n_users) < dropout_rate
    post_metric[mask] = np.nan

    df = pd.DataFrame({
        'user_id': user_id,
        'group': group,
        'stratum': strata,
        'pre_metric': pre_metric,
        'post_metric': post_metric
    })

    return df
