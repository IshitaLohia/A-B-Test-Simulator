
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def model_uplift_effects(df):
    """
    Estimate heterogeneous treatment effects using uplift modeling.
    """
    df = df.copy()
    df = df[df['post_metric'].notnull()]  # Drop missing post metrics

    # Create treatment indicator (1 for treatment, 0 for control)
    df['treatment_flag'] = (df['group'] == 'treatment').astype(int)

    # Create binary outcome (e.g., post_metric > threshold)
    outcome_threshold = df['post_metric'].median()
    df['label'] = (df['post_metric'] > outcome_threshold).astype(int)

    # Encode categorical variables (like stratum)
    if df['stratum'].dtype == 'object':
        df['stratum_encoded'] = LabelEncoder().fit_transform(df['stratum'])
    else:
        df['stratum_encoded'] = df['stratum']

    # Feature set
    X = df[['pre_metric', 'stratum_encoded', 'treatment_flag']]
    y = df['label']
    X['interaction'] = X['pre_metric'] * X['treatment_flag']

    # Train uplift model
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Predict probabilities for full test set
    proba = model.predict_proba(X_test)[:, 1]

    # Estimate uplift score as correlation with treatment flag
    uplift_score = proba[X_test['treatment_flag'] == 1].mean() - proba[X_test['treatment_flag'] == 0].mean()

    uplift_summary = {
        'uplift_auc': roc_auc_score(y_test, proba),
        'example_uplift_scores': proba[:5].tolist(),
        'estimated_avg_uplift': uplift_score
    }

    return uplift_summary
