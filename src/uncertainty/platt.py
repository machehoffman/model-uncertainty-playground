
import pandas as pd
from sklearn.linear_model import LogisticRegression

def apply_platt_scaling(results_df: pd.DataFrame):
    logits = results_df["logits"].values.reshape(-1, 1)
    labels = results_df["label"].values
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(logits, labels)
    platt_probs = clf.predict_proba(logits)[:, 1]
    df_scaled = results_df.copy()
    df_scaled["prob"] = platt_probs
    df_scaled["pred"] = (platt_probs > 0.5).astype(int)
    print("[Platt Scaling] Coefficients:", clf.coef_, "Intercept:", clf.intercept_)
    return df_scaled