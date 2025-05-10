import pandas as pd
from sklearn.isotonic import IsotonicRegression

def apply_isotonic_calibration(results_df: pd.DataFrame):
    logits = results_df["logits"].values.flatten()
    labels = results_df["label"].values
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(logits, labels)
    iso_probs = ir.transform(logits)
    df_scaled = results_df.copy()
    df_scaled["prob"] = iso_probs
    df_scaled["pred"] = (iso_probs > 0.5).astype(int)
    print("[Isotonic Calibration] Fitted isotonic model on logits")
    return df_scaled