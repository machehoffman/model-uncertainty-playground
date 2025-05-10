import numpy as np
import pandas as pd
from scipy.optimize import minimize

#post-hoc calibration
def apply_temperature_scaling(results_df: pd.DataFrame):
    logits = results_df["logits"].values.reshape(-1, 1)
    labels = results_df["label"].values

    def loss_fn(temp):
        scaled_probs = 1 / (1 + np.exp(-logits / temp))
        eps = 1e-7
        log_loss = -np.mean(labels * np.log(scaled_probs + eps) + (1 - labels) * np.log(1 - scaled_probs + eps))
        return log_loss

    base_loss = loss_fn(1.0)
    print(f"[Temperature Scaling] Log-loss at T=1.0 (before scaling): {base_loss:.5f}")

    res = minimize(loss_fn, x0=np.array([1.0]), bounds=[(0.05, 10.0)], method='L-BFGS-B')
    best_temp = res.x[0]
    best_loss = res.fun
    print(f"[Temperature Scaling] Best temperature found: T={best_temp:.4f}")
    print(f"[Temperature Scaling] Log-loss at T={best_temp:.4f} (after scaling): {best_loss:.5f}")

    scaled_probs = 1 / (1 + np.exp(-logits / best_temp))

    df_scaled = results_df.copy()
    df_scaled["prob"] = scaled_probs.flatten()
    df_scaled["pred"] = (df_scaled["prob"] > 0.5).astype(int)
    return df_scaled, best_temp


if __name__ == "__main__":
    results_df = pd.read_csv("misc/inference_results.csv")
    df_scaled, best_temp = apply_temperature_scaling(results_df)
    print(f"Best temperature: {best_temp}")
    df_scaled.to_csv("misc/inference_results_scaled.csv", index=False)
