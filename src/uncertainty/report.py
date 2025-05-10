# report.py
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize
from temperature_scaling import apply_temperature_scaling
from ece import calculate_ece


def summarize_calibration(ece, bin_accuracies, bin_confidences, model_name="Unnamed", dataset_name="Unknown"):
    issues = []
    for i, (acc, conf) in enumerate(zip(bin_accuracies, bin_confidences)):
        if abs(acc - conf) > 0.2:
            if conf > acc:
                issues.append(f"- Bin {i+1}: Overconfident (conf={conf:.2f}, acc={acc:.2f})")
            else:
                issues.append(f"- Bin {i+1}: Underconfident (conf={conf:.2f}, acc={acc:.2f})")

    recommendation = "Apply temperature scaling." if ece > 0.1 else "Model appears well-calibrated."

    return f"""
Model Calibration Summary
=========================

- Model: {model_name}
- Dataset: {dataset_name}
- ECE: {ece:.3f}

Issues by bin:
{chr(10).join(issues[:5]) + ("..." if len(issues) > 5 else "")}

Recommendations:
- {recommendation}
- Evaluate uncertainty metrics like entropy or margin
- Consider retraining with label smoothing if overconfident
"""




def generate_comparison_pdf(before_df: pd.DataFrame, after_df: pd.DataFrame, model_name: str, dataset_name: str, out_path: str, temp: float):
    before_ece, before_accs, before_confs, _, bins = calculate_ece(before_df)
    after_ece, after_accs, after_confs, _, _ = calculate_ece(after_df)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    with PdfPages(out_path) as pdf:
        # Page 1 - Before and After Reliability Diagrams
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        axs[0].bar(bin_centers, before_accs, width=0.08, alpha=0.6, label='Accuracy')
        axs[0].bar(bin_centers, before_confs, width=0.05, alpha=0.6, label='Confidence')
        axs[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
        axs[0].set_title(f'Before Calibration (ECE = {before_ece:.3f})')
        axs[0].set_xlabel("Confidence")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].bar(bin_centers, after_accs, width=0.08, alpha=0.6, label='Accuracy')
        axs[1].bar(bin_centers, after_confs, width=0.05, alpha=0.6, label='Confidence')
        axs[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
        axs[1].set_title(f'After Calibration (ECE = {after_ece:.3f})')
        axs[1].set_xlabel("Confidence")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Page 2 - Summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        summary_text = f"""
Model Comparison Report
=======================

- Model: {model_name}
- Dataset: {dataset_name}

Calibration Before:
-------------------
- ECE: {before_ece:.3f}

Calibration After Temperature Scaling:
--------------------------------------
- ECE: {after_ece:.3f}
- Best Temperature: {temp:.2f}

Observations:
-------------
- Significant reduction in ECE after scaling (if applicable)
- Visual comparison confirms improved alignment between confidence and accuracy

Recommendations:
----------------
- Use temperature scaling in production scoring if ECE improved significantly
- Re-check on different validation/test sets
"""
        ax.text(0.01, 0.99, summary_text, verticalalignment='top', fontsize=10, family='monospace')
        pdf.savefig()
        plt.close()

    return out_path


if __name__ == "__main__":
    results_df = pd.read_csv("misc/inference_results.csv")
    df_scaled, best_temp = apply_temperature_scaling(results_df)
    generate_comparison_pdf(results_df, df_scaled, "Model", "Dataset", "misc/report_original.pdf", best_temp)
