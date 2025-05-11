import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# def calculate_ece(results_df: pd.DataFrame, n_bins: int = 10):
#     # Calculate ECE
#     # Define number of bins for ECE
#     bins = np.linspace(0, 1, n_bins + 1)
#     bin_indices = np.digitize(results_df["prob"], bins) - 1

#     # Compute ECE and collect data for plotting
#     ece = 0
#     bin_accuracies = []
#     bin_confidences = []
#     bin_counts = []

#     for i in range(n_bins):
#         bin_mask = bin_indices == i
#         bin_size = np.sum(bin_mask)
#         if bin_size > 0:
#             bin_acc = np.mean(results_df["pred"][bin_mask] == results_df["label"][bin_mask])
#             bin_conf = np.mean(results_df["prob"][bin_mask])
#             bin_accuracies.append(bin_acc)
#             bin_confidences.append(bin_conf)
#             bin_counts.append(bin_size)
#             ece += (bin_size / len(results_df)) * abs(bin_acc - bin_conf)
#         else:
#             bin_accuracies.append(0)
#             bin_confidences.append(0)
#             bin_counts.append(0)

#     return ece, bin_accuracies, bin_confidences, bin_counts, bins
def calculate_ece(results_df: pd.DataFrame, n_bins: int = 10):
    """
    Calculate Expected Calibration Error (ECE) for binary classification hence the confidence is the probability of the positive class.
    Args:
        results_df: pd.DataFrame
            The dataframe containing the results of the model.
        n_bins: int
            The number of bins to use for the ECE calculation.
    Returns:
        ece: float
    """
    # Ensure 'confidence' column exists based on prediction correctness
    if "confidence" not in results_df.columns:
        results_df = results_df.copy()
        results_df["confidence"] = np.where(
            results_df["pred"] == 1,
            results_df["prob"],
            1 - results_df["prob"]
        )

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(results_df["confidence"], bins) - 1

    ece = 0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(results_df["pred"][bin_mask] == results_df["label"][bin_mask])
            bin_conf = np.mean(results_df["confidence"][bin_mask])
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_size)
            ece += (bin_size / len(results_df)) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)

    return ece, bin_accuracies, bin_confidences, bin_counts, bins
def plot_ece(results_df: pd.DataFrame):
    # Plot ECE
    ece, bin_accuracies, bin_confidences, bin_counts ,bins= calculate_ece(results_df)
    # Plot reliability diagram
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.6, label='Accuracy')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.bar(bin_centers, bin_confidences, width=0.05, alpha=0.6, label='Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram (ECE = {ece:.3f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Save plot
    plt.savefig("misc/ece1.png")

if __name__ == "__main__":
    results_df = pd.read_csv("misc/inference_results.csv")
    # ece = calculate_ece(results_df)
    # print(f"ECE: {ece}")
    plot_ece(results_df)
