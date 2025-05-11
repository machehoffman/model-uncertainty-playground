import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
def mc_dropout_inference(model: torch.nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         device: torch.device,
                         threshold: float = 0.5,
                         mc_samples: int = 20):
    
    results = []
    model.train()  # Keep dropout active
    for batch in tqdm(dataloader, desc="MC Dropout Inference"):
        images, labels, image_paths = batch
        images = images.to(device)

        probs_list = []
        with torch.no_grad():
            for _ in range(mc_samples):
                outputs = model(images)
                probs = torch.sigmoid(outputs)  # For binary classification
                probs_list.append(probs.detach().cpu().numpy())

        probs_array = np.stack(probs_list, axis=0)  # shape: [mc_samples, batch, 1]
        mean_probs = probs_array.mean(axis=0)
        std_probs = probs_array.std(axis=0)

        preds = (mean_probs > threshold).astype(float)

        for i in range(len(image_paths)):
            results.append({
                "image_path": image_paths[i],
                "pred": float(preds[i]),
                "prob_mean": float(mean_probs[i]),
                "prob_std": float(std_probs[i]),
                "label": labels[i].item()
            })

    return pd.DataFrame(results)



class McDropoutUncertaintyAnalyzer:
    def __init__(self, results_df: pd.DataFrame,save_path: str = None):
        self.df = results_df.copy()
        self._classify_errors()
        self.save_path = save_path

    def _classify_errors(self):
        def classify(row):
            if row["pred"] == row["label"]:
                return "Correct"
            elif row["pred"] == 1.0 and row["label"] == 0.0:
                return "False Positive"
            elif row["pred"] == 0.0 and row["label"] == 1.0:
                return "False Negative"
            else:
                return "Unknown"
        self.df["error_type"] = self.df.apply(classify, axis=1)

    def plot_uncertainty_histogram(self):
        plt.figure(figsize=(8, 4))
        plt.hist(self.df["prob_std"], bins=30, color='skyblue', edgecolor='black')
        plt.title("Histogram of Prediction Uncertainty (std)")
        plt.xlabel("Uncertainty (std)")
        plt.ylabel("Number of Samples")
        plt.grid(True)
        plt.tight_layout()
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            save_path = os.path.join(self.save_path,"uncertainty_histogram.png")
            plt.savefig(save_path)
        plt.show()

    def plot_prob_vs_uncertainty(self):
        plt.figure(figsize=(6, 6))
        plt.scatter(self.df["prob_mean"], self.df["prob_std"], alpha=0.5, s=20)
        plt.title("MC Dropout: Probability vs. Uncertainty")
        plt.xlabel("Mean Probability")
        plt.ylabel("Uncertainty (std)")
        plt.grid(True)
        plt.tight_layout()
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            save_path = os.path.join(self.save_path,"prob_vs_uncertainty.png")
            plt.savefig(save_path)
        plt.show()

    def plot_error_type_scatter(self):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=self.df,
            x="prob_mean",
            y="prob_std",
            hue="error_type",
            palette={"Correct": "green", "False Positive": "red", "False Negative": "blue"},
            alpha=0.6
        )
        plt.title("Prediction Type vs. MC Dropout Uncertainty")
        plt.xlabel("Mean Probability")
        plt.ylabel("Uncertainty (std)")
        plt.grid(True)
        plt.tight_layout()
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            save_path = os.path.join(self.save_path,"prob_vs_uncertainty.png")
            plt.savefig(save_path)
        plt.show()

    def top_uncertain_errors(self, error_type="False Positive", top_k=5):
        df_filtered = self.df[self.df["error_type"] == error_type]
        return df_filtered.sort_values("prob_std", ascending=False).head(top_k)

    def summary(self):
        print("Sample count by error type:")
        print(self.df["error_type"].value_counts())

    def create_collage(self,num_images=16):
        df_filtered = self.df[self.df["error_type"] != "Correct"]
        df_filtered = df_filtered.sort_values("prob_std", ascending=False).head(num_images).reset_index(drop=True)

        plt.figure(figsize=(10, 10))
        for i, row in df_filtered.iterrows():
            img = Image.open(row["image_path"])
            plt.subplot(4, 4, i + 1)
            plt.imshow(img)
            plt.title(f"Error Type: {row['error_type']}")
            plt.axis('off')
        plt.tight_layout()
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            save_path = os.path.join(self.save_path,"collage.png")
            plt.savefig(save_path)
        plt.show()



if __name__ == "__main__":
    # Load results from CSV
    results_df = pd.read_csv("misc/inference_results_mc_dropout.csv")

    # Initialize uncertainty analyzer
    analyzer = McDropoutUncertaintyAnalyzer(results_df,save_path="misc/mc_dropout_uncertainty_analysis/")

    # Plot uncertainty histogram
    analyzer.plot_uncertainty_histogram()

    # Plot probability vs uncertainty   
    analyzer.plot_prob_vs_uncertainty()

    # Plot error type scatter
    analyzer.plot_error_type_scatter()

    # Top uncertain errors  
    print(analyzer.top_uncertain_errors(error_type="False Positive", top_k=5))

    # Summary
    analyzer.summary()

    # Create collage
    analyzer.create_collage()
