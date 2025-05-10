import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
def calculate_metrics(results_df: pd.DataFrame):
    calc_f1 = f1_score(results_df["label"], results_df["pred"])
    calc_acc = accuracy_score(results_df["label"], results_df["pred"])
    calc_prec = precision_score(results_df["label"], results_df["pred"])
    calc_rec = recall_score(results_df["label"], results_df["pred"])
    return calc_f1, calc_acc, calc_prec, calc_rec



if __name__ == "__main__":
    results_df = pd.read_csv("misc/inference_results.csv")
    calc_f1, calc_acc, calc_prec, calc_rec = calculate_metrics(results_df)
    print(f"F1: {calc_f1}, Acc: {calc_acc}, Prec: {calc_prec}, Rec: {calc_rec}")
