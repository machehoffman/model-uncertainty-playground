import torch
import pandas as pd
from tqdm import tqdm
from mc_dropout_inference import mc_dropout_inference
# from timm.models.efficientnet import EfficientNet
# from torch.serialization import add_safe_globals

# # Add EfficientNet to safe globals
# add_safe_globals([EfficientNet])

def run_inference(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  use_mc_dropout: bool = False,
                  mc_samples: int = 20):
    
    if use_mc_dropout:
        res_pd = mc_dropout_inference(model, dataloader, device, mc_samples=mc_samples)
    else:
        res_pd = model_inference(model, dataloader, device)
    return res_pd


def model_inference(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                      device: torch.device,
                      threshold: float = 0.5):
    
    results = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            images, labels, image_paths = batch
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.sigmoid(outputs)
            preds = (probs > threshold).float()
            for i in range(len(image_paths)):
                # results.append({
                #     "image_path": image_paths[i],
                #     "pred": preds[i].detach().cpu().numpy(),
                #     "prob": probs[i].detach().cpu().numpy(),
                #     "label": labels[i].detach().cpu().numpy(),
                #     "logits": outputs[i].detach().cpu().numpy()
                # }
                results.append({
                    "image_path": image_paths[i],
                    "pred": float(preds[i].item()),
                    "prob": float(probs[i].item()),
                    "label": labels[i].item(),
                    "logits": float(outputs[i].item())
                })
    return pd.DataFrame(results)


if __name__ == "__main__":

    from src.data.data import ImageDataset
    from src.data.augmentations import  transforms
    from torch.utils.data import DataLoader
    
    model = torch.load("misc/model.pt", weights_only=False)
    dataset = ImageDataset(annotations_file="misc/demo_cipo.csv", 
                           img_dir="misc/cipo_demo/", 
                           transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Running MC Dropout inference on {device}")
    res_pd = run_inference(model, dataloader, device, use_mc_dropout=True, mc_samples=20)
    print(res_pd)

    res_pd.to_csv("misc/inference_results_mc_dropout.csv", index=False)