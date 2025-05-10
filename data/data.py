# dataset and dataloader

from torch.utils.data import Dataset
import cv2
import pandas as pd
import os
from augmentations import apply_transform, transforms

def read_image(image_path, bgr2rgb=False):
    image = cv2.imread(image_path)
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transforms):
        self.df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        image = read_image(os.path.join(self.img_dir, sample["image_name"]))
        label = sample["class"] == "Green"
        image = apply_transform(image, self.transforms)
        return image, label
    
    def debug(self, idx):
        sample = self.df.iloc[idx]
        image = read_image(os.path.join(self.img_dir, sample["image_name"]))
        label = sample["class"] == "Green"
        
        # Create debug directory if it doesn't exist
        os.makedirs("debug_images", exist_ok=True)
        
        # Save the image
        output_path = f"debug_images/image_{idx}_class_{label}.jpg"
        cv2.imwrite(output_path, image)
        print(f"Saved debug image to: {output_path}")


if __name__ == "__main__":
    dataset = ImageDataset(annotations_file="misc/demo_cipo.csv",
                            img_dir="misc/cipo_demo/",
                            transforms=transforms)
    print(len(dataset))
    dataset.debug(0)
    a=1
