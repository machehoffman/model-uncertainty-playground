import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def apply_transform(image, transform):
    return transform(image=image)["image"]

transforms = A.Compose(
    [
        A.ToFloat(always_apply=True),
        ToTensorV2(),
    ]
)