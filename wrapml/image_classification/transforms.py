import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size, additional_transforms=[]):
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(.3, 1), p=1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        *additional_transforms,
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255, always_apply=True
        ),
        ToTensorV2()
    ])

def get_valid_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size, p=1),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255, always_apply=True
        ),
        ToTensorV2()
    ])
