import pathlib
import torch
import numpy as np
from torch.utils.data import Dataset

from wrapml.utils import read_image

class ImageClassificationDataset(Dataset):
    '''
        data should contain columns [img_path, label]
        label is needed in train mode
    '''
    def __init__(self, data, img_folder, train_mode=True, transforms=None):
        self.data = data
        self.img_folder = pathlib.Path(img_folder)
        self.transforms = transforms
        self.train_mode = train_mode

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = np.array(read_image(self.img_folder/self.data.iloc[index].img_path))

        if self.transforms:
            image = self.transforms(image=image)["image"]

        if not self.train_mode:
            return image

        return (
            image,
            torch.tensor(self.data.iloc[index]['label'], dtype=torch.long)
        )
