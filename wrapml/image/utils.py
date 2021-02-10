import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def read_image(path):
    return Image.open(path)

def show_image_grid(img_paths, n_cols=3, figsize=(15, 12)):
    nrows = int(np.ceil(len(img_paths) / n_cols))
    fig, ax = plt.subplots(nrows, n_cols, figsize=figsize)
    for i, axes in enumerate(ax.flatten()):
        if i < len(img_paths):
            axes.imshow(read_image(img_paths[i]))