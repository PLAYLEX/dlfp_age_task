from typing import Optional
import matplotlib.pyplot as plt
import os
from contextlib import contextmanager
import time

def set_seed(seed: Optional[int]):
    if seed is None:
        return
    else:
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def display_images(images, labels, class_names=None, title_prefix="Label: ", num_rows=2, num_cols=5, figsize=(15, 6)):
    num_images = len(images)
    if num_images == 0:
        print("No images to display.")
        return
    plt.figure(figsize=figsize)
    for i in range(min(num_images, num_rows * num_cols)):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        label_text = str(labels[i])
        if class_names and int(labels[i]) < len(class_names):
            label_text = class_names[int(labels[i])]
        plt.title(f"{title_prefix}{label_text}")
    plt.tight_layout()
    plt.show()


def create_directory(directory_path):
    abs_directory_path = os.path.abspath(directory_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {abs_directory_path}")
    else:
        print(f"Directory already exists: {abs_directory_path}")
    return abs_directory_path


@contextmanager
def use_timer(name: str) -> None:
    """
    Context manager to measure execution time of code blocks.
    """
    start_time = time.time()
    yield
    elapsed_seconds = time.time() - start_time

    # Format time as hh:mm:ss
    hours, remainder = divmod(int(elapsed_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((elapsed_seconds - int(elapsed_seconds)) * 1000)

    if hours > 0:
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    print(f"{name} completed in {time_str}")

