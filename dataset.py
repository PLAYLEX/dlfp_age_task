import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg"):
                try:
                    age = int(filename.split('_')[0])
                    label = self.age_to_class(age)
                    self.image_paths.append((os.path.join(root_dir, filename), label))
                except Exception as e:
                    continue

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    def age_to_class(self, age):
        # 4-class binning
        if age <= 18:
            return 0  # child
        elif 19 <= age <= 30:
            return 1  # young adult
        elif 31 <= age <= 45:
            return 2  # adult
        else:
            return 3  # older adult
