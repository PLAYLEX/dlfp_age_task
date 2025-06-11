from torch.utils.data import Dataset
from PIL import Image
import os

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith('.jpg')
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Extract age from filename (e.g., 25_1_0_...jpg → 25)
        filename = os.path.basename(img_path)
        try:
            age = int(filename.split('_')[0])
        except ValueError:
            age = -1  # fallback if filename is broken

        if self.transform:
            img = self.transform(img)

        return img, age
