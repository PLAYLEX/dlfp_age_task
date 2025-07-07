import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import decode_image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Taken from https://github.com/hamkerlab/DL_for_practitioners/blob/main/06_1_SSL_SimCLR/06_1_SSL_SimCLR.ipynb
class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self._mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        actual_b_size = z_i.shape[0]
        if actual_b_size != self.batch_size:
            print(f"WARNING: batch size from z_i ({actual_b_size}) != self.batch_size ({self.batch_size})")
            
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, diagonal=self.batch_size) #torch.diag(input = sim, diagonal = self.batch_size)
        sim_j_i = torch.diag(input = sim, diagonal =-self.batch_size)
        
        if sim_i_j.nelement() == 0 or sim_j_i.nelement() == 0: # Check if empty
            print("ERROR: sim_i_j or sim_j_i is empty!")
            print(f"Shape of sim: {sim.shape}, diagonal for sim_i_j: {actual_b_size}")

        # We have 2N samples
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

def create_df_with_age_categories(dataset_path):
    data = []
    for file in os.listdir(dataset_path):
        if "UTKFace" in dataset_path:
            try:
                age, _, _, _ = file.split("_")
            except ValueError:
                age, _, _ = file.split("_")
        if "AgeDB" in dataset_path:
            try:
                _, _, age, _ = file.split("_")
            except ValueError:
                raise ValueError(f"File {file} does not match expected format for AgeDB dataset.")
        
        img_path = os.path.join(dataset_path, file)
        data.append([img_path, int(age)])
    # Create DataFrame
    df = pd.DataFrame(data, columns=["image_path", "age"])

    # Define age categories
    bins = [0, 18, 40, 60, np.inf]
    labels = ["<18", "18-40", "41-60", ">60"]
    df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels)
    return df

# Verify class distributions in each split
def plot_class_distribution(y_train, y_val, y_test, title="Class Distribution"):
    # train_dist = np.bincount(y_train, minlength=num_classes) / len(y_train)
    # val_dist = np.bincount(y_val, minlength=num_classes) / len(y_val)
    # test_dist = np.bincount(y_test, minlength=num_classes) / len(y_test)
    train_dist = y_train.value_counts(normalize=True).sort_index()
    val_dist = y_val.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    
    class_names = train_dist.index.astype(str).tolist()
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x - width, train_dist, width, label='Train', color='skyblue')
    plt.bar(x, val_dist, width, label='Validation', color='lightgreen')
    plt.bar(x + width, test_dist, width, label='Test', color='salmon')
    
    plt.xlabel('Animal Class')
    plt.ylabel('Proportion')
    plt.title(title)
    plt.xticks(x, class_names)
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_class_weights(y_train, num_classes):
    # Calculate class weights inversely proportional to class frequencies
    class_counts = y_train.value_counts().sort_index().values
    class_weights = 1. / class_counts
    class_weights = class_weights / np.sum(class_weights) * num_classes  # Normalize
    return class_weights

def plot_class_weights(class_weights, y_train, title='Class Weights for Imbalanced Data'):
    # Create labels for the classes
    train_dist = y_train.value_counts(normalize=True).sort_index()
    class_names = train_dist.index.astype(str).tolist()
    x = np.arange(len(class_names))
    # Display weights
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, class_weights, color='skyblue')
    plt.xticks(x, class_names)
    plt.xlabel('Age Class')
    plt.ylabel('Weight')
    plt.title(title)

    # Add weight values above bars
    for i, (weight, bar) in enumerate(zip(class_weights, bars)):
        plt.text(i, weight + 0.1, f'{weight:.2f}', ha='center')

    plt.tight_layout()
    plt.show()

class AgeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Custom dataset for loading images and their corresponding age labels.

        Args:
            image_paths (list): List of image file paths.
            labels (list): List of age labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths.values
        self.labels = labels.values
        self.transform = transform

        # Create a mapping from age categories to numeric labels
        self.label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        # image = Image.open(img_path).convert('RGB')
        image = decode_image(img_path) # , mode='rgb'
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        label = self.label_to_index[label]
        label = torch.tensor(label, dtype=torch.long)

        return image, label

class UTKFaceSimCLRDataset(Dataset):
    def __init__(self, image_paths, labels, image_size, transform=None):
        self.image_paths = image_paths.values
        self.labels = labels.values
        self.image_size = image_size
        self.transform = transform

        # Create a mapping from age categories to numeric labels
        self.label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = decode_image(img_path)
        label = self.labels[idx]
        label = self.label_to_index[label]
        label = torch.tensor(label, dtype=torch.long)

        x_i = self.transform(image)
        x_j = self.transform(image)
        return x_i, x_j, label

def create_datasets_loader(X, y, transform, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the AgeDataset (e.g. UTKFace or AgeDB).

    Args:
        X (list): List of image file paths.
        y (list): List of age labels corresponding to the images.
        transform (callable): Transformations to be applied to the images.
        batch_size (int, optional): Number of samples per batch. Default is 32.
        shuffle (bool, optional): Whether to shuffle the dataset at every epoch. Default is True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Default is 0.
    
    Returns:
        dataset (AgeDataset): The dataset object containing images and labels.
        dataloader (DataLoader): The DataLoader object for batching and shuffling the dataset.
    """
    # create dataset
    dataset = AgeDataset(image_paths=X, labels=y, transform=transform)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return dataset, dataloader

def create_datasets_loader_simclr(X, y, transform, batch_size=32, shuffle=True, num_workers=0):
    # create dataset
    dataset = UTKFaceSimCLRDataset(image_paths=X, labels=y, image_size=224, transform=transform)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True, pin_memory=torch.cuda.is_available())
    return dataset, dataloader

# Funktion zum Anzeigen von Bildern mit Labels und Pfaden
def plot_images_with_labels_and_filenames(data_loader: DataLoader, dataset, num_images=5, title="Images with Labels and Filenames"):
    # Hole eine Batch von Bildern und Labels
    images, labels = next(iter(data_loader))
    
    # Denormalisierung der Bilder für die Anzeige
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = images.permute(0, 2, 3, 1)  # Form ändern zu (Batch, Höhe, Breite, Kanäle)
    images = images * torch.tensor(std).view(1, 1, 1, 3) + torch.tensor(mean).view(1, 1, 1, 3)
    images = images.numpy()
    
    # Hole die Pfade aus dem Dataset
    filenames = [os.path.basename(path) for path in dataset.image_paths[:num_images]]

    # Umkehren der label_to_index-Mapping-Tabelle
    index_to_label = {v: k for k, v in dataset.label_to_index.items()}
    
    # Erstelle die Anzeige
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        # Konvertiere das numerische Label zurück in die age_category
        age_category = index_to_label[labels[i].item()]
        plt.title(f"Age Category: {age_category}\nFile: {filenames[i]}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.suptitle(title, y=1)
    plt.show()