import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import UTKFaceDataset  # assumes dataset.py is in the same folder
import os

# Step 1: Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Step 2: Load dataset
dataset = UTKFaceDataset(root_dir='data/UTKFace', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 3: Load Model (Regression Output)
model = models.resnet18(weights=None)  # change to 'weights="DEFAULT"' to use pretrained
model.fc = nn.Linear(model.fc.in_features, 1)  # Regression output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 4: Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training Loop
print("🚀 Training started...")
for epoch in range(3):  # You can increase the number of epochs
    total_loss = 0
    for i, (imgs, ages) in enumerate(dataloader):
        imgs = imgs.to(device)
        ages = ages.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, ages)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/3], Step [{i+1}/{len(dataloader)}], Batch Loss: {loss.item():.4f}")

    print(f"✅ Epoch [{epoch+1}] completed | Total Loss: {total_loss:.4f}")

# Step 6: Save the model
torch.save(model.state_dict(), "age_model.pth")
print("🎉 Training finished. Model saved as age_model.pth")

