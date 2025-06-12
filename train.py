import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import UTKFaceDataset

# Step 1: Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Step 2: Load dataset (no 'train=True' used)
dataset = UTKFaceDataset(root_dir='data/UTKFace', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 3: Load Model (Classification Output)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 age bins

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 4: Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training Loop
print("🚀 Training started...")
for epoch in range(10):  # 10 epochs
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"✅ Epoch {epoch+1} completed | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "age_model_classification.pth")
print("✅ Training finished. Model saved as age_model_classification.pth")
