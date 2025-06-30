import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import UTKFaceDataset          # keep your custom dataset file
from coatnet import coatnet_0               # make sure coatnet.py is present in your project

# Step 1: Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Step 2: Load Dataset
dataset = UTKFaceDataset(root_dir='data/UTKFace', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 3: Load Model (CoAtNet)
model = coatnet_0(num_classes=4)  # 4 age bins
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
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/10] - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), "coatnet_finetuned.pth")
print("✅ Training completed and model saved as coatnet_finetuned.pth")
