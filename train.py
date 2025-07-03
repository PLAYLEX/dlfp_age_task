import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import os

from dataset import UTKFaceDataset  # make sure this matches your dataset class

# ================================
# Config
# ================================
COATNET_MODEL_NAME = "coatnet_0_224"
PRETRAINED_PATH = "coatnet_0_224_simclr_encoder_best_val_loss.pth"
NUM_CLASSES = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# Transform
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ================================
# Dataset
# ================================
dataset = UTKFaceDataset(root_dir='data/UTKFace', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ================================
# Load CoAtNet with timm
# ================================
print("🔄 Loading CoAtNet model from timm...")

model = timm.create_model(
    COATNET_MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES
)

# Load pre-trained encoder weights
if os.path.exists(PRETRAINED_PATH):
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device), strict=False)
    print(f"✅ Loaded best encoder weights from {PRETRAINED_PATH}")
else:
    print("⚠️ Pretrained path not found. Using random initialization.")

model.to(device)

# ================================
# Loss & Optimizer
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ================================
# Training loop
# ================================
print("🚀 Training started...")

for epoch in range(10):
    model.train()
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
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/10] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# ================================
# Save fine-tuned model
# ================================
torch.save(model.state_dict(), "coatnet_finetuned.pth")
print("✅ Training complete and model saved as coatnet_finetuned.pth")
