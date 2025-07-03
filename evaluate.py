import torch
from torchvision import transforms
from dataset import UTKFaceDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ===============================
# Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===============================
# Load dataset
# ===============================
dataset = UTKFaceDataset(root_dir='data/UTKFace', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# ===============================
# Load model
# ===============================
from coatnet import coatnet_0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = coatnet_0(num_classes=4)  # ✅ FIXED: removed pretrained argument
model.load_state_dict(torch.load("coatnet_finetuned.pth", map_location=device))
model.to(device)
model.eval()

# ===============================
# Evaluate
# ===============================
all_preds = []
all_labels = []
correct = 0
total = 0

print("\n📸 Evaluating images...")

with torch.no_grad():
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.append(predicted.cpu().numpy()[0])
        all_labels.append(labels.cpu().numpy()[0])

accuracy = 100 * correct / total
print(f"\n✅ Evaluation Accuracy: {accuracy:.2f}%")

# ===============================
# Confusion Matrix
# ===============================
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<18", "18-40", "41-60", ">60"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Epoch 6 Checkpoint")
plt.savefig("confusion_matrix_epoch6.png")
plt.show()

print("\n✅ Confusion matrix saved as confusion_matrix_epoch6.png")
