import torch
from torchvision import transforms, models
from dataset import UTKFaceDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define same transform used during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load dataset
dataset = UTKFaceDataset(root_dir='data/UTKFace', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Load model
from coatnet import coatnet_0
model = coatnet_0(num_classes=4)
model.load_state_dict(torch.load("coatnet_finetuned.pth", map_location=torch.device("cpu")))
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("\n📸 Evaluating images...")

# Initialize variables for overall accuracy and per-class
correct = 0
total = 0
num_classes = 4
class_correct = [0] * num_classes
class_total = [0] * num_classes

true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i].item()
            pred = predicted[i].item()
            class_correct[label] += int(pred == label)
            class_total[label] += 1

        true_labels.append(labels.item())
        predicted_labels.append(predicted.item())

        print(f"File evaluated -> True label: {labels.item()}, Predicted: {predicted.item()}")

# Overall accuracy
accuracy = 100 * correct / total
print(f"\n✅ Overall Evaluation Accuracy: {accuracy:.2f}%")

# Per-class accuracy
print("\n📊 Per-class accuracy:")
for i in range(num_classes):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(f"Class {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"Class {i}: N/A (no samples)")

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<18", "18-40", "41-60", ">60"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_result.png")
plt.show()
