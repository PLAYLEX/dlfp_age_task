import torch
from torchvision import transforms, models
from PIL import Image
import os

# Define the same transform used during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the trained model
model = models.resnet18(weights=None)  # Use weights='DEFAULT' if pretrained
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Age is a single value

# Load your trained weights
model.load_state_dict(torch.load("age_model.pth", map_location=torch.device("cpu")))
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Folder where your images are
folder = "data/UTKFace"

# Get only image files and skip hidden/system files like .DS_Store
images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
images = sorted(images)[:5]  # You can change to more or fewer

print("📷 Evaluating images...\n")

for fname in images:
    img_path = os.path.join(folder, fname)
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor).item()

    print(f"{fname} → Predicted Age: {round(prediction, 2)}")

print("\n✅ Evaluation completed.")
