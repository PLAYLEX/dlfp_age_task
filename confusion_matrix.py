import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Example: load your true labels and predicted labels
# Replace these with your actual lists if you saved them
labels = [0, 1, 2, 0, 1, 2, 1, 0]
predicted = [0, 2, 1, 0, 1, 2, 1, 0]

# If you already have true and predicted lists from your evaluation, load them here
cm = confusion_matrix(labels, predicted)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save image so you can open later
plt.show()
