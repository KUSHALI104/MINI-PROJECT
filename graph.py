import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    precision_recall_fscore_support
)

# ------------------- Configuration -------------------
DATA_DIR = "cattle_data/train"   # same dataset path
MODEL_PATH = "cattle_model.pth"
BATCH_SIZE = 16
NUM_CLASSES = 5

# ------------------- Device -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ------------------- Validation Transform -------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------- Dataset & Loader -------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
class_names = dataset.classes

val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------- Load Model -------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ------------------- Prediction -------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ------------------- Metrics -------------------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print("\n📊 MODEL PERFORMANCE")
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\n📄 CLASSIFICATION REPORT\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ------------------- Class-wise Bar Graph -------------------
precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None
)

x = np.arange(len(class_names))

plt.figure(figsize=(10, 5))
plt.bar(x - 0.25, precision_c, width=0.25, label="Precision")
plt.bar(x, recall_c, width=0.25, label="Recall")
plt.bar(x + 0.25, f1_c, width=0.25, label="F1-score")

plt.xticks(x, class_names, rotation=30)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Precision, Recall, and F1-score per Cattle Breed")
plt.legend()
plt.tight_layout()
plt.show()
