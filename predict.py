import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button

# ---------------- Configuration ----------------
MODEL_PATH = "cattle_model.pth"
NUM_CLASSES = 5
CLASS_NAMES = [
    'Red Dane cattle',
    'Jersey cattle',
    'Holstein Friesian cattle',
    'Brown Swiss cattle',
    'Ayrshire cattle'
]
IMG_SIZE = 224

# ---------------- Model Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- Prediction Function ----------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
    breed = CLASS_NAMES[predicted.item()]
    confidence = conf.item() * 100
    return breed, confidence

# ---------------- GUI Setup ----------------
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Display the uploaded image
    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Run prediction
    breed, confidence = predict_image(file_path)
    result_label.config(
        text=f"✅ Predicted: {breed}\n📊 Confidence: {confidence:.2f}%", fg="green")

# Create main window
root = tk.Tk()
root.title("🐄 Cattle Breed Classifier")
root.geometry("400x500")
root.resizable(False, False)

Label(root, text="Cattle Breed Classifier", font=("Arial", 18, "bold")).pack(pady=10)
Button(root, text="📁 Upload Image", command=upload_and_predict, font=("Arial", 12), bg="#4CAF50", fg="white").pack(pady=10)

img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
