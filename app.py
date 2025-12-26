import os
import torch
from torchvision import models, transforms
from flask import Flask, render_template, request
from PIL import Image

# ---------------- Configuration ----------------
MODEL_PATH = "cattle_model.pth"
NUM_CLASSES = 5
CLASS_NAMES = ['Red Dane cattle', 'Jersey cattle', 'Holstein Friesian cattle',
               'Brown Swiss cattle', 'Ayrshire cattle']
IMG_SIZE = 224
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

# ---------------- Flask Routes ----------------
@app.route('/')
def index():
    return render_template('index.html', breed=None, confidence=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No image selected")
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    breed, confidence = predict_image(filepath)
    return render_template('index.html', breed=breed, confidence=confidence, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
