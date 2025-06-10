from flask import Flask, render_template, request, jsonify
from model.tumor_classifier import TumorClassifier
from PIL import Image
import torch
from torchvision import transforms

app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TumorClassifier(num_classes=4)
model.load_state_dict(torch.load(r'C:\Users\SID\Downloads\TumorClassifierProject\TumorClassifierProject\model_weights\brain_tumor_model.pth', map_location=device))
model.to(device)
model.eval()

def predict_image(image_file):
    """Predict tumor type from an uploaded image file"""
    image = Image.open(image_file.stream).convert("RGB")  # Convert to RGB to avoid issues
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    return class_names[predicted.item()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    predicted_class = predict_image(file)
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
