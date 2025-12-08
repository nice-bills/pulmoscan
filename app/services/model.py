import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import time

MODEL_PATH = "models/covid/mobilenetv3_best.pth"
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']

class COVIDClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        try:
            print(f"Loading model from {MODEL_PATH}...")
            model = models.mobilenet_v3_large(weights=None) 
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(num_ftrs, len(CLASSES))
            
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            print("Model loaded successfully.")
            return model
        except FileNotFoundError:
            print("Model file not found. API will start but predictions will fail.")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict(self, image_bytes):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        t0 = time.time()
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        t1 = time.time()
        
        return {
            "class": CLASSES[predicted_idx.item()],
            "confidence": float(confidence.item()),
            "inference_time": t1 - t0
        }

classifier = COVIDClassifier()
