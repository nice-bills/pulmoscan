import torch
import torch.nn as nn
from torchvision import models, transforms
import onnxruntime as ort
import numpy as np
import time
import os
import sys
from PIL import Image
import glob

# Add project root to path
sys.path.append(os.getcwd())

MODEL_PATH = "models/covid/mobilenetv3_best.pth"
ONNX_PATH = "models/covid/mobilenetv3.onnx"
QUANTIZED_PATH = "models/covid/mobilenetv3_int8.onnx"
DATA_DIR = "data/covid19"
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']

def load_pytorch_model():
    print("Loading PyTorch model...")
    device = torch.device("cpu")
    model = models.mobilenet_v3_large(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(CLASSES))
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_real_images(limit_per_class=20):
    images = []
    labels = []
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"Loading {limit_per_class} images per class for accuracy test...")
    for idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)
        # Handle 'images' subdirectory if it exists (Kaggle dataset structure)
        if os.path.exists(os.path.join(class_dir, 'images')):
             class_dir = os.path.join(class_dir, 'images')
             
        file_list = glob.glob(os.path.join(class_dir, "*.png")) + glob.glob(os.path.join(class_dir, "*.jpg"))
        
        for img_path in file_list[:limit_per_class]:
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0)
                images.append(tensor)
                labels.append(idx)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
                
    return images, labels

def evaluate_accuracy(model_func, images, labels, model_type="Model"):
    correct = 0
    total = len(images)
    start_time = time.time()
    
    for i, img_tensor in enumerate(images):
        output = model_func(img_tensor)
        pred = np.argmax(output)
        if pred == labels[i]:
            correct += 1
            
    duration = time.time() - start_time
    acc = 100 * correct / total
    avg_time = (duration / total) * 1000
    
    print(f"  {model_type:<15} Accuracy: {acc:.1f}% | Avg Inference: {avg_time:.2f}ms")
    return acc, avg_time

def test_inference():
    # 1. Setup Models
    torch_model = load_pytorch_model()
    
    print(f"Loading ONNX model: {ONNX_PATH}...")
    ort_session = ort.InferenceSession(ONNX_PATH)
    
    print(f"Loading Quantized ONNX model: {QUANTIZED_PATH}...")
    try:
        q_ort_session = ort.InferenceSession(QUANTIZED_PATH)
    except Exception:
        q_ort_session = None
        print("Quantized model not found or invalid.")

    # Wrapper functions for unified interface
    def run_torch(tensor):
        with torch.no_grad():
            return to_numpy(torch_model(tensor))
            
    def run_onnx(tensor):
        inputs = {ort_session.get_inputs()[0].name: to_numpy(tensor)}
        return ort_session.run(None, inputs)[0]
        
    def run_q_onnx(tensor):
        if not q_ort_session: return np.array([[0,0,0]])
        inputs = {q_ort_session.get_inputs()[0].name: to_numpy(tensor)}
        return q_ort_session.run(None, inputs)[0]

    # 2. Sanity Check (Dummy Input)
    print("\n--- Sanity Check (Dummy Input) ---")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch_out = run_torch(dummy_input)
    onnx_out = run_onnx(dummy_input)
    
    np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-05)
    print("ONNX and PyTorch logits match on dummy input!")

    # 3. Accuracy Evaluation (Real Data)
    print("\n--- Accuracy Evaluation (Real Data) ---")
    images, labels = get_real_images(limit_per_class=50) # 150 images total
    
    if not images:
        print("No images found in data/covid19. Skipping accuracy test.")
    else:
        print(f"Evaluating on {len(images)} real images...")
        acc_torch, time_torch = evaluate_accuracy(run_torch, images, labels, "PyTorch")
        acc_onnx, time_onnx = evaluate_accuracy(run_onnx, images, labels, "ONNX")
        
        if q_ort_session:
            acc_q, time_q = evaluate_accuracy(run_q_onnx, images, labels, "ONNX (INT8)")
            
            print("\n--- Final Comparison ---")
            print(f"PyTorch:     {time_torch:.2f}ms, Accuracy: {acc_torch:.1f}%")
            print(f"ONNX:        {time_onnx:.2f}ms, Accuracy: {acc_onnx:.1f}% (Speedup: {time_torch/time_onnx:.2f}x)")
            print(f"ONNX (INT8): {time_q:.2f}ms, Accuracy: {acc_q:.1f}% (Speedup: {time_torch/time_q:.2f}x)")

if __name__ == "__main__":
    test_inference()
