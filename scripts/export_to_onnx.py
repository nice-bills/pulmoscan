import torch
import torch.nn as nn
from torchvision import models
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.getcwd())

MODEL_PATH = "models/covid/mobilenetv3_best.pth"
ONNX_PATH = "models/covid/mobilenetv3.onnx"
QUANTIZED_PATH = "models/covid/mobilenetv3_int8.onnx"
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']

def load_pytorch_model():
    print(f"Loading PyTorch model from {MODEL_PATH}...")
    device = torch.device("cpu") # Export is typically done on CPU
    model = models.mobilenet_v3_large(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(CLASSES))
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def export_to_onnx(model):
    print(f"Exporting to ONNX: {ONNX_PATH}...")
    
    # Create dummy input: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    file_size = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"✅ Export success! Size: {file_size:.2f} MB")

def quantize_onnx():
    print(f"Quantizing model to INT8: {QUANTIZED_PATH}...")
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            ONNX_PATH,
            QUANTIZED_PATH,
            weight_type=QuantType.QUInt8
        )
        
        original_size = os.path.getsize(ONNX_PATH) / (1024 * 1024)
        quantized_size = os.path.getsize(QUANTIZED_PATH) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"✅ Quantization success!")
        print(f"   Original: {original_size:.2f} MB")
        print(f"   Quantized: {quantized_size:.2f} MB")
        print(f"   Reduction: {reduction:.1f}%")
        
    except ImportError:
        print("❌ 'onnx' or 'onnxruntime' not installed. Skipping quantization.")
        print("   Run: uv pip install onnx onnxruntime")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization")
    args = parser.parse_args()

    model = load_pytorch_model()
    export_to_onnx(model)
    
    if args.quantize:
        quantize_onnx()
