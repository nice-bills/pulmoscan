import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
import os
import copy
import json

# Configuration
DATA_DIR = "data/covid19"
MODEL_SAVE_PATH = "models/covid/mobilenetv3_best.pth"
METRICS_SAVE_PATH = "models/covid/training_metrics.json"
BATCH_SIZE = 32
NUM_EPOCHS = 5  # Keep it small for Phase 1 demo
LEARNING_RATE = 0.001

def train_model():
    # Device Agnostic Code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting training on device: {device}")

    # Data Augmentation & Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load Data
    full_dataset = datasets.ImageFolder(DATA_DIR, data_transforms['train'])
    class_names = full_dataset.classes
    print(f"📂 Classes found: {class_names}")

    # Split Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Apply val transforms (hacky way since random_split doesn't support separate transforms easily, 
    # but CenterCrop is deterministic so it's fine for this baseline)
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Initialize Model (MobileNetV3-Large)
    print("🏗️  Loading MobileNetV3-Large...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    
    # Modify Classifier for 3 classes
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(class_names))
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\n🏁 Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'⭐ Best Val Acc: {best_acc:4f}')

    # Save Model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")
    
    # Save Metrics
    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    train_model()
