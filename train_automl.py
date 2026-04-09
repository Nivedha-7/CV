import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
 
# =========================
# SETTINGS
# =========================
data_dir = r"C:\Users\VT448EX\OneDrive - EY\Desktop\Reduced_dataset"
batch_size = 16
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
 
train_data = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
val_data = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
 
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
 
num_classes = len(train_data.classes)
 
# =========================
# MODEL LIST (AutoML)
# =========================
def get_models():
    return {
        "ResNet18": models.resnet18(weights=None),
        "MobileNet": models.mobilenet_v2(weights=None),
        "EfficientNet": models.efficientnet_b0(weights=None)
    }
 
# =========================
# TRAIN FUNCTION
# =========================
def train_model(model):
    model = model.to(device)
 
    # adjust final layer
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
 
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
 
            outputs = model(images)
            loss = criterion(outputs, labels)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
    return model
 
# =========================
# VALIDATION
# =========================
def evaluate(model):
    model.eval()
    correct = 0
    total = 0
 
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
 
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
 
            correct += (preds == labels).sum().item()
            total += labels.size(0)
 
    return correct / total
 
# =========================
# AUTOML PROCESS
# =========================
models_dict = get_models()
 
best_model = None
best_acc = 0
best_name = ""
 
for name, model in models_dict.items():
    print(f"\nTraining {name}...")
 
    model = train_model(model)
    acc = evaluate(model)
 
    print(f"{name} Accuracy: {acc:.4f}")
 
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name
 
# =========================
# SAVE BEST MODEL
# =========================
torch.save({
    "model_state_dict": best_model.state_dict(),
    "model_name": best_name,
    "accuracy": best_acc
}, "best_automl_model.pth")
 
print("\n✅ Best Model Selected:", best_name)
print("Accuracy:", best_acc)
 