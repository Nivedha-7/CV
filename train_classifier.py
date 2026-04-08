import os
import pandas as pd
from PIL import Image
import sys
print(sys.executable) 
 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
 
# =========================
# PATHS
# =========================
csv_path = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\classification_labels.csv"
model_save_path = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\best_crack_classifier.pth"
 
# =========================
# SETTINGS
# =========================
batch_size = 16
num_epochs = 10
learning_rate = 0.0001
image_size = 224
num_workers = 0  # keep 0 for Windows if issues happen
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
 
# =========================
# LOAD CSV
# =========================
df = pd.read_csv(csv_path)
 
print("Total samples:", len(df))
print(df["label"].value_counts())
 
# encode labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])
 
print("Classes:", list(label_encoder.classes_))
 
# split data
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df = df[df["split"] == "val"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)
 
# =========================
# DATASET CLASS
# =========================
class CrackDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
 
    def __len__(self):
        return len(self.dataframe)
 
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label_encoded"])
 
        image = Image.open(img_path).convert("RGB")
 
        if self.transform:
            image = self.transform(image)
 
        return image, label
 
# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
 
val_test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
 
# =========================
# DATALOADERS
# =========================
train_dataset = CrackDataset(train_df, transform=train_transform)
val_dataset = CrackDataset(val_df, transform=val_test_transform)
test_dataset = CrackDataset(test_df, transform=val_test_transform)
 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
 
# =========================
# MODEL
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(label_encoder.classes_))
model = model.to(device)
 
# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
 
# =========================
# TRAINING FUNCTION
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
 
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
 
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
 
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
 
# =========================
# VALIDATION FUNCTION
# =========================
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
 
    all_labels = []
    all_preds = []
 
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
 
            outputs = model(images)
            loss = criterion(outputs, labels)
 
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
 
            correct += (preds == labels).sum().item()
            total += labels.size(0)
 
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
 
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds
 
# =========================
# TRAIN LOOP
# =========================
best_val_acc = 0.0
 
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
 
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print("-" * 50)
 
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "label_classes": list(label_encoder.classes_)
        }, model_save_path)
        print("Best model saved.")
 
print("Training completed.")
print("Best validation accuracy:", best_val_acc)
 
# =========================
# TEST EVALUATION
# =========================
checkpoint = torch.load(model_save_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
 
test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
 
print("\nTest Accuracy:", test_acc)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
 
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
 