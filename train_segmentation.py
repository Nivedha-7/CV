import os
from PIL import Image
 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
 
# =========================
# PATHS
# =========================
dataset_root = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\reduced_dataset"
model_save_path = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\best_unet_model.pth"
 
# =========================
# SETTINGS
# =========================
image_size = 256
batch_size = 4
num_epochs = 10
learning_rate = 1e-4
num_workers = 0
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
 
# =========================
# DATASET
# =========================
class CrackSegmentationDataset(Dataset):
    def __init__(self, root_dir, split):
        self.img_dir = os.path.join(root_dir, split, "IMG")
        self.mask_dir = os.path.join(root_dir, split, "GT")
        self.files = sorted(os.listdir(self.img_dir))
 
    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        file_name = self.files[idx]
 
        img_path = os.path.join(self.img_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)
 
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
 
        image = image.resize((image_size, image_size))
        mask = mask.resize((image_size, image_size))
 
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
 
        # make mask binary
        mask = (mask > 0.5).float()
 
        return image, mask
 
train_dataset = CrackSegmentationDataset(dataset_root, "train")
val_dataset = CrackSegmentationDataset(dataset_root, "val")
test_dataset = CrackSegmentationDataset(dataset_root, "test")
 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
 
# =========================
# U-NET MODEL
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.conv(x)
 
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
 
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
 
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
 
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
 
        self.bottleneck = DoubleConv(512, 1024)
 
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
 
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
 
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
 
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
 
        self.final = nn.Conv2d(64, 1, kernel_size=1)
 
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
 
        bn = self.bottleneck(self.pool4(d4))
 
        u4 = self.up4(bn)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)
 
        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)
 
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
 
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)
 
        return self.final(u1)
 
model = UNet().to(device)
 
# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
 
# =========================
# METRIC
# =========================
def dice_score(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
 
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
 
    return (2.0 * intersection + smooth) / (union + smooth)
 
# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(loader):
    model.train()
    total_loss = 0
 
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
 
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
    return total_loss / len(loader)
 
def evaluate(loader):
    model.eval()
    total_loss = 0
    total_dice = 0
 
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
 
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
 
            total_dice += dice_score(outputs, masks).item()
 
    return total_loss / len(loader), total_dice / len(loader)
 
# =========================
# TRAIN LOOP
# =========================
best_val_dice = 0.0
 
for epoch in range(num_epochs):
    train_loss = train_one_epoch(train_loader)
    val_loss, val_dice = evaluate(val_loader)
 
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Dice:   {val_dice:.4f}")
    print("-" * 40)
 
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), model_save_path)
        print("Best model saved.")
 
print("Training completed.")
 
# =========================
# TEST EVALUATION
# =============
# ============
model.load_state_dict(torch.load(model_save_path, map_location=device))
test_loss, test_dice = evaluate(test_loader)
 
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Dice Score: {test_dice:.4f}")