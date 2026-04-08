import os
import subprocess
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
 
st.set_page_config(page_title="Crack Detection System", layout="wide")
 
# =========================
# PATHS
# =========================
clf_path = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\best_crack_classifier.pth"
seg_model_path = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\best_unet_model.pth"
 
# training scripts
clf_train_script = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\train_classifier.py"
seg_train_script = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\train_segmentation.py"
 
# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# =========================
# SEGMENTATION MODEL
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
 
# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_classification_model():
    checkpoint = torch.load(clf_path, map_location=device)
    class_names = checkpoint["label_classes"]
 
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
 
    return model, class_names
 
@st.cache_resource
def load_segmentation_model():
    model = UNet().to(device)
    model.load_state_dict(torch.load(seg_model_path, map_location=device))
    model.eval()
    return model
 
clf_model, class_names = load_classification_model()
seg_model = load_segmentation_model()
 
# =========================
# TRANSFORMS
# =========================
clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
 
seg_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
 
# =========================
# HELPERS
# =========================
def get_message(label):
    if label == "Low":
        return "Minor crack detected. Routine monitoring is recommended."
    elif label == "Medium":
        return "Moderate crack detected. Maintenance should be scheduled."
    elif label == "High":
        return "Severe crack detected. Immediate repair is recommended."
    return "Unknown result."
 
def predict_classification(image):
    input_tensor = clf_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = clf_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
 
    label = class_names[pred.item()]
    confidence = float(conf.item()) * 100
    return label, confidence
 
def predict_segmentation(image):
    input_tensor = seg_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = seg_model(input_tensor)
        mask = torch.sigmoid(output)
        mask = (mask > 0.5).float()
 
    return mask.squeeze().cpu().numpy()
 
def run_training(script_path):
    if not os.path.exists(script_path):
        return False, f"Training script not found: {script_path}"
 
    try:
        st.write("Python used for retraining:", os.sys.executable)
        result = subprocess.run(
            [os.sys.executable, script_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)
 
# =========================
# UI
# =========================
st.title("Crack Detection and Analysis System")
st.markdown("Upload an image and choose the required task.")
 
with st.sidebar:
    st.header("Settings")
    task = st.selectbox("Select Task", ["Classification", "Segmentation"])
    mode = st.radio("Model Selection Mode", ["Manual", "Auto"])
 
    if mode == "Auto":
        if task == "Classification":
            selected_model = "ResNet18 (Auto selected)"
        else:
            selected_model = "U-Net (Auto selected)"
    else:
        if task == "Classification":
            selected_model = st.selectbox("Choose Model", ["ResNet18", "EfficientNet", "MobileNet"])
        else:
            selected_model = st.selectbox("Choose Model", ["U-Net", "DeepLabV3", "FCN"])
 
    st.write("Selected model:", selected_model)
 
uploaded_file = st.file_uploader("Upload crack image", type=["jpg", "jpeg", "png"])
 
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
 
    if st.button("Run Prediction"):
        if task == "Classification":
            label, confidence = predict_classification(image)
            message = get_message(label)
 
            with col2:
                st.subheader("Prediction Result")
                st.success(f"Predicted Severity: {label}")
                st.info(f"Confidence: {confidence:.2f}%")
                st.warning(message)
 
        elif task == "Segmentation":
            mask = predict_segmentation(image)
 
            with col2:
                st.subheader("Predicted Crack Mask")
                st.image(mask, use_container_width=True, clamp=True)
 
# =========================
# RETRAIN SECTION
# =========================
st.markdown("---")
st.subheader("Retrain Model")
 
if st.button("Retrain"):
    with st.spinner("Retraining model..."):
        if task == "Classification":
            success, output = run_training(clf_train_script)
        else:
            success, output = run_training(seg_train_script)
 
    if success:
        st.success("Retraining completed successfully.")
        st.text_area("Training Output", output, height=200)
    else:
        st.error("Retraining failed.")
        st.text_area("Error Output", output, height=200)
 
# =========================
# EXPORT MODEL SECTION
# =========================
st.markdown("---")
st.subheader("Export Model")
 
if task == "Classification" and os.path.exists(clf_path):
    with open(clf_path, "rb") as f:
        st.download_button(
            label="Download Classification Model",
            data=f,
            file_name="best_crack_classifier.pth",
            mime="application/octet-stream"
        )
 
elif task == "Segmentation" and os.path.exists(seg_model_path):
    with open(seg_model_path, "rb") as f:
        st.download_button(
            label="Download Segmentation Model",
            data=f,
            file_name="best_unet_model.pth",
            mime="application/octet-stream"
        )
 
 
 