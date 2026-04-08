import random
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
 
csv_path = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\classification_labels.csv"
model_path = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\best_crack_classifier.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
df = pd.read_csv(csv_path)
test_df = df[df["split"] == "test"].reset_index(drop=True)
 
checkpoint = torch.load(model_path, map_location=device)
class_names = checkpoint["label_classes"]
 
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()
 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
 
def get_message(label):
    if label == "Low":
        return "Minor crack detected. Routine monitoring recommended."
    elif label == "Medium":
        return "Moderate crack detected. Maintenance should be scheduled."
    elif label == "High":
        return "Severe crack detected. Immediate repair is recommended."
    return "Unknown result."
 
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
 
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
 
    return class_names[pred.item()]
 
samples = test_df.sample(5, random_state=42)
 
for _, row in samples.iterrows():
    image_path = row["image_path"]
    actual = row["label"]
    predicted = predict_image(image_path)
    message = get_message(predicted)
 
    print("=" * 60)
    print("Image:", image_path)
    print("Actual:", actual)
    print("Predicted:", predicted)
    print("Message:", message)
 
    img = Image.open(image_path).convert("RGB")
    plt.imshow(img)
    plt.title(f"Actual: {actual} | Pred: {predicted}")
    plt.axis("off")
    plt.show()
 