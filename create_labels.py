import os
import csv
import cv2
 
# ===== paths =====
dataset_root = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\reduced_dataset"
output_csv = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\classification_labels.csv"
 
splits = ["train", "val", "test"]
 
# severity thresholds
LOW_THRESHOLD = 0.02      # 2%
MEDIUM_THRESHOLD = 0.05   # 5%
 
rows = []
 
for split in splits:
    img_dir = os.path.join(dataset_root, split, "IMG")
    gt_dir = os.path.join(dataset_root, split, "GT")
 
    img_files = os.listdir(img_dir)
 
    for file_name in img_files:
        img_path = os.path.join(img_dir, file_name)
        gt_path = os.path.join(gt_dir, file_name)
 
        if not os.path.exists(gt_path):
            continue
 
        # read GT mask in grayscale
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
 
        if mask is None:
            continue
 
        total_pixels = mask.shape[0] * mask.shape[1]
        crack_pixels = cv2.countNonZero(mask)
        crack_ratio = crack_pixels / total_pixels
 
        # assign label
        if crack_ratio < LOW_THRESHOLD:
            label = "Low"
        elif crack_ratio < MEDIUM_THRESHOLD:
            label = "Medium"
        else:
            label = "High"
 
        rows.append([split, img_path, gt_path, crack_ratio, label])
 
# write to CSV
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "image_path", "gt_path", "crack_ratio", "label"])
    writer.writerows(rows)
 
print(f"CSV created successfully: {output_csv}")
print(f"Total samples labeled: {len(rows)}")
 