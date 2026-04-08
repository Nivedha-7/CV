import os
import random
import shutil
 
# ===== paths =====
source_root = r"C:\Users\MH784SK\Downloads\CrackVision12K\split_dataset_final"
target_root = r"C:\Users\MH784SK\OneDrive - EY\Desktop\CV\reduced_dataset"
 
# how much data to keep
ratio = 0.5   # 50%
 
# for reproducibility
random.seed(42)
 
splits = ["train", "val", "test"]
 
for split in splits:
    img_dir = os.path.join(source_root, split, "IMG")
    gt_dir = os.path.join(source_root, split, "GT")
 
    target_img_dir = os.path.join(target_root, split, "IMG")
    target_gt_dir = os.path.join(target_root, split, "GT")
 
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_gt_dir, exist_ok=True)
 
    # get all image files
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    img_files.sort()
 
    # shuffle and sample
    random.shuffle(img_files)
    sample_count = int(len(img_files) * ratio)
    selected_files = img_files[:sample_count]
 
    copied = 0
    skipped = 0
 
    for file_name in selected_files:
        src_img = os.path.join(img_dir, file_name)
        src_gt = os.path.join(gt_dir, file_name)
 
        dst_img = os.path.join(target_img_dir, file_name)
        dst_gt = os.path.join(target_gt_dir, file_name)
 
        if os.path.exists(src_gt):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_gt, dst_gt)
            copied += 1
        else:
            skipped += 1
 
    print(f"{split}: copied {copied} pairs, skipped {skipped} files (GT missing)")
 
print("Reduced dataset created successfully.")
 