import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from src.model import DilatedAttentionNetwork
from src.dataset import LoveDADataset

# --- Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "output/best_model.pth"
DATA_DIR = r"D:\2025\LULC\LandCover_Project\data\LoveDA"
BATCH_SIZE = 4
NUM_CLASSES = 7

# Class Names
CLASS_NAMES = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agriculture']

def compute_iou(conf_matrix):
    """
    Calculate IoU for each class from the confusion matrix.
    IoU = TP / (TP + FP + FN)
    """
    true_positive = np.diag(conf_matrix)
    false_positive = conf_matrix.sum(axis=0) - true_positive
    false_negative = conf_matrix.sum(axis=1) - true_positive
    
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = true_positive / (true_positive + false_positive + false_negative)
    
    return iou

def evaluate_model():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = DilatedAttentionNetwork(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Load Validation Data
    print("Loading validation dataset...")
    val_ds = LoveDADataset(DATA_DIR, split='Val')
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Evaluation Loop
    print("Starting evaluation...")
    total_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE) # Shape: [B, H, W]

            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1) # Shape: [B, H, W]

            # Flatten for confusion matrix
            preds_flat = preds.cpu().numpy().flatten()
            masks_flat = masks.cpu().numpy().flatten()

            # Update Confusion Matrix
            # We only care about valid pixels (0-6). 
            # If there are any 255 (ignore) pixels, we should filter them out if the dataset produces them.
            # Based on dataset.py, 255 is used for ignore regions.
            valid_indices = masks_flat != 255
            preds_flat = preds_flat[valid_indices]
            masks_flat = masks_flat[valid_indices]

            cm = confusion_matrix(masks_flat, preds_flat, labels=range(NUM_CLASSES))
            total_conf_matrix += cm

    # 4. Calculate Metrics
    print("\n--- Evaluation Results ---")
    
    # Pixel Accuracy
    pixel_acc = np.diag(total_conf_matrix).sum() / total_conf_matrix.sum()
    print(f"Overall Pixel Accuracy: {pixel_acc:.4f}")

    # IoU
    iou_per_class = compute_iou(total_conf_matrix)
    mean_iou = np.nanmean(iou_per_class)
    
    print(f"Mean IoU (mIoU): {mean_iou:.4f}")
    print("\nClass-wise IoU:")
    for i, class_name in enumerate(CLASS_NAMES):
        iou = iou_per_class[i]
        print(f"  {class_name:<12}: {iou:.4f}")

if __name__ == "__main__":
    evaluate_model()
