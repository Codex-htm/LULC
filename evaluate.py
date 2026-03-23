import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from src.model import DilatedAttentionNetwork
from src.dataset import LoveDADataset
from torch.utils.data import DataLoader

# --- Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "output/best_model.pth" # We load the best saved weights
DATA_DIR = r"D:\2025\LULC\LandCover_Project\data\LoveDA"
OUTPUT_DIR = "output/predictions"

# Define Colors for the 7 Classes (R, G, B)
# 0:Background, 1:Building, 2:Road, 3:Water, 4:Barren, 5:Forest, 6:Agri
COLOR_MAP = {
    0: [0, 0, 0],       # Background (Black)
    1: [255, 0, 0],     # Building (Red)
    2: [255, 255, 0],   # Road (Yellow)
    3: [0, 0, 255],     # Water (Blue)
    4: [128, 0, 128],   # Barren (Purple)
    5: [0, 255, 0],     # Forest (Green)
    6: [0, 255, 255]    # Agriculture (Cyan)
}

def colorize_mask(mask):
    """
    Converts a 2D mask (indices) into a 3D RGB image.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in COLOR_MAP.items():
        # Find pixels belonging to this class
        indices = (mask == class_id)
        color_mask[indices] = color
        
    return color_mask

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load the trained model
    print(f"Loading model from {MODEL_PATH}...")
    model = DilatedAttentionNetwork(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Set to evaluation mode

    # 2. Load Test Data
    # We use 'Val' split here to compare Prediction vs Ground Truth
    test_ds = LoveDADataset(DATA_DIR, split='Val') 
    loader = DataLoader(test_ds, batch_size=1, shuffle=True)

    print("Generating predictions...")
    
    # 3. Visualizing 5 Random Samples
    with torch.no_grad():
        for i, (img_tensor, mask_tensor) in enumerate(loader):
            if i >= 5: break # Stop after 5 images
            
            img_input = img_tensor.to(DEVICE)
            
            # Predict
            output = model(img_input)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Get Original Image and Ground Truth for comparison
            # Undo normalization for visualization
            orig_img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            orig_img = (orig_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            orig_img = np.clip(orig_img, 0, 1)
            
            true_mask = mask_tensor.squeeze().cpu().numpy()

            # Colorize
            pred_color = colorize_mask(pred_mask)
            true_color = colorize_mask(true_mask)

            # Plotting
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(orig_img)
            axs[0].set_title("Satellite Input")
            axs[0].axis('off')
            
            axs[1].imshow(true_color)
            axs[1].set_title("Ground Truth (Actual)")
            axs[1].axis('off')
            
            axs[2].imshow(pred_color)
            axs[2].set_title("AI Prediction")
            axs[2].axis('off')
            
            save_path = os.path.join(OUTPUT_DIR, f"result_{i}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved visualization to {save_path}")

    print("Done! Check the 'output/predictions' folder.")

if __name__ == "__main__":
    main()