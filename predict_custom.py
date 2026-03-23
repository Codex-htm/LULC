import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms as T
from src.model import DilatedAttentionNetwork

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "output/best_model.pth"
# Use 'r' for raw string to fix path errors
INPUT_IMAGE_PATH = r"D:\2025\LULC\LandCover_Project\data\LoveDA\Test\Rural\images_png\4211.png"
OUTPUT_FOLDER = "output/custom_predictions" # New folder for saving results

# Define Colors
COLOR_MAP = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [255, 255, 0], 
    3: [0, 0, 255], 4: [128, 0, 128], 5: [0, 255, 0], 6: [0, 255, 255]
}

CLASS_NAMES = {
    0: "Background", 1: "Building", 2: "Road", 
    3: "Water", 4: "Barren", 5: "Forest", 6: "Agriculture"
}

def main():
    # 0. Setup Output Folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 1. Load Model
    print(f"Loading model...")
    model = DilatedAttentionNetwork(num_classes=7).to(DEVICE)
    # weights_only=True is safer for future versions
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # Delegate to reusable function
    input_path, output_path = predict_image()
    print(f"✅ Result saved to: {os.path.join(OUTPUT_FOLDER, output_path)}")
    # Show interactively when run as script
    try:
        from matplotlib import image as mpimg
        display_input = mpimg.imread(os.path.join(OUTPUT_FOLDER, input_path))
        display_output = mpimg.imread(os.path.join(OUTPUT_FOLDER, output_path))
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.imshow(display_input)
        plt.title('Input Image')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(display_output)
        plt.title('AI Output Map')
        plt.axis('off')
        plt.show()
    except Exception:
        pass


def predict_image(input_image_path=None, model=None, model_path=None, output_folder=None, device=None):
    """Run prediction for a single image and save both the resized input and the colorized prediction into output_folder.

    Returns: (input_filename, output_filename, stats_dict) relative to output_folder.
    stats_dict: {class_name: percentage}
    """
    if input_image_path is None:
        input_image_path = INPUT_IMAGE_PATH
    if model_path is None:
        model_path = MODEL_PATH
    if output_folder is None:
        output_folder = OUTPUT_FOLDER
    if device is None:
        device = DEVICE

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Use existing model if provided, otherwise load from disk
    if model is None:
        model = DilatedAttentionNetwork(num_classes=7).to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        model.eval()

    # Read image
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {input_image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = cv2.resize(image, (512, 512))

    # Preprocess
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(display_image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Colorize
    color_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        color_mask[pred_mask == class_id] = color

    # Save resized input and color mask
    base_name = os.path.basename(input_image_path)
    input_save_name = f"input_{os.path.splitext(base_name)[0]}.png"
    out_save_name = f"prediction_{os.path.splitext(base_name)[0]}.png"
    input_save_path = os.path.join(output_folder, input_save_name)
    out_save_path = os.path.join(output_folder, out_save_name)

    plt.imsave(input_save_path, display_image)
    plt.imsave(out_save_path, color_mask)

    # Calculate statistics (percentages)
    total_pixels = pred_mask.size
    stats = {}
    for class_id, name in CLASS_NAMES.items():
        count = np.sum(pred_mask == class_id)
        percentage = (count / total_pixels) * 100
        stats[name] = round(float(percentage), 2)

    return input_save_name, out_save_name, stats

if __name__ == "__main__":
    main()