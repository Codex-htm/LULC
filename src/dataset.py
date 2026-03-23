import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms as T

class LoveDADataset(Dataset):
    def __init__(self, root_dir, split='Train', image_size=512):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.images = []
        self.masks = []
        
        domains = ['Urban', 'Rural']
        for domain in domains:
            img_folder = os.path.join(root_dir, split, domain, 'images_png')
            mask_folder = os.path.join(root_dir, split, domain, 'masks_png')
            
            if not os.path.exists(img_folder):
                continue
            
            files = sorted(os.listdir(img_folder))
            for f in files:
                if f.endswith('.png'):
                    self.images.append(os.path.join(img_folder, f))
                    if split != 'Test':
                        self.masks.append(os.path.join(mask_folder, f))

        print(f"Split: {split} | Found {len(self.images)} images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Prepare Mask (only for Train/Val)
        mask = None
        if self.split != 'Test':
            mask_path = self.masks[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            
            # Remap Labels: 1-7 -> 0-6
            mask = mask.astype(np.int64)
            mask = mask - 1
            mask[mask == -1] = 255

        # --- DATA AUGMENTATION (Only for Training) ---
        if self.split == 'Train':
            # 1. Random Scale (Zoom In/Out)
            if random.random() > 0.5:
                scale = random.uniform(0.8, 1.2)
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                
                image = cv2.resize(image, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
                # Crop or Pad to restore 512x512
                if scale > 1.0: # Crop Center
                    start_h = (new_h - self.image_size) // 2
                    start_w = (new_w - self.image_size) // 2
                    image = image[start_h:start_h+self.image_size, start_w:start_w+self.image_size]
                    mask = mask[start_h:start_h+self.image_size, start_w:start_w+self.image_size]
                else: # Pad
                    pad_h = (self.image_size - new_h) // 2
                    pad_w = (self.image_size - new_w) // 2
                    # Pad with reflection for image, constant for mask
                    image = cv2.copyMakeBorder(image, pad_h, self.image_size-new_h-pad_h, pad_w, self.image_size-new_w-pad_w, cv2.BORDER_REFLECT)
                    mask = cv2.copyMakeBorder(mask, pad_h, self.image_size-new_h-pad_h, pad_w, self.image_size-new_w-pad_w, cv2.BORDER_CONSTANT, value=255)

            # 2. Random Horizontal Flip
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            
            # 3. Random Vertical Flip
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
                
            # 4. Random Rotation (0, 90, 180, 270)
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()
        # ---------------------------------------------

        # Normalize & Photometric Augmentations
        transforms_list = [T.ToTensor()]
        
        if self.split == 'Train':
            transforms_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
            if random.random() > 0.8:
                transforms_list.append(T.GaussianBlur(kernel_size=3))
                
        transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        t_transform = T.Compose(transforms_list)
        image = t_transform(image)

        if self.split != 'Test':
            mask = torch.from_numpy(mask).long()
            return image, mask
        
        return image, img_path