import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import our custom modules
from src.dataset import LoveDADataset
from src.model import DilatedAttentionNetwork
from src.utils import calculate_iou

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 2  # Keep at 2 for 4GB GPU
NUM_EPOCHS = 30 # Target for high accuracy
DATA_DIR = r"D:\2025\LULC\LandCover_Project\data\LoveDA"

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training")
    epoch_loss = 0
    
    for data, targets in loop:
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward Pass
        with torch.amp.autocast('cuda'): 
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward Pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return epoch_loss / len(loader)

def validate(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader, desc="Validating")
    total_iou = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, targets in loop:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
            total_loss += loss.item()
            total_iou += calculate_iou(predictions, targets)
            
    return total_loss/len(loader), total_iou/len(loader)

def main():
    print(f"Using Device: {DEVICE}")
    
    # 1. Load Data
    train_ds = LoveDADataset(DATA_DIR, split='Train')
    val_ds = LoveDADataset(DATA_DIR, split='Val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Load Model
    model = DilatedAttentionNetwork(num_classes=7).to(DEVICE)
    
    # --- RESUME TRAINING LOGIC ---
    model_save_path = "output/best_model.pth"
    
    if os.path.exists(model_save_path):
        print(f"🔄 Found saved model at {model_save_path}. Loading weights to resume training...")
        model.load_state_dict(torch.load(model_save_path))
        print("✅ Weights loaded successfully! Continuing from where we left off...")
    else:
        print("🆕 No previous model found. Starting training from scratch.")
    # -----------------------------
    
    # 3. Setup Optimization
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # Scheduler: Squeeze out more accuracy by lowering LR when stuck
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Check baseline accuracy
    best_iou = 0.0
    if os.path.exists(model_save_path):
        print("Calculating baseline accuracy of loaded model...")
        _, best_iou = validate(val_loader, model, loss_fn)
        print(f"Starting Best mIoU: {best_iou:.4f}")
    
    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        val_loss, val_iou = validate(val_loader, model, loss_fn)
        
        # Update Scheduler
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")
        
        # Save if accuracy improves
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), model_save_path)
            print(f"🔥 New Best Accuracy: {best_iou:.4f}! Model Saved.")

if __name__ == "__main__":
    main()