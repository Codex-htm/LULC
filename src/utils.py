import torch
import numpy as np

def calculate_iou(pred, label, num_classes=7):
    """
    Calculates Intersection over Union (IoU) for each class.
    """
    pred = torch.argmax(pred, dim=1).flatten()
    label = label.flatten()
    
    # Ignore '255' (No Data) pixels
    valid_mask = label != 255
    pred = pred[valid_mask]
    label = label[valid_mask]

    iou_list = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = label == cls
        
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            iou_list.append(np.nan) # No pixels for this class in this batch
        else:
            iou_list.append(intersection / union)
            
    return np.nanmean(iou_list) # Return Mean IoU