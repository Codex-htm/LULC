import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

# --- 1. The Attention Module (CBAM) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP to learn channel importance
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Compresses channels to 2 (Avg + Max)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply Channel Attention
        out = x * self.ca(x)
        # Apply Spatial Attention
        out = out * self.sa(out)
        return out

# --- 2. The Main Network (Dilated DeepLab + Attention) ---
class DilatedAttentionNetwork(nn.Module):
    def __init__(self, num_classes=7):
        super(DilatedAttentionNetwork, self).__init__()
        
        # A. Load Standard DeepLabV3 with ResNet50 backbone
        # We don't need pretrained weights during inference, saving 160MB+ RAM!
        self.base_model = deeplabv3_resnet50(weights=None)
        
        # B. Modify the Classifier Head (The output layer)
        # DeepLab's classifier is: (256 channels -> num_classes)
        # We intercept the features before the final classification
        self.classifier = self.base_model.classifier
        
        # C. Insert Attention Mechanism
        # The classifier input has 2048 channels (from ResNet50 backbone)
        self.attention = CBAMBlock(in_planes=2048)
        
        # D. Final Output Layer for our 7 classes
        self.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        # Extract features using the backbone
        # The output is a dictionary
        input_shape = x.shape[-2:]
        features = self.base_model.backbone(x)
        x = features['out'] # This is the feature map with 2048 channels
        
        # Apply YOUR Attention Mechanism
        x = self.attention(x)
        
        # Pass through the DeepLab ASPP classifier
        x = self.classifier(x)
        
        # Resize back to original image size (Bilinear Interpolation)
        x = torch.nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x

# --- Quick Test ---
if __name__ == "__main__":
    # Simulate an image batch: 2 images, 3 channels, 512x512 size
    dummy_input = torch.randn(2, 3, 512, 512)
    
    model = DilatedAttentionNetwork(num_classes=7)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [2, 7, 512, 512]
    print("Model built successfully with Attention!")