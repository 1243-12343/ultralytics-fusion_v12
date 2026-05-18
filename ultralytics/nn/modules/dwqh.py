import torch
import torch.nn as nn
import torch.nn.functional as F

class DWQH(nn.Module):
    """Dynamic-Weight-based Quad-Head Detector (DWQH) module."""
    
    def __init__(self, in_channels, out_channels):
        """Initialize DWQH module.
        
        Args:
            in_channels (int): Number of input channels for each feature level
            out_channels (int): Number of output channels after fusion
        """
        super().__init__()
        
        # 1x1 convolutions for channel unification
        self.conv1x1_l2 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1x1_l3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1x1_l4 = nn.Conv2d(in_channels, out_channels, 1)
        
        # Feature alignment layers
        self.align_l2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.align_l3 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
        # Learnable weight parameters
        self.weight_params = nn.Parameter(torch.ones(3, out_channels))
        
    def forward(self, x2, x3, x4):
        """Forward pass of DWQH.
        
        Args:
            x2 (torch.Tensor): Low-level features
            x3 (torch.Tensor): Mid-level features
            x4 (torch.Tensor): High-level features
            
        Returns:
            torch.Tensor: Fused features with dynamic weights
        """
        # Channel unification
        x2 = self.conv1x1_l2(x2)
        x3 = self.conv1x1_l3(x3)
        x4 = self.conv1x1_l4(x4)
        
        # Feature alignment
        x2_aligned = self.align_l2(x2)  # 4x downsampling
        x3_aligned = self.align_l3(x3)  # 2x downsampling
        x4_aligned = x4  # original resolution
        
        # Generate dynamic weights using softmax
        weights = F.softmax(self.weight_params, dim=0)  # [3, out_channels]
        alpha, beta, gamma = weights[0], weights[1], weights[2]
        
        # Apply weights channel-wise and sum
        fused_features = (
            alpha.view(1, -1, 1, 1) * x2_aligned +
            beta.view(1, -1, 1, 1) * x3_aligned +
            gamma.view(1, -1, 1, 1) * x4_aligned
        )
        
        return fused_features 