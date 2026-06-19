import torch
import math
import torch.nn as nn
from torchvision.ops import DeformConv2d
import torch.nn.functional as F


class NonLocalBlock(nn.Module):

    def __init__(self, in_channels, reduction = 8):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(in_channels // reduction, 1)  
        
        self.query = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.key = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in [self.query, self.key]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        # q: [B, C', H, W] -> [B, C', HW] -> [B, HW, C']
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        # k: [B, C', H, W] -> [B, C', HW]
        k = self.key(x).view(B, -1, H * W)

        # Q * K^T -> [B, HW, HW]
        affinity = torch.bmm(q, k) * (self.inter_channels ** -0.5)

        return affinity


class DeConBlock(nn.Module):
    """
    Hybrid attention block combining deformable convolution with multi-scale
    depth-wise convolutions to generate a local-detail attention map.
    """

    def __init__(self, channels, dilation_rates=(2, 4)):
        super().__init__()
        rate1, rate2 = dilation_rates

        # Deformable convolution for irregular defect shapes
        self.offset_conv = nn.Conv2d(channels, 2 * 3 * 3, kernel_size=3, padding=1)
        self.dcn = DeformConv2d(channels, channels, kernel_size=3, padding=1)

        # Depth-wise convolutions with dilation for multi-scale context
        self.dw_conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                                  padding=rate1, dilation=rate1, groups=channels)
        self.dw_conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                                  padding=rate2, dilation=rate2, groups=channels)

        self._init_weights()

    def _init_weights(self):
        for m in [self.offset_conv, self.dw_conv1, self.dw_conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # DCN offsets initialized to zero for stable start
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, HW, C] patch tokens from ViT
        Returns:
            A_local: [B, HW, HW] local-detail attention map
        """
        B, hw, dim = x.shape
        h = w = int(math.sqrt(hw))

        # Reshape to 2D: [B, HW, C] -> [B, C, H, W]
        x_img = x.transpose(1, 2).view(B, dim, h, w).contiguous()

        offset = self.offset_conv(x_img)
        geom_feat = self.dcn(x_img, offset)

        ctx_feat1 = self.dw_conv1(x_img)
        ctx_feat2 = self.dw_conv2(x_img)

        multi = torch.cat([geom_feat, ctx_feat1, ctx_feat2], dim=1)

        multi_flat = multi.flatten(2)                     
        q = multi_flat.transpose(1, 2)                    
        k = multi_flat                                     
        d_multi = multi_flat.size(1)                       
        affinity = torch.bmm(q, k) * (d_multi ** -0.5)    
        return affinity