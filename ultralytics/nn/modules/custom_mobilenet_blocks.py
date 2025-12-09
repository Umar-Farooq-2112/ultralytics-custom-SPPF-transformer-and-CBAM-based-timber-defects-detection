# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom MobileNetV3-based blocks for lightweight YOLO models."""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

__all__ = (
    "DWConvCustom",
    "ConvBNAct",
    "CBAM_ChannelOnly",
    "P5Transformer",
    "MobileNetV3BackboneDW",
    "UltraLiteNeckDW",
)


class DWConvCustom(nn.Module):
    """Depthwise Separable Conv: depthwise 3x3 + pointwise 1x1."""
    
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, norm=True, activation=True):
        """Initialize depthwise separable convolution.
        
        Args:
            in_ch (int): Input channels
            out_ch (int): Output channels
            kernel_size (int): Kernel size for depthwise conv
            stride (int): Stride
            padding (int): Padding
            bias (bool): Use bias
            norm (bool): Use batch normalization
            activation (bool): Use activation
        """
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)
        self.use_norm = norm
        if norm:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = nn.Identity()
        self.use_act = activation
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        """Forward pass through depthwise separable convolution."""
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvBNAct(nn.Module):
    """Standard conv 1x1 or 3x3 with BatchNorm and activation."""
    
    def __init__(self, in_ch, out_ch, k=1, stride=1, padding=0, act=True):
        """Initialize standard convolution block.
        
        Args:
            in_ch (int): Input channels
            out_ch (int): Output channels
            k (int): Kernel size
            stride (int): Stride
            padding (int): Padding
            act (bool): Use activation
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        """Forward pass through conv-bn-act."""
        return self.act(self.bn(self.conv(x)))


class CBAM_ChannelOnly(nn.Module):
    """Enhanced CBAM with channel and spatial attention for better feature refinement."""
    
    def __init__(self, channels, reduction=8):
        """Initialize enhanced CBAM module.
        
        Args:
            channels (int): Number of input channels
            reduction (int): Channel reduction ratio
        """
        super().__init__()
        # Enhanced channel attention with dual pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(1, channels // reduction)
        
        # Deeper channel attention network (no BatchNorm in 1x1 spatial dims)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention for location-aware refinement
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass with channel and spatial attention."""
        # Channel attention using both avg and max pooling
        avg_attn = self.channel_fc(self.avg_pool(x))
        max_attn = self.channel_fc(self.max_pool(x))
        channel_attn = avg_attn + max_attn
        x = x * channel_attn
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_attn = self.spatial_conv(spatial_input)
        x = x * spatial_attn
        
        return x


class P5Transformer(nn.Module):
    """Small transformer for P5 features."""
    
    def __init__(self, in_channels, embed_dim=48, ff_dim=96, num_layers=2):
        """Initialize transformer module for P5 features.
        
        Args:
            in_channels (int): Input channels
            embed_dim (int): Embedding dimension
            ff_dim (int): Feedforward dimension
            num_layers (int): Number of transformer layers
        """
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=1,
                dim_feedforward=ff_dim,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
    
    def forward(self, x):
        """Forward pass through transformer."""
        b, c, h, w = x.shape
        x = self.proj(x)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        for layer in self.layers:
            tokens = layer(tokens)
        x = tokens.transpose(1, 2).view(b, -1, h, w)
        return self.out_proj(x)


class MobileNetV3BackboneDW(nn.Module):
    """Enhanced MobileNetV3 Small backbone with multi-scale feature enhancement."""
    
    def __init__(self, pretrained=True):
        """Initialize enhanced MobileNetV3 backbone.
        
        Args:
            pretrained (bool): Use pretrained weights
        """
        super().__init__()
        m = mobilenet_v3_small(pretrained=pretrained)
        feats = m.features

        # Stage slicing for multi-scale features
        # feats[:3]  -> P3 (stride 8),  out_channels = 24
        # feats[3:7] -> P4 (stride 16), out_channels = 40
        # feats[7:]  -> P5 (stride 32), out_channels = 576
        self.stage1 = feats[:3]
        self.stage2 = feats[3:7]
        self.stage3 = feats[7:]

        # Enhanced P3 processing - critical for small defects (reduced layers)
        self.conv_p3_1 = DWConvCustom(24, 48, kernel_size=3, stride=1, padding=1)
        self.conv_p3_2 = DWConvCustom(48, 64, kernel_size=3, stride=1, padding=1)
        self.conv_p3_3 = DWConvCustom(64, 64, kernel_size=3, stride=1, padding=1)
        self.cbam_p3 = CBAM_ChannelOnly(64, reduction=4)
        
        # Enhanced P4 processing - balanced features (reduced layers)
        self.conv_p4_1 = DWConvCustom(40, 80, kernel_size=3, stride=1, padding=1)
        self.conv_p4_2 = DWConvCustom(80, 128, kernel_size=3, stride=1, padding=1)
        self.conv_p4_3 = DWConvCustom(128, 128, kernel_size=3, stride=1, padding=1)
        self.cbam_p4 = CBAM_ChannelOnly(128, reduction=4)
        
        # Enhanced P5 processing - context and large defects (reduced layers)
        self.conv_p5_1 = DWConvCustom(576, 192, kernel_size=3, stride=1, padding=1)
        self.conv_p5_2 = DWConvCustom(192, 256, kernel_size=3, stride=1, padding=1)
        self.conv_p5_3 = DWConvCustom(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_p5_4 = DWConvCustom(256, 256, kernel_size=3, stride=1, padding=1)
        self.cbam_p5 = CBAM_ChannelOnly(256, reduction=4)

        # Output channels for each stage (significantly increased)
        self.out_channels = [64, 128, 256]

    def forward(self, x):
        """Forward pass through enhanced backbone.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            list: Multi-scale features [P3, P4, P5]
        """
        # Extract features from MobileNetV3 stages first
        p3_base = self.stage1(x)      # 24 channels
        p4_base = self.stage2(p3_base)  # 40 channels  
        p5_base = self.stage3(p4_base)  # 576 channels
        
        # Then enhance each level independently (reduced from 5-6 layers to 3-4 layers)
        # P3 path - preserve fine details for small defects (3 conv layers, no residual)
        p3 = self.conv_p3_1(p3_base)
        p3 = self.conv_p3_2(p3)
        p3 = self.conv_p3_3(p3)
        p3 = self.cbam_p3(p3)
        
        # P4 path - balanced feature extraction (3 conv layers, no residual)
        p4 = self.conv_p4_1(p4_base)
        p4 = self.conv_p4_2(p4)
        p4 = self.conv_p4_3(p4)
        p4 = self.cbam_p4(p4)
        
        # P5 path - deep context for large defects (4 conv layers, no residual)
        p5 = self.conv_p5_1(p5_base)
        p5 = self.conv_p5_2(p5)
        p5 = self.conv_p5_3(p5)
        p5 = self.conv_p5_4(p5)
        p5 = self.cbam_p5(p5)
        
        return [p3, p4, p5]


class UltraLiteNeckDW(nn.Module):
    """Enhanced neck with multi-scale fusion, attention, and context aggregation."""
    
    def __init__(self, in_channels=[64, 128, 256]):
        """Initialize enhanced neck.
        
        Args:
            in_channels (list): Input channels for [P3, P4, P5]
        """
        super().__init__()
        c3, c4, c5 = in_channels

        # P3 path - preserve fine-grained features (reduced capacity)
        self.p3_pre = DWConvCustom(c3, 96, kernel_size=3, padding=1)
        self.p3_extra1 = DWConvCustom(96, 128, kernel_size=3, padding=1)
        self.p3_cbam = CBAM_ChannelOnly(128, reduction=8)
        self.p3_refine = DWConvCustom(128, 128, kernel_size=3, padding=1)
        
        # P4 path - balanced feature processing (removed SPPF, reduced capacity)
        self.p4_pre = DWConvCustom(c4, 160, kernel_size=3, padding=1)
        self.p4_extra1 = DWConvCustom(160, 192, kernel_size=3, padding=1)
        self.p4_cbam = CBAM_ChannelOnly(192, reduction=8)
        self.p4_refine = DWConvCustom(192, 192, kernel_size=3, padding=1)
        
        # P5 path - deep context with transformer (removed SPPF, reduced transformer and channels)
        self.p5_pre = DWConvCustom(c5, 192, kernel_size=3, padding=1)
        self.p5_extra1 = DWConvCustom(192, 192, kernel_size=3, padding=1)
        self.p5_trans = P5Transformer(in_channels=192, embed_dim=96, ff_dim=192, num_layers=1)
        self.p5_cbam = CBAM_ChannelOnly(96, reduction=8)
        self.p5_refine = DWConvCustom(96, 192, kernel_size=3, padding=1)
        
        # Top-down feature fusion (FPN-style)
        self.p5_to_p4 = ConvBNAct(192, 192, k=1)
        self.p4_to_p3 = ConvBNAct(192, 128, k=1)
        
        # Bottom-up feature fusion (PAN-style)
        self.p3_to_p4 = DWConvCustom(128, 192, kernel_size=3, stride=2, padding=1)
        self.p4_to_p5 = DWConvCustom(192, 192, kernel_size=3, stride=2, padding=1)
        
        # Final output convolutions (single layer per scale for efficiency)
        self.out_p3 = DWConvCustom(128, 128, kernel_size=3, padding=1)
        self.out_p4 = DWConvCustom(192, 192, kernel_size=3, padding=1)
        self.out_p5 = DWConvCustom(192, 192, kernel_size=3, padding=1)

    def forward(self, feats):
        """Forward pass through enhanced neck with bidirectional fusion.
        
        Args:
            feats (list): Multi-scale features [P3, P4, P5]
            
        Returns:
            list: Enhanced and fused features [P3_out, P4_out, P5_out]
        """
        p3, p4, p5 = feats

        # Initial processing (optimized - no SPPF)
        p3 = self.p3_pre(p3)
        p3 = self.p3_extra1(p3)
        p3 = self.p3_cbam(p3)
        
        p4 = self.p4_pre(p4)
        p4 = self.p4_extra1(p4)
        p4 = self.p4_cbam(p4)
        
        p5 = self.p5_pre(p5)
        p5 = self.p5_extra1(p5)
        p5 = self.p5_trans(p5)
        p5 = self.p5_cbam(p5)
        p5 = self.p5_refine(p5)
        
        # Top-down fusion (coarse to fine)
        p5_up = nn.functional.interpolate(self.p5_to_p4(p5), size=p4.shape[-2:], mode='nearest')
        p4 = p4 + p5_up
        p4 = self.p4_refine(p4)
        
        p4_up = nn.functional.interpolate(self.p4_to_p3(p4), size=p3.shape[-2:], mode='nearest')
        p3 = p3 + p4_up
        p3 = self.p3_refine(p3)
        
        # Bottom-up fusion (fine to coarse) - PAN
        p3_down = self.p3_to_p4(p3)
        p4 = p4 + p3_down
        
        p4_down = self.p4_to_p5(p4)
        p5 = p5 + p4_down
        
        # Final refinement
        p3_out = self.out_p3(p3)
        p4_out = self.out_p4(p4)
        p5_out = self.out_p5(p5)

        return [p3_out, p4_out, p5_out]
