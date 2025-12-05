# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom MobileNetV3-YOLO model integration."""

import torch
import torch.nn as nn

from ultralytics.nn.modules import CSPResNetBackbone, YOLONeckP2Enhanced, YOLONeckP2EnhancedV2, Detect, Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER


class EnhancedDetectHead(nn.Module):
    """Enhanced detection head with P2 support (4-scale detection)."""
    
    def __init__(self, nc, ch):
        """Initialize enhanced detection head.
        
        Args:
            nc (int): Number of classes
            ch (tuple): Input channels for each detection level [P2, P3, P4, P5]
        """
        super().__init__()
        
        # Single refinement layer before detection for each scale
        self.pre_detect_p2 = Conv(ch[0], ch[0], k=3, s=1)
        self.pre_detect_p3 = Conv(ch[1], ch[1], k=3, s=1)
        self.pre_detect_p4 = Conv(ch[2], ch[2], k=3, s=1)
        self.pre_detect_p5 = Conv(ch[3], ch[3], k=3, s=1)
        
        # Standard YOLO detect head with 4 scales
        self.detect = Detect(nc=nc, ch=ch)
        
    def forward(self, x):
        """Forward pass through enhanced detection head.
        
        Args:
            x (list): Feature maps from neck [P2, P3, P4, P5]
            
        Returns:
            Detection output
        """
        # Refine features before detection
        x2 = self.pre_detect_p2(x[0])
        x3 = self.pre_detect_p3(x[1])
        x4 = self.pre_detect_p4(x[2])
        x5 = self.pre_detect_p5(x[3])
        
        # Detection
        return self.detect([x2, x3, x4, x5])


class MobileNetV3YOLO(nn.Module):
    """Custom YOLO model with CSPResNet backbone for timber defect detection.
    
    This model combines:
    - CSPResNet backbone with ECA attention (better feature extraction)
    - P2-enhanced neck with FPN+PAN (4-scale detection)
    - Enhanced detection head (4 detection scales: P2, P3, P4, P5)
    
    Optimized for timber defect detection:
    - P2 (160x160): Small defects like cracks
    - P3 (80x80): Medium defects like small knots
    - P4 (40x40): Large defects like big knots
    - P5 (20x20): Very large defects and context
    
    Attributes:
        backbone (CSPResNetBackbone): Feature extraction backbone
        neck (YOLONeckP2Enhanced): Feature fusion neck
        head (EnhancedDetectHead): Detection head with 4 scales
        stride (torch.Tensor): Model stride values [4, 8, 16, 32]
        names (dict): Class names
        model (nn.ModuleList): Sequential list of modules for compatibility with v8DetectionLoss
    """
    
    def __init__(self, nc=80, pretrained=True, verbose=True):
        """Initialize MobileNetV3-YOLO model.
        
        Args:
            nc (int): Number of classes
            pretrained (bool): Use pretrained MobileNetV3 backbone
            verbose (bool): Print model information
        """
        super().__init__()
        self.nc = nc
        self.task = 'detect'  # Task type
        self.names = {i: f"{i}" for i in range(nc)}
        
        # Initialize backbone and neck
        self.backbone = CSPResNetBackbone(pretrained=pretrained)
        self.neck = YOLONeckP2Enhanced(in_channels=self.backbone.out_channels)
        
        # Neck output channels: [64, 96, 128, 160] for [P2, P3, P4, P5]
        neck_out_channels = self.neck.out_channels  # [64, 96, 128, 160]
        
        # Initialize enhanced detection head with 4 scales
        self.head = EnhancedDetectHead(nc=nc, ch=tuple(neck_out_channels))
        
        # Create model list for compatibility with v8DetectionLoss
        # It expects model.model[-1] to be the Detect head
        self.model = nn.ModuleList([self.backbone, self.neck, self.head.detect])
        
        # Store args for loss function (will be set by trainer)
        self.args = None
        
        # Initialize head
        self._initialize_head()
        
        if verbose:
            self.info()
        
        # Model metadata
        self.stride = torch.tensor([4, 8, 16, 32])  # P2, P3, P4, P5 strides
        self.yaml = {'nc': nc, 'custom_model': 'cspresnet-yolo-p2'}  # Model config
        self.args = {}  # Training arguments (set by trainer)
        self.pt_path = None  # Path to checkpoint
        
        # Initialize head
        self._initialize_head()
        
        if verbose:
            self.info()
    
    def _initialize_head(self):
        """Initialize detection head with proper strides."""
        m = self.head.detect  # Access Detect inside EnhancedDetectHead
        if isinstance(m, Detect):
            s = 640  # default image size
            m.inplace = True
            
            # Forward pass to get strides
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))])
            self.stride = m.stride
            m.bias_init()  # Initialize biases
    
    def forward(self, x, *args, **kwargs):
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor | dict): Input tensor for inference or batch dict for training
            
        Returns:
            (torch.Tensor | tuple): Detection outputs or (loss, loss_items) for training
        """
        # Training mode - if input is dict, compute loss
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        
        # Inference mode - standard forward pass
        # Backbone: extract multi-scale features
        feats = self.backbone(x)  # [P2, P3, P4, P5] with channels [64, 128, 256, 384]
        
        # Neck: fuse features
        fused_feats = self.neck(feats)  # [P2, P3, P4, P5] with channels [64, 96, 128, 160]
        
        # Head: generate predictions
        outputs = self.head(fused_feats)
        
        return outputs
    
    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
            
        Returns:
            (tuple): (total_loss, loss_items)
        """
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)
    
    def init_criterion(self):
        """Initialize the loss criterion for the detection model."""
        from ultralytics.utils.loss import v8DetectionLoss
        return v8DetectionLoss(self)
    
    def fuse(self, verbose=True):
        """Fuse Conv2d + BatchNorm2d layers for inference optimization.
        
        Args:
            verbose (bool): Print fusion information
            
        Returns:
            (MobileNetV3YOLO): Fused model
        """
        if not self.is_fused():
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)) and hasattr(m, 'bn'):
                    # Fuse convolution and batch norm
                    from ultralytics.utils.torch_utils import fuse_conv_and_bn
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
            if verbose:
                self.info()
        return self
    
    def is_fused(self, thresh=10):
        """Check if model has less than threshold BatchNorm layers.
        
        Args:
            thresh (int): Threshold for BatchNorm layers
            
        Returns:
            (bool): True if fused
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh
    
    def info(self, detailed=False, verbose=True, imgsz=640):
        """Print model information.
        
        Args:
            detailed (bool): Show detailed layer information
            verbose (bool): Print to console
            imgsz (int): Input image size for computation
        """
        from ultralytics.utils.torch_utils import model_info
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)
    
    def predict(self, x, profile=False, visualize=False, augment=False):
        """Perform inference.
        
        Args:
            x (torch.Tensor): Input tensor
            profile (bool): Profile computation time
            visualize (bool): Visualize features
            augment (bool): Apply test-time augmentation
            
        Returns:
            (torch.Tensor): Model predictions
        """
        if augment:
            # Simple TTA: original + horizontal flip
            y = []
            for xi in [x, torch.flip(x, [-1])]:
                yi = self.forward(xi)
                y.append(yi)
            return torch.cat(y, -1)
        return self.forward(x)
    
    def load(self, weights):
        """Load weights from checkpoint.
        
        Args:
            weights (str | dict): Path to checkpoint or state dict
        """
        if isinstance(weights, str):
            import torch
            ckpt = torch.load(weights, map_location='cpu')
            if isinstance(ckpt, dict):
                state_dict = ckpt.get('model', ckpt)
                if hasattr(state_dict, 'state_dict'):
                    state_dict = state_dict.state_dict()
            else:
                state_dict = ckpt
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(weights, strict=False)


class MobileNetV3YOLOV2(nn.Module):
    """
    Priority 2: Enhanced YOLO with Deformable Convolutions and Coordinate Attention.
    
    Improvements over V1:
    - Deformable convolutions in neck PAN pathway (adapts to irregular defect shapes)
    - Coordinate Attention in P4/P5 (position-aware feature refinement)
    - Better for: curved cracks, odd-shaped knots, spatially-varying defects
    
    Expected performance gain: +1.5% mAP over Priority 1
    - Deformable convs: +1.0% mAP
    - Coordinate attention: +0.5% mAP
    
    Architecture:
    - Backbone: CSPResNet with ECA attention (same as V1)
    - Neck: YOLONeckP2EnhancedV2 (deformable convs + CAM)
    - Head: EnhancedDetectHead (same as V1)
    - Detection scales: P2, P3, P4, P5
    """
    
    def __init__(self, nc=80, pretrained=True, verbose=True):
        """Initialize Priority 2 model.
        
        Args:
            nc (int): Number of classes
            pretrained (bool): Use pretrained backbone (not used for CSPResNet)
            verbose (bool): Print model information
        """
        super().__init__()
        self.nc = nc
        self.task = 'detect'
        self.names = {i: f"{i}" for i in range(nc)}
        
        # Backbone - same as Priority 1
        self.backbone = CSPResNetBackbone(pretrained=pretrained)
        
        # Neck - Priority 2: Enhanced with deformable convs and coordinate attention
        self.neck = YOLONeckP2EnhancedV2(in_channels=self.backbone.out_channels)
        
        # Neck output channels: [64, 96, 128, 160] for [P2, P3, P4, P5]
        neck_out_channels = self.neck.out_channels
        
        # Head - same as Priority 1
        self.head = EnhancedDetectHead(nc=nc, ch=tuple(neck_out_channels))
        
        # Model list for v8DetectionLoss compatibility
        self.model = nn.ModuleList([self.backbone, self.neck, self.head.detect])
        
        # Store args for loss function
        self.args = None
        
        # Model metadata
        self.stride = torch.tensor([4, 8, 16, 32])  # P2, P3, P4, P5 strides
        self.yaml = {'nc': nc, 'custom_model': 'cspresnet-yolo-p2-v2'}
        self.args = {}
        self.pt_path = None
        
        # Initialize head
        self._initialize_head()
        
        if verbose:
            self.info()
    
    def _initialize_head(self):
        """Initialize detection head with proper strides."""
        m = self.head.detect
        if isinstance(m, Detect):
            s = 640
            m.inplace = True
            
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))])
            self.stride = m.stride
            m.bias_init()
    
    def forward(self, x, *args, **kwargs):
        """Forward pass."""
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        
        # Backbone
        feats = self.backbone(x)  # [P2, P3, P4, P5]: [64, 128, 256, 384]
        
        # Neck - V2 with deformable convs and coordinate attention
        fused_feats = self.neck(feats)  # [P2, P3, P4, P5]: [64, 96, 128, 160]
        
        # Head
        outputs = self.head(fused_feats)
        
        return outputs
    
    def loss(self, batch, preds=None):
        """Compute loss."""
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.criterion = self.init_criterion()
        
        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)
    
    def init_criterion(self):
        """Initialize loss criterion."""
        from ultralytics.utils.loss import v8DetectionLoss
        return v8DetectionLoss(self)
    
    def fuse(self, verbose=True):
        """Fuse Conv2d + BatchNorm2d layers."""
        if not self.is_fused():
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)) and hasattr(m, 'bn'):
                    from ultralytics.utils.torch_utils import fuse_conv_and_bn
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
            if verbose:
                self.info()
        return self
    
    def is_fused(self, thresh=10):
        """Check if model is fused."""
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh
    
    def info(self, detailed=False, verbose=True, imgsz=640):
        """Print model information."""
        from ultralytics.utils.torch_utils import model_info
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)
    
    def predict(self, x, profile=False, visualize=False, augment=False):
        """Perform inference."""
        if augment:
            y = []
            for xi in [x, torch.flip(x, [-1])]:
                yi = self.forward(xi)
                y.append(yi)
            return torch.cat(y, -1)
        return self.forward(x)
    
    def load(self, weights):
        """Load weights."""
        if isinstance(weights, str):
            import torch
            ckpt = torch.load(weights, map_location='cpu')
            if isinstance(ckpt, dict):
                state_dict = ckpt.get('model', ckpt)
                if hasattr(state_dict, 'state_dict'):
                    state_dict = state_dict.state_dict()
            else:
                state_dict = ckpt
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(weights, strict=False)

