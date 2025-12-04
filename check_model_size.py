"""Check enhanced model size and architecture."""

from ultralytics import YOLO

# Load model
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')

# Print model info
print("\n" + "="*80)
print("ENHANCED MODEL INFORMATION")
print("="*80)
model.model.info(detailed=False, verbose=True)

print("\n" + "="*80)
print("PARAMETER BREAKDOWN")
print("="*80)

# Count parameters by component
backbone_params = sum(p.numel() for p in model.model.backbone.parameters())
neck_params = sum(p.numel() for p in model.model.neck.parameters())
head_params = sum(p.numel() for p in model.model.head.parameters())
total_params = sum(p.numel() for p in model.model.parameters())

print(f"Backbone parameters: {backbone_params:,} ({backbone_params/1e6:.2f}M)")
print(f"Neck parameters:     {neck_params:,} ({neck_params/1e6:.2f}M)")
print(f"Head parameters:     {head_params:,} ({head_params/1e6:.2f}M)")
print(f"Total parameters:    {total_params:,} ({total_params/1e6:.2f}M)")

# Check if under 5M
if total_params < 5_000_000:
    print(f"\n✓ Model is under 5M parameters! ({total_params/1e6:.2f}M)")
    remaining = 5_000_000 - total_params
    print(f"  Remaining budget: {remaining:,} ({remaining/1e6:.2f}M)")
else:
    print(f"\n✗ Model exceeds 5M parameters! ({total_params/1e6:.2f}M)")
    excess = total_params - 5_000_000
    print(f"  Over budget by: {excess:,} ({excess/1e6:.2f}M)")
