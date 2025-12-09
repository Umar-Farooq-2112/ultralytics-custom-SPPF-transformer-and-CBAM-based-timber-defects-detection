from ultralytics import YOLO
import torch

print("="*80)
print("MobileNetV3-YOLO Input Size Comparison Test")
print("="*80)

# Load model
print("\nLoading model from YAML config...")
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
print("✓ Model loaded successfully!")

# Test different input sizes
input_sizes = [640, 480, 416, 320]

print("\n" + "="*80)
print("Testing Different Input Sizes")
print("="*80)

results = []

for size in input_sizes:
    print(f"\n{'='*80}")
    print(f"Testing with input size: {size}×{size}")
    print(f"{'='*80}")
    
    # Create test input
    x = torch.randn(1, 3, size, size)
    
    # Test forward pass
    print(f"1. Testing forward pass...")
    with torch.no_grad():
        outputs = model.model(x)
    
    print(f"   ✓ Forward pass successful!")
    print(f"   - Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"   - Output {i} (P{i+3}): {out.shape}")
    
    # Get model info
    print(f"\n2. Model Summary for {size}×{size}:")
    model.info(detailed=False, verbose=False)
    
    # Calculate FLOPs manually
    from thop import profile
    flops, params = profile(model.model, inputs=(x,), verbose=False)
    gflops = flops / 1e9
    
    print(f"\n   Detailed Metrics:")
    print(f"   - Parameters: {params:,.0f}")
    print(f"   - GFLOPs: {gflops:.2f}")
    print(f"   - Memory (input): {x.element_size() * x.nelement() / 1024**2:.2f} MB")
    
    results.append({
        'size': size,
        'params': params,
        'gflops': gflops,
        'outputs': [out.shape for out in outputs]
    })

# Summary comparison
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print(f"\n{'Input Size':<15} {'Parameters':<15} {'GFLOPs':<12} {'Reduction':<12} {'Speed Gain'}")
print("-" * 80)

baseline_gflops = results[0]['gflops']
for r in results:
    reduction = ((baseline_gflops - r['gflops']) / baseline_gflops) * 100
    speed_gain = baseline_gflops / r['gflops']
    print(f"{r['size']}×{r['size']:<8} {r['params']:>12,.0f}   {r['gflops']:>8.2f}    {reduction:>8.1f}%     {speed_gain:>6.2f}×")

print("\n" + "="*80)
print("Feature Map Resolutions at Different Input Sizes")
print("="*80)

for r in results:
    size = r['size']
    print(f"\n{size}×{size} input:")
    for i, shape in enumerate(r['outputs']):
        stride = 2 ** (i + 3)  # P3=8, P4=16, P5=32
        print(f"  P{i+3} (stride {stride:>2}): {shape[2]:>3}×{shape[3]:<3} = {shape[2]*shape[3]:>5} pixels")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print(f"""
Based on the results:

1. 640×640 (Baseline):
   - Best accuracy for small defects
   - GFLOPs: {results[0]['gflops']:.2f}
   - Use for: Final inspection, high-accuracy requirements

2. 480×480:
   - Balanced speed/accuracy
   - GFLOPs: {results[1]['gflops']:.2f} ({((baseline_gflops - results[1]['gflops']) / baseline_gflops * 100):.1f}% reduction)
   - Use for: General purpose detection

3. 416×416:
   - Good compromise
   - GFLOPs: {results[2]['gflops']:.2f} ({((baseline_gflops - results[2]['gflops']) / baseline_gflops * 100):.1f}% reduction)
   - Use for: Real-time processing

4. 320×320:
   - Maximum speed
   - GFLOPs: {results[3]['gflops']:.2f} ({((baseline_gflops - results[3]['gflops']) / baseline_gflops * 100):.1f}% reduction)
   - Use for: Fast screening, edge devices
   - Warning: Small defect detection may suffer
""")

print("="*80)
print("✅ Analysis Complete!")
print("="*80)
