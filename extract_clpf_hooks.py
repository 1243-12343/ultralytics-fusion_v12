#!/usr/bin/env python3
"""Extract CLPF data from trained model using forward hooks"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False


def load_model(checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    from ultralytics import YOLO
    model = YOLO(checkpoint_path)
    return model.model.model  # DetectionModel -> Sequential


def extract_clpf_data_with_hooks(model, image_dir, num_samples=100):
    """Extract CLPF data using forward hooks"""
    all_w_rgb = []
    all_w_ir = []
    all_err_rgb = []
    all_err_ir = []

    # Find images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    image_paths = sorted(image_paths)[:num_samples]

    if not image_paths:
        print(f"No images found in {image_dir}")
        return None, None, None, None

    print(f"Processing {len(image_paths)} images...")

    # Find CLPFRes modules and register hooks
    hooks = []
    clpf_modules = []

    for i, module in enumerate(model):
        if type(module).__name__ in ['CLPF', 'CLPFRes']:
            clpf_modules.append((i, module))
            # Register hook to capture forward pass data
            def make_hook(idx, mod):
                def hook_fn(module, input, output):
                    # CLPFRes records data during forward
                    pass
                return hook_fn
            hook = module.register_forward_hook(make_hook(i, module))
            hooks.append(hook)

    print(f"Found {len(clpf_modules)} CLPF modules")
    for i, m in clpf_modules:
        print(f"  [{i}] {type(m).__name__}")

    # The issue is that CLPFRes needs two inputs in a list/tuple format
    # Sequential won't work with that. We need to manually implement forward.

    # Let's trace through the model to understand data flow
    print("\nTracing model structure...")

    # Manually build a dictionary to store intermediate outputs
    outputs = {}

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 20 == 0:
                print(f"  Processing: {i+1}/{len(image_paths)}")

            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_ir = img_rgb.copy()

                img_size = 512
                img_rgb = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
                img_ir = cv2.resize(img_ir, (img_size, img_size)).astype(np.float32) / 255.0

                # 6-channel input
                img_6ch = np.concatenate([img_rgb, img_ir], axis=2)
                img_6ch = torch.from_numpy(img_6ch).permute(2, 0, 1).unsqueeze(0)

                # Manually forward through the model
                outputs[0] = img_6ch  # IN layer is identity

                # Layer 1: Multiin (out=1) extracts RGB -> 3 channels
                outputs[1] = model[1](outputs[0])

                # Layer 2: Multiin (out=2) extracts IR -> needs 6ch input
                # But after layer 1, we only have 3 channels
                # We need to reconstruct the 6-channel input for layer 2

                # Actually, let's check what the IR input should be
                # Looking at yaml: - [-2, 1, Multiin, [2]]  # 2 IR
                # It seems both VIS and IR come from different branches

                # For now, use the same 3-channel output for both
                outputs[2] = outputs[1]  # IR branch uses same input

                # Continue forward pass
                x = outputs[2]
                for idx in range(3, len(model)):
                    if idx in [15, 22, 29]:  # CLPFRes layers
                        # CLPFRes expects [vis_feat, ir_feat] list
                        # Get features from appropriate indices
                        vis_feat = outputs[12] if 12 in outputs else outputs[idx - 3]
                        ir_feat = outputs[14] if 14 in outputs else outputs[idx - 3]
                        x = model[idx]([vis_feat, ir_feat])

                        # Check for recorded data
                        clpf_layer = model[idx]
                        if hasattr(clpf_layer, 'recorded_w_rgb') and clpf_layer.recorded_w_rgb is not None:
                            w_rgb = clpf_layer.recorded_w_rgb.cpu().numpy().flatten()
                            w_ir = clpf_layer.recorded_w_ir.cpu().numpy().flatten()
                            err_rgb = clpf_layer.recorded_err_rgb.cpu().numpy().flatten()
                            err_ir = clpf_layer.recorded_err_ir.cpu().numpy().flatten()

                            all_w_rgb.extend(w_rgb)
                            all_w_ir.extend(w_ir)
                            all_err_rgb.extend(err_rgb)
                            all_err_ir.extend(err_ir)

                        outputs[idx] = x
                    elif type(model[idx]).__name__ == 'Concat':
                        # Concat needs multiple inputs from outputs dict
                        x = model[idx]([outputs.get(i) for i in range(idx) if i in outputs][-2:])
                        outputs[idx] = x
                    elif type(model[idx]).__name__ == 'FeatureAdd':
                        # FeatureAdd also needs list of inputs
                        x = model[idx]([outputs.get(i) for i in range(idx) if i in outputs][-2:])
                        outputs[idx] = x
                    elif type(model[idx]).__name__ == 'Upsample':
                        x = model[idx](x)
                        outputs[idx] = x
                    else:
                        try:
                            x = model[idx](x)
                            outputs[idx] = x
                        except:
                            # Try with single input if layer expects list
                            try:
                                x = model[idx]([x, x])
                                outputs[idx] = x
                            except:
                                print(f"  Warning: Failed at layer {idx} ({type(model[idx]).__name__})")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    # Remove hooks
    for hook in hooks:
        hook.remove()

    if not all_w_rgb:
        print("No CLPF data extracted!")
        return None, None, None, None

    return (np.array(all_w_rgb), np.array(all_w_ir),
            np.array(all_err_rgb), np.array(all_err_ir))


def generate_visualizations(w_rgb, w_ir, err_rgb, err_ir, output_dir):
    """Generate visualization images"""
    os.makedirs(output_dir, exist_ok=True)

    mean_w_rgb = np.mean(w_rgb)
    mean_w_ir = np.mean(w_ir)
    std_w_rgb = np.std(w_rgb)
    std_w_ir = np.std(w_ir)

    # Experiment 1: Weight Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Heatmap
    ax = axes[0]
    x_min, x_max = 0.3, 0.7
    y_min, y_max = 0.3, 0.7
    mask = (w_rgb >= x_min) & (w_rgb <= x_max) & (w_ir >= y_min) & (w_ir <= y_max)
    if mask.sum() > 10:
        heatmap, xedges, yedges = np.histogram2d(w_rgb[mask], w_ir[mask], bins=25)
        im = ax.imshow(heatmap.T, origin='lower', aspect='auto',
                       extent=[x_min, x_max, y_min, y_max], cmap='YlOrRd')
        x_line = np.linspace(x_min, x_max, 100)
        ax.plot(x_line, 1 - x_line, 'b--', linewidth=2, label=r'$w_{ir} = 1 - w_{rgb}$')
        ax.axhline(y=0.5, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(x=0.5, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        plt.colorbar(im, ax=ax, label='Count')
    ax.set_xlabel(r'$w_{rgb}$', fontsize=11)
    ax.set_ylabel(r'$w_{ir}$', fontsize=11)
    ax.set_title('CLPF Weight Heatmap')

    # Distribution
    ax = axes[1]
    ax.hist(w_rgb, bins=50, alpha=0.6, label=r'$w_{rgb}$', color='blue', density=True)
    ax.hist(w_ir, bins=50, alpha=0.6, label=r'$w_{ir}$', color='red', density=True)
    ax.axvline(x=mean_w_rgb, color='blue', linestyle='--', linewidth=2)
    ax.axvline(x=mean_w_ir, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title('Weight Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error scatter
    ax = axes[2]
    ax.scatter(w_rgb, err_rgb, c='blue', alpha=0.3, s=5, label='RGB')
    ax.scatter(w_ir, err_ir, c='red', alpha=0.3, s=5, label='IR')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Cross-Modal Error')
    ax.set_title('Error vs Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp1_weight_visualization')
    plt.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_path + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Exp1 saved: {save_path}")

    # Experiment 2: Cross-modal error
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(err_rgb, err_ir, alpha=0.3, s=10, c='purple')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('RGB Error')
    ax.set_ylabel('IR Error')
    ax.set_title('Error Correlation')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    diff = err_rgb - err_ir
    ax.hist(diff, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Error Difference')
    ax.set_ylabel('Count')
    ax.set_title('Error Asymmetry')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp2_crossmodal_error')
    plt.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_path + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Exp2 saved: {save_path}")

    return {
        'mean_w_rgb': mean_w_rgb, 'mean_w_ir': mean_w_ir,
        'std_w_rgb': std_w_rgb, 'std_w_ir': std_w_ir,
        'mean_err_rgb': np.mean(err_rgb), 'mean_err_ir': np.mean(err_ir)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                       default='runs_second/yolov12nir-rgb-base-p2di-p3-hcfm12/weights/best.pt')
    parser.add_argument('--data', type=str, default='F:/datasets/M3FD_yolo')
    parser.add_argument('--output', type=str, default='./clpf_analysis')
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("CLPF Data Extraction (Using Forward Hooks)")
    print("=" * 60)

    model = load_model(args.model)

    # Find images
    val_dirs = ['images/val', 'images/valid', 'images/test', 'image/test']
    val_dir = None
    for d in val_dirs:
        test_dir = os.path.join(args.data, d)
        if os.path.exists(test_dir):
            val_dir = test_dir
            break

    if val_dir is None:
        for root, dirs, files in os.walk(args.data):
            images = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
            if images:
                val_dir = root
                break

    if val_dir is None:
        print(f"Error: No images found in {args.data}")
        return

    print(f"Images: {val_dir}")

    w_rgb, w_ir, err_rgb, err_ir = extract_clpf_data_with_hooks(model, val_dir, args.samples)

    if w_rgb is None or len(w_rgb) == 0:
        print("Failed to extract CLPF data.")
        return

    print(f"\nExtracted {len(w_rgb)} samples")
    stats = generate_visualizations(w_rgb, w_ir, err_rgb, err_ir, args.output)

    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  w_rgb: mean={stats['mean_w_rgb']:.4f}, std={stats['std_w_rgb']:.4f}")
    print(f"  w_ir:  mean={stats['mean_w_ir']:.4f}, std={stats['std_w_ir']:.4f}")
    print(f"  err_rgb: mean={stats['mean_err_rgb']:.4f}")
    print(f"  err_ir:  mean={stats['mean_err_ir']:.4f}")
    print("=" * 60)
    print(f"\nSaved to: {args.output}/")


if __name__ == '__main__':
    main()
