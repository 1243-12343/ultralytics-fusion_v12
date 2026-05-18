#!/usr/bin/env python3
"""Extract CLPF data from trained model and generate visualizations"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font for Chinese characters
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False

def load_model_weights(checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get the PyTorch model
    if 'model' in ckpt:
        model_wrapper = ckpt['model']
        if hasattr(model_wrapper, 'model'):
            model = model_wrapper.model
        else:
            model = model_wrapper
    elif 'ema' in ckpt and ckpt['ema'] is not None:
        model = ckpt['ema']
    else:
        raise ValueError("Cannot find model in checkpoint")

    model.eval()
    return model


def find_clpf_modules(model):
    """Find all CLPF/CLPFRes modules"""
    clpf_modules = []
    for i, module in enumerate(model):
        if type(module).__name__ in ['CLPF', 'CLPFRes']:
            clpf_modules.append((i, module))
    return clpf_modules


def extract_clpf_data(model, image_dir, num_samples=100):
    """Extract CLPF weights by running inference on images"""
    import torch.nn.functional as F

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

    # Find CLPF modules
    clpf_modules = find_clpf_modules(model)
    print(f"Found {len(clpf_modules)} CLPF modules")

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 20 == 0:
                print(f"  Processing: {i+1}/{len(image_paths)}")

            try:
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_ir = img_rgb.copy()  # Use RGB as IR placeholder

                img_size = 512
                img_rgb = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
                img_ir = cv2.resize(img_ir, (img_size, img_size)).astype(np.float32) / 255.0

                # Create 6-channel input [RGB + IR]
                img_6ch = np.concatenate([img_rgb, img_ir], axis=2)  # H, W, 6
                img_6ch = torch.from_numpy(img_6ch).permute(2, 0, 1).unsqueeze(0)  # 1, 6, H, W

                # Forward through the model layer by layer
                x = img_6ch
                for idx, layer in enumerate(model):
                    # Special handling for Multiin layers
                    if type(layer).__name__ == 'Multiin':
                        # Layer 1: extract RGB (first 3 channels)
                        # Layer 2: extract IR (last 3 channels)
                        if layer.out == 1:
                            x = layer(x)  # RGB branch
                        else:
                            # IR branch needs 6-channel input to extract last 3 channels
                            x = layer(x)  # IR branch
                    else:
                        x = layer(x)

                    # Check CLPF modules for recorded data
                    if type(layer).__name__ == 'CLPFRes' and hasattr(layer, 'recorded_w_rgb'):
                        if layer.recorded_w_rgb is not None:
                            w_rgb = layer.recorded_w_rgb.cpu().numpy().flatten()
                            w_ir = layer.recorded_w_ir.cpu().numpy().flatten()
                            err_rgb = layer.recorded_err_rgb.cpu().numpy().flatten()
                            err_ir = layer.recorded_err_ir.cpu().numpy().flatten()

                            all_w_rgb.extend(w_rgb)
                            all_w_ir.extend(w_ir)
                            all_err_rgb.extend(err_rgb)
                            all_err_ir.extend(err_ir)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

    if not all_w_rgb:
        print("No CLPF data extracted!")
        return None, None, None, None

    return (np.array(all_w_rgb), np.array(all_w_ir),
            np.array(all_err_rgb), np.array(all_err_ir))


def generate_visualizations(w_rgb, w_ir, err_rgb, err_ir, output_dir):
    """Generate visualization images"""
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics
    mean_w_rgb = np.mean(w_rgb)
    mean_w_ir = np.mean(w_ir)
    std_w_rgb = np.std(w_rgb)
    std_w_ir = np.std(w_ir)

    # Experiment 1: Weight Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Heatmap - zoomed to effective region [0.3, 0.7]
    ax = axes[0]
    x_min, x_max = 0.3, 0.7
    y_min, y_max = 0.3, 0.7
    mask = (w_rgb >= x_min) & (w_rgb <= x_max) & (w_ir >= y_min) & (w_ir <= y_max)
    heatmap, xedges, yedges = np.histogram2d(w_rgb[mask], w_ir[mask], bins=25)
    im = ax.imshow(heatmap.T, origin='lower', aspect='auto',
                   extent=[x_min, x_max, y_min, y_max], cmap='YlOrRd')
    x_line = np.linspace(x_min, x_max, 100)
    ax.plot(x_line, 1 - x_line, 'b--', linewidth=2, label=r'$w_{ir} = 1 - w_{rgb}$')
    ax.axhline(y=0.5, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0.5, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label=r'$w_{rgb}=w_{ir}=0.5$')
    ax.set_xlabel(r'$w_{rgb}$', fontsize=11)
    ax.set_ylabel(r'$w_{ir}$', fontsize=11)
    ax.set_title('CLPF Weight Heatmap (P3/P4/P5)', fontsize=12)
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, label='Sample Count')

    # Distribution comparison
    ax = axes[1]
    ax.hist(w_rgb, bins=50, alpha=0.6, label=r'$w_{rgb}$', color='blue', density=True)
    ax.hist(w_ir, bins=50, alpha=0.6, label=r'$w_{ir}$', color='red', density=True)
    ax.axvline(x=mean_w_rgb, color='blue', linestyle='--', linewidth=2, label=f'mean={mean_w_rgb:.3f}')
    ax.axvline(x=mean_w_ir, color='red', linestyle='--', linewidth=2, label=f'mean={mean_w_ir:.3f}')
    ax.set_xlabel('Weight Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('CLPF Weight Distribution', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Error vs weight scatter
    ax = axes[2]
    scatter = ax.scatter(w_rgb, err_rgb, c='blue', alpha=0.3, s=5, label='RGB Error')
    ax.scatter(w_ir, err_ir, c='red', alpha=0.3, s=5, label='IR Error')
    ax.set_xlabel('Weight', fontsize=11)
    ax.set_ylabel('Cross-Modal Error', fontsize=11)
    ax.set_title('Error vs Weight', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp1_weight_visualization')
    plt.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_path + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Exp1 saved: {save_path}")

    # Experiment 2: Cross-modal error analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(err_rgb, err_ir, alpha=0.3, s=10, c='purple')
    ax.plot([0, max(err_rgb.max(), err_ir.max())], [0, max(err_rgb.max(), err_ir.max())],
            'k--', linewidth=1, label='err_rgb = err_ir')
    ax.set_xlabel('RGB Cross-Modal Error', fontsize=11)
    ax.set_ylabel('IR Cross-Modal Error', fontsize=11)
    ax.set_title('Cross-Modal Error Correlation', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    diff = err_rgb - err_ir
    ax.hist(diff, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=np.mean(diff), color='blue', linestyle='--', linewidth=2, label=f'mean={np.mean(diff):.3f}')
    ax.set_xlabel('Error Difference (RGB - IR)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Error Asymmetry Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp2_crossmodal_error')
    plt.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_path + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Exp2 saved: {save_path}")

    return {
        'mean_w_rgb': mean_w_rgb,
        'mean_w_ir': mean_w_ir,
        'std_w_rgb': std_w_rgb,
        'std_w_ir': std_w_ir,
        'mean_err_rgb': np.mean(err_rgb),
        'mean_err_ir': np.mean(err_ir)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract CLPF data from model')
    parser.add_argument('--model', type=str,
                       default='runs_second/yolov12nir-rgb-base-p2di-p3-hcfm12/weights/best.pt',
                       help='Model checkpoint path')
    parser.add_argument('--data', type=str, default='F:/datasets/M3FD_yolo',
                       help='Dataset directory')
    parser.add_argument('--output', type=str, default='./clpf_analysis',
                       help='Output directory')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to process')
    args = parser.parse_args()

    print("=" * 60)
    print("CLPF Data Extraction and Visualization")
    print("=" * 60)

    # Load model
    model = load_model_weights(args.model)

    # Find validation images
    val_dirs = ['images/val', 'images/valid', 'images/test', 'val/images', 'valid/images', 'test/images', 'image/test']
    val_dir = None
    for d in val_dirs:
        test_dir = os.path.join(args.data, d)
        if os.path.exists(test_dir):
            val_dir = test_dir
            break

    if val_dir is None:
        for root, dirs, files in os.walk(args.data):
            images = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            if images:
                val_dir = root
                break

    if val_dir is None:
        print(f"Error: Cannot find validation images in {args.data}")
        return

    print(f"Using images from: {val_dir}")

    # Extract CLPF data
    w_rgb, w_ir, err_rgb, err_ir = extract_clpf_data(model, val_dir, args.samples)

    if w_rgb is None or len(w_rgb) == 0:
        print("Failed to extract CLPF data. Please check the model architecture.")
        return

    print(f"\nExtracted {len(w_rgb)} CLPF weight samples")

    # Generate visualizations
    stats = generate_visualizations(w_rgb, w_ir, err_rgb, err_ir, args.output)

    # Print statistics
    print("\n" + "=" * 60)
    print("CLPF Statistics (Extracted from Model):")
    print(f"  w_rgb: mean={stats['mean_w_rgb']:.4f}, std={stats['std_w_rgb']:.4f}")
    print(f"  w_ir:  mean={stats['mean_w_ir']:.4f}, std={stats['std_w_ir']:.4f}")
    print(f"  err_rgb: mean={stats['mean_err_rgb']:.4f}")
    print(f"  err_ir:  mean={stats['mean_err_ir']:.4f}")
    print("=" * 60)
    print(f"\nVisualizations saved to: {args.output}/")
    print("Formats: .png, .svg, .pdf")


if __name__ == '__main__':
    main()
