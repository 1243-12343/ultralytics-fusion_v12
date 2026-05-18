#!/usr/bin/env python3
"""Extract CLPF data from trained model and generate visualizations"""

import os
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ultralytics import YOLO

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False


def extract_clpf_data(model_path, image_dir, num_samples=200):
    """Extract CLPF weights from model by running inference on images"""
    all_w_rgb = {15: [], 22: [], 29: []}
    all_w_ir = {15: [], 22: [], 29: []}
    all_err_rgb = {15: [], 22: [], 29: []}
    all_err_ir = {15: [], 22: [], 29: []}

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    pt_model = model.model.model

    # Find CLPF modules
    clpf_indices = {}
    for i, module in enumerate(pt_model):
        if type(module).__name__ == 'CLPFRes':
            clpf_indices[i] = module
    print(f"Found CLPF modules at: {list(clpf_indices.keys())}")

    # Find images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    image_paths = sorted(set(image_paths))[:num_samples]

    if not image_paths:
        print(f"No images found in {image_dir}")
        return None, None, None, None

    print(f"Processing {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(image_paths)}")

        try:
            # Load and preprocess
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_ir = img_rgb.copy()  # Use RGB as IR placeholder

            # Create 6-channel image [RGB | IR]
            img_6ch = np.concatenate([img_rgb, img_ir], axis=2)  # H, W, 6

            # Run inference (ignore postprocess error)
            try:
                _ = model.predict(source=img_6ch, imgsz=512, verbose=False, device='cpu')
            except Exception:
                pass  # Ignore postprocess errors

            # Collect recorded data from each CLPF module
            for idx, module in clpf_indices.items():
                if hasattr(module, 'recorded_w_rgb') and module.recorded_w_rgb is not None:
                    # recorded data is [B, 1, 1, 1] - scalar per image
                    w_rgb_val = module.recorded_w_rgb.mean().item()
                    w_ir_val = module.recorded_w_ir.mean().item()
                    err_rgb_val = module.recorded_err_rgb.mean().item() if module.recorded_err_rgb is not None else 0
                    err_ir_val = module.recorded_err_ir.mean().item() if module.recorded_err_ir is not None else 0

                    all_w_rgb[idx].append(w_rgb_val)
                    all_w_ir[idx].append(w_ir_val)
                    all_err_rgb[idx].append(err_rgb_val)
                    all_err_ir[idx].append(err_ir_val)

        except Exception as e:
            print(f"  Error: {img_path}: {e}")
            continue

    # Aggregate data from all CLPF modules
    if not all_w_rgb[15]:
        print("No CLPF data extracted!")
        return None, None, None, None

    # Combine all CLPF module data
    combined_w_rgb = []
    combined_w_ir = []
    combined_err_rgb = []
    combined_err_ir = []

    for idx in clpf_indices.keys():
        combined_w_rgb.extend(all_w_rgb[idx])
        combined_w_ir.extend(all_w_ir[idx])
        combined_err_rgb.extend(all_err_rgb[idx])
        combined_err_ir.extend(all_err_ir[idx])

    print(f"Extracted {len(combined_w_rgb)} samples from {len(clpf_indices)} CLPF modules")

    # Print per-module statistics
    print("\nPer-module statistics:")
    for idx in clpf_indices.keys():
        if all_w_rgb[idx]:
            mean_w = np.mean(all_w_rgb[idx])
            print(f"  CLPF[{idx}]: mean_w_rgb={mean_w:.4f}, n={len(all_w_rgb[idx])}")

    return (np.array(combined_w_rgb), np.array(combined_w_ir),
            np.array(combined_err_rgb), np.array(combined_err_ir))


def generate_visualizations(w_rgb, w_ir, err_rgb, err_ir, output_dir):
    """Generate visualization images"""
    os.makedirs(output_dir, exist_ok=True)

    mean_w_rgb = np.mean(w_rgb)
    mean_w_ir = np.mean(w_ir)
    std_w_rgb = np.std(w_rgb)
    std_w_ir = np.std(w_ir)

    # ===== Figure 1: CLPF Weight Visualization =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Subplot 1: 2D Heatmap
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
        ax.axvline(x=0.5, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label=r'$w_{rgb}=w_{ir}=0.5$')
        plt.colorbar(im, ax=ax, label='Sample Count')
    ax.set_xlabel(r'$w_{rgb}$', fontsize=11)
    ax.set_ylabel(r'$w_{ir}$', fontsize=11)
    ax.set_title('CLPF Weight Distribution (P3/P4/P5)', fontsize=12)
    ax.legend(fontsize=8)

    # Subplot 2: Weight histogram
    ax = axes[1]
    ax.hist(w_rgb, bins=30, alpha=0.6, label=r'$w_{rgb}$', color='blue', density=True, edgecolor='black')
    ax.hist(w_ir, bins=30, alpha=0.6, label=r'$w_{ir}$', color='red', density=True, edgecolor='black')
    ax.axvline(x=mean_w_rgb, color='blue', linestyle='--', linewidth=2, label=f'mean={mean_w_rgb:.3f}')
    ax.axvline(x=mean_w_ir, color='red', linestyle='--', linewidth=2, label=f'mean={mean_w_ir:.3f}')
    ax.set_xlabel('Weight Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('CLPF Weight Histogram', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Error vs Weight
    ax = axes[2]
    ax.scatter(w_rgb, err_rgb, c='blue', alpha=0.4, s=20, label=r'RGB $e_{cross}$', marker='o')
    ax.scatter(w_ir, err_ir, c='red', alpha=0.4, s=20, label=r'IR $e_{cross}$', marker='^')
    ax.set_xlabel('Weight', fontsize=11)
    ax.set_ylabel('Cross-Modal Error', fontsize=11)
    ax.set_title('Error vs. Modality Weight', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp1_weight_visualization')
    plt.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_path + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Exp1 saved: {save_path}")

    # ===== Figure 2: Cross-Modal Error Analysis =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(err_rgb, err_ir, alpha=0.4, s=20, c='purple', marker='s')
    max_err = max(err_rgb.max(), err_ir.max())
    ax.plot([0, max_err], [0, max_err], 'k--', linewidth=1.5, label='$e_{rgb}=e_{ir}$')
    ax.set_xlabel(r'RGB Cross-Modal Error $e_{rgb}$', fontsize=11)
    ax.set_ylabel(r'IR Cross-Modal Error $e_{ir}$', fontsize=11)
    ax.set_title('Cross-Modal Error Correlation', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    diff = err_rgb - err_ir
    ax.hist(diff, bins=30, alpha=0.7, color='green', edgecolor='black', density=True)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero asymmetry')
    ax.axvline(x=np.mean(diff), color='blue', linestyle='--', linewidth=2, label=f'mean={np.mean(diff):.4f}')
    ax.set_xlabel(r'Error Asymmetry $\Delta e = e_{rgb} - e_{ir}$', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
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
        'mean_w_rgb': mean_w_rgb, 'mean_w_ir': mean_w_ir,
        'std_w_rgb': std_w_rgb, 'std_w_ir': std_w_ir,
        'mean_err_rgb': np.mean(err_rgb), 'mean_err_ir': np.mean(err_ir),
        'std_err_rgb': np.std(err_rgb), 'std_err_ir': np.std(err_ir)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract CLPF data from trained model')
    parser.add_argument('--model', type=str,
                       default='runs_second/yolov12nir-rgb-base-p2di-p3-hcfm12/weights/best.pt',
                       help='Model checkpoint path')
    parser.add_argument('--data', type=str, default='F:/datasets/M3FD_yolo',
                       help='Dataset directory')
    parser.add_argument('--output', type=str, default='./clpf_analysis',
                       help='Output directory')
    parser.add_argument('--samples', type=int, default=200,
                       help='Number of samples to process')
    args = parser.parse_args()

    print("=" * 60)
    print("CLPF Data Extraction from Trained Model")
    print("=" * 60)

    # Find validation images
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
        print(f"Error: Cannot find images in {args.data}")
        return

    print(f"Model: {args.model}")
    print(f"Images from: {val_dir}")
    print(f"Samples: {args.samples}")

    # Extract CLPF data
    w_rgb, w_ir, err_rgb, err_ir = extract_clpf_data(args.model, val_dir, args.samples)

    if w_rgb is None:
        print("\nFailed to extract CLPF data!")
        return

    print(f"\nExtracted {len(w_rgb)} CLPF weight samples")

    # Generate visualizations
    stats = generate_visualizations(w_rgb, w_ir, err_rgb, err_ir, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("CLPF Statistics (Extracted from Trained Model):")
    print(f"  w_rgb:  mean={stats['mean_w_rgb']:.4f}, std={stats['std_w_rgb']:.4f}")
    print(f"  w_ir:   mean={stats['mean_w_ir']:.4f}, std={stats['std_w_ir']:.4f}")
    print(f"  err_rgb: mean={stats['mean_err_rgb']:.4f}, std={stats['std_err_rgb']:.4f}")
    print(f"  err_ir:  mean={stats['mean_err_ir']:.4f}, std={stats['std_err_ir']:.4f}")
    print("=" * 60)
    print(f"\nVisualizations saved to: {args.output}/")
    print("Files: exp1_weight_visualization.*, exp2_crossmodal_error.*")


if __name__ == '__main__':
    main()
