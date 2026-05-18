#!/usr/bin/env python3
"""Extract CLPF data and generate enhanced visualizations"""

import os
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from ultralytics import YOLO

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False


def extract_clpf_data_detailed(model_path, image_dir, num_samples=200):
    """Extract CLPF weights with per-module data"""
    all_data = {15: {'w_rgb': [], 'w_ir': [], 'err_rgb': [], 'err_ir': []},
                22: {'w_rgb': [], 'w_ir': [], 'err_rgb': [], 'err_ir': []},
                29: {'w_rgb': [], 'w_ir': [], 'err_rgb': [], 'err_ir': []}}

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    pt_model = model.model.model

    clpf_indices = {}
    for i, module in enumerate(pt_model):
        if type(module).__name__ == 'CLPFRes':
            clpf_indices[i] = module
    print(f"Found CLPF modules at: {list(clpf_indices.keys())}")

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    image_paths = sorted(set(image_paths))[:num_samples]

    if not image_paths:
        print(f"No images found in {image_dir}")
        return None

    print(f"Processing {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(image_paths)}")

        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_ir = img_rgb.copy()
            img_6ch = np.concatenate([img_rgb, img_ir], axis=2)

            try:
                _ = model.predict(source=img_6ch, imgsz=512, verbose=False, device='cpu')
            except Exception:
                pass

            for idx, module in clpf_indices.items():
                if hasattr(module, 'recorded_w_rgb') and module.recorded_w_rgb is not None:
                    all_data[idx]['w_rgb'].append(module.recorded_w_rgb.mean().item())
                    all_data[idx]['w_ir'].append(module.recorded_w_ir.mean().item())
                    err_rgb_val = module.recorded_err_rgb.mean().item() if module.recorded_err_rgb is not None else 0
                    err_ir_val = module.recorded_err_ir.mean().item() if module.recorded_err_ir is not None else 0
                    all_data[idx]['err_rgb'].append(err_rgb_val)
                    all_data[idx]['err_ir'].append(err_ir_val)

        except Exception as e:
            continue

    return all_data


def generate_enhanced_visualizations(all_data, output_dir):
    """Generate enhanced visualization with scatter plots"""
    os.makedirs(output_dir, exist_ok=True)

    # Define colors for each CLPF module
    colors = {15: '#e74c3c', 22: '#3498db', 29: '#2ecc71'}  # Red, Blue, Green
    labels = {15: 'P3 (CLPF[15])', 22: 'P4 (CLPF[22])', 29: 'P5 (CLPF[29])'}
    markers = {15: 'o', 22: 's', 29: '^'}

    # ===== Figure 1: Enhanced Weight Scatter Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ----- Subplot 1: 2D Scatter with density contours -----
    ax = axes[0]

    # Plot each module separately with different colors
    for idx in [15, 22, 29]:
        if all_data[idx]['w_rgb']:
            ax.scatter(all_data[idx]['w_rgb'], all_data[idx]['w_ir'],
                      c=colors[idx], marker=markers[idx], s=60, alpha=0.7,
                      label=labels[idx], edgecolors='white', linewidths=0.5)

    # Add reference lines
    x_range = np.linspace(0.35, 0.65, 100)
    ax.plot(x_range, 1 - x_range, 'k--', linewidth=2, alpha=0.5, label=r'$w_{ir}=1-w_{rgb}$')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    # Add 45-degree line
    ax.plot([0.35, 0.65], [0.35, 0.65], 'k-', linewidth=1, alpha=0.3, label=r'$w_{rgb}=w_{ir}$')

    # Add annotation for balance point
    ax.annotate('Balanced\nFusion', xy=(0.5, 0.5), xytext=(0.62, 0.42),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel(r'$w_{rgb}$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$w_{ir}$', fontsize=12, fontweight='bold')
    ax.set_title('CLPF Weight 2D Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0.35, 0.65)
    ax.set_ylim(0.35, 0.65)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ----- Subplot 2: Weight Distribution Histogram -----
    ax = axes[1]

    bins = np.linspace(0.35, 0.65, 25)
    for idx in [15, 22, 29]:
        if all_data[idx]['w_rgb']:
            ax.hist(all_data[idx]['w_rgb'], bins=bins, alpha=0.5,
                   color=colors[idx], label=f'{labels[idx]}', edgecolor='black', linewidth=0.5)

    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Ideal (0.5)')
    ax.set_xlabel(r'$w_{rgb}$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Weight Distribution by Layer', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ----- Subplot 3: Box plot for comparison -----
    ax = axes[2]

    box_data = [all_data[15]['w_rgb'], all_data[22]['w_rgb'], all_data[29]['w_rgb']]
    bp = ax.boxplot(box_data, labels=['P3', 'P4', 'P5'], patch_artist=True, notch=True)

    for patch, color in zip(bp['boxes'], [colors[15], colors[22], colors[29]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Balanced (0.5)')
    ax.set_ylabel(r'$w_{rgb}$', fontsize=12, fontweight='bold')
    ax.set_title('Weight Distribution Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.35, 0.65)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp1_weight_enhanced')
    plt.savefig(save_path + '.png', dpi=200, bbox_inches='tight')
    plt.savefig(save_path + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Enhanced Exp1 saved: {save_path}")

    # ===== Figure 2: Error Analysis =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ----- Subplot 1: Error vs Weight Scatter -----
    ax = axes[0]
    for idx in [15, 22, 29]:
        if all_data[idx]['w_rgb']:
            ax.scatter(all_data[idx]['w_rgb'], all_data[idx]['err_rgb'],
                      c=colors[idx], marker=markers[idx], s=50, alpha=0.6,
                      label=f'{labels[idx]} RGB', edgecolors='white', linewidths=0.3)

    ax.set_xlabel(r'$w_{rgb}$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Cross-Modal Error $e_{rgb}$', fontsize=12, fontweight='bold')
    ax.set_title('RGB Error vs Weight', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ----- Subplot 2: Error distribution by module -----
    ax = axes[1]

    positions_rgb = [1, 2, 3]
    positions_ir = [1.3, 2.3, 3.3]

    bp1 = ax.boxplot([all_data[15]['err_rgb'], all_data[22]['err_rgb'], all_data[29]['err_rgb']],
                     positions=positions_rgb, widths=0.25, patch_artist=True)
    bp2 = ax.boxplot([all_data[15]['err_ir'], all_data[22]['err_ir'], all_data[29]['err_ir']],
                     positions=positions_ir, widths=0.25, patch_artist=True)

    for patch in bp1['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    for patch in bp2['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.6)

    ax.set_xticks([1.15, 2.15, 3.15])
    ax.set_xticklabels(['P3', 'P4', 'P5'])
    ax.set_ylabel('Cross-Modal Error', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution by Layer', fontsize=13, fontweight='bold')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['RGB Error', 'IR Error'], loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp2_error_analysis')
    plt.savefig(save_path + '.png', dpi=200, bbox_inches='tight')
    plt.savefig(save_path + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Enhanced Exp2 saved: {save_path}")

    # ===== Print statistics =====
    print("\n" + "=" * 60)
    print("CLPF Statistics by Layer:")
    print("=" * 60)
    for idx in [15, 22, 29]:
        if all_data[idx]['w_rgb']:
            w_rgb = np.array(all_data[idx]['w_rgb'])
            w_ir = np.array(all_data[idx]['w_ir'])
            err_rgb = np.array(all_data[idx]['err_rgb'])
            err_ir = np.array(all_data[idx]['err_ir'])

            layer_name = f"P3" if idx == 15 else f"P4" if idx == 22 else "P5"
            print(f"\n{layer_name} (CLPF[{idx}]):")
            print(f"  w_rgb: mean={w_rgb.mean():.4f}, std={w_rgb.std():.4f}, range=[{w_rgb.min():.3f}, {w_rgb.max():.3f}]")
            print(f"  w_ir:  mean={w_ir.mean():.4f}, std={w_ir.std():.4f}")
            print(f"  err_rgb: mean={err_rgb.mean():.4f}, std={err_rgb.std():.4f}")
            print(f"  err_ir:  mean={err_ir.mean():.4f}, std={err_ir.std():.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                       default='runs_second/yolov12nir-rgb-base-p2di-p3-hcfm12/weights/best.pt')
    parser.add_argument('--data', type=str, default='F:/datasets/M3FD_yolo')
    parser.add_argument('--output', type=str, default='./clpf_analysis')
    parser.add_argument('--samples', type=int, default=200)
    args = parser.parse_args()

    print("=" * 60)
    print("CLPF Enhanced Visualization")
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
        print(f"Error: No images found in {args.data}")
        return

    print(f"Model: {args.model}")
    print(f"Images from: {val_dir}")

    # Extract data
    all_data = extract_clpf_data_detailed(args.model, val_dir, args.samples)

    if all_data is None:
        return

    # Generate visualizations
    generate_enhanced_visualizations(all_data, args.output)

    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {args.output}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
