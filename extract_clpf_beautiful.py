#!/usr/bin/env python3
"""Enhanced CLPF visualization with beautiful styling"""

import os
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats
from ultralytics import YOLO

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 11


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


def generate_beautiful_visualizations(all_data, output_dir):
    """Generate beautiful publication-quality visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    # Beautiful color palette
    colors = {
        15: {'main': '#E74C3C', 'light': '#FADBD8'},  # Red for P3
        22: {'main': '#3498DB', 'light': '#D4E6F1'},  # Blue for P4
        29: {'main': '#27AE60', 'light': '#D5F5E3'}   # Green for P5
    }
    labels = {15: 'P3 (High-res)', 22: 'P4 (Mid-res)', 29: 'P5 (Low-res)'}
    layer_names = ['P3', 'P4', 'P5']
    indices = [15, 22, 29]

    # ===== Figure 1: Main Weight Analysis =====
    fig = plt.figure(figsize=(16, 5))

    # Subplot 1: 2D Scatter with annotations
    ax1 = fig.add_subplot(131)
    for idx in indices:
        if all_data[idx]['w_rgb']:
            ax1.scatter(all_data[idx]['w_rgb'], all_data[idx]['w_ir'],
                       c=colors[idx]['main'], marker='o', s=80, alpha=0.6,
                       label=labels[idx], edgecolors='white', linewidths=0.8)

    # Reference lines
    x_range = np.linspace(0.2, 0.85, 100)
    ax1.plot(x_range, 1 - x_range, 'k--', linewidth=2, alpha=0.4, label=r'$w_{ir}=1-w_{rgb}$')
    ax1.plot([0.2, 0.85], [0.2, 0.85], 'k:', linewidth=1, alpha=0.3, label=r'$w_{rgb}=w_{ir}$')
    ax1.axhline(y=0.5, color='gray', linestyle='-', linewidth=1, alpha=0.4)
    ax1.axvline(x=0.5, color='gray', linestyle='-', linewidth=1, alpha=0.4)

    # Add centered annotation
    ax1.annotate('Balanced\nFusion Zone', xy=(0.5, 0.5), xytext=(0.68, 0.38),
                fontsize=10, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FCF3CF', edgecolor='#F39C12', alpha=0.9))

    ax1.set_xlabel(r'RGB Weight $w_{rgb}$', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'IR Weight $w_{ir}$', fontsize=12, fontweight='bold')
    ax1.set_title('(a) CLPF Weight Distribution', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax1.set_xlim(0.3, 0.75)
    ax1.set_ylim(0.25, 0.70)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Subplot 2: Violin + Strip plots for weight distribution
    ax2 = fig.add_subplot(132)

    positions = [1, 2, 3]
    data_for_violin = [np.array(all_data[idx]['w_rgb']) for idx in indices]

    # Violin plot
    parts = ax2.violinplot(data_for_violin, positions=positions, widths=0.7,
                          showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[indices[i]]['main'])
        pc.set_alpha(0.7)
        pc.set_edgecolor('white')

    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(2)

    # Add strip plot (individual points) beside violin
    for i, idx in enumerate(indices):
        y = all_data[idx]['w_rgb']
        x = np.random.normal(positions[i], 0.06, size=len(y))
        ax2.scatter(x, y, c=colors[idx]['main'], s=15, alpha=0.3, zorder=3)

    # Reference line at 0.5
    ax2.axhline(y=0.5, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8, label='Balanced (0.5)')

    ax2.set_xticks(positions)
    ax2.set_xticklabels(layer_names, fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'RGB Weight $w_{rgb}$', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Weight Distribution', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim(0.30, 0.75)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Subplot 3: Mean ± Std bar chart
    ax3 = fig.add_subplot(133)

    means = [np.mean(all_data[idx]['w_rgb']) for idx in indices]
    stds = [np.std(all_data[idx]['w_rgb']) for idx in indices]

    x_pos = np.arange(len(layer_names))
    bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, color=[colors[i]['main'] for i in indices],
                   edgecolor='white', linewidth=2, alpha=0.85, error_kw={'linewidth': 2})

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Reference line
    ax3.axhline(y=0.5, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8, label='Balanced')

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(layer_names, fontsize=12, fontweight='bold')
    ax3.set_ylabel(r'Mean $w_{rgb} \pm \sigma$', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Mean Weight Comparison', fontsize=13, fontweight='bold', pad=10)
    ax3.set_ylim(0, 0.75)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig_clpf_weight_analysis')
    plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Figure 1 saved: {save_path}")

    # ===== Figure 2: Error Analysis =====
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Error vs Weight scatter
    ax = axes[0]
    for idx in indices:
        if all_data[idx]['w_rgb']:
            ax.scatter(all_data[idx]['w_rgb'], all_data[idx]['err_rgb'],
                      c=colors[idx]['main'], marker='o', s=60, alpha=0.5,
                      label=f'{labels[idx]}', edgecolors='white', linewidths=0.5)

    ax.set_xlabel(r'RGB Weight $w_{rgb}$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Cross-Modal Error $e_{cross}$', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Modal Error vs Weight', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Subplot 2: Error boxplot comparison (RGB vs IR)
    ax = axes[1]

    # Prepare grouped data
    err_data = []
    tick_labels = []
    colors_err = []
    for idx in indices:
        err_data.append(all_data[idx]['err_rgb'])
        err_data.append(all_data[idx]['err_ir'])
        tick_labels.extend([f'{labels[idx]}\nRGB', f'{labels[idx]}\nIR'])
        colors_err.extend([colors[idx]['main'], '#E74C3C'])

    positions_grouped = [1, 1.4, 3, 3.4, 5, 5.4]
    bp = ax.boxplot(err_data, positions=positions_grouped, widths=0.35, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors_err):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('white')
        patch.set_linewidth(2)

    # Custom legend
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='RGB Error'),
        Patch(facecolor='#E74C3C', alpha=0.7, label='IR Error')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_xticks([1.2, 3.2, 5.2])
    ax.set_xticklabels(['P3', 'P4', 'P5'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Modal Error', fontsize=12, fontweight='bold')
    ax.set_title('Error Comparison: RGB vs IR', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    save_path2 = os.path.join(output_dir, 'fig_clpf_error_analysis')
    plt.savefig(save_path2 + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path2 + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(save_path2 + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Figure 2 saved: {save_path2}")

    # ===== Print Statistics Table =====
    print("\n" + "=" * 80)
    print("CLPF Statistics Summary")
    print("=" * 80)
    print(f"{'Layer':<8} {'w_rgb Mean':<12} {'w_rgb Std':<12} {'w_rgb Range':<18} {'err_rgb Mean':<12} {'err_ir Mean':<12}")
    print("-" * 80)
    for idx in indices:
        if all_data[idx]['w_rgb']:
            w_rgb = np.array(all_data[idx]['w_rgb'])
            w_ir = np.array(all_data[idx]['w_ir'])
            err_rgb = np.array(all_data[idx]['err_rgb'])
            err_ir = np.array(all_data[idx]['err_ir'])

            layer = 'P3' if idx == 15 else 'P4' if idx == 22 else 'P5'
            print(f"{layer:<8} {w_rgb.mean():<12.4f} {w_rgb.std():<12.4f} "
                  f"[{w_rgb.min():.3f}, {w_rgb.max():.3f}]{'':<5} {err_rgb.mean():<12.4f} {err_ir.mean():<12.4f}")


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
    print("CLPF Beautiful Visualization")
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

    all_data = extract_clpf_data_detailed(args.model, val_dir, args.samples)

    if all_data is None:
        return

    generate_beautiful_visualizations(all_data, args.output)

    print("\n" + "=" * 60)
    print(f"Figures saved to: {args.output}/")
    print("  - fig_clpf_weight_analysis.*  (Main weight analysis)")
    print("  - fig_clpf_error_analysis.*  (Error analysis)")
    print("=" * 60)


if __name__ == '__main__':
    main()
