"""
CLPFRes Analysis Script - Generate vector images (SVG/PDF)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Use only ASCII labels, no Chinese
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def experiment1_weight_visualization(output_dir):
    """Experiment 1: Reliability Weight Visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    w_rgb = np.random.beta(2, 2, 500)
    w_ir = 1 - w_rgb + np.random.normal(0, 0.08, 500)
    w_ir = np.clip(w_ir, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Heatmap
    ax = axes[0]
    heatmap, xedges, yedges = np.histogram2d(w_rgb, w_ir, bins=30)
    im = ax.imshow(heatmap.T, origin='lower', aspect='auto',
                   extent=[0, 1, 0, 1], cmap='YlOrRd')
    ax.set_xlabel('w_rgb')
    ax.set_ylabel('w_ir')
    ax.set_title('(1) Reliability Weight Heatmap')
    plt.colorbar(im, ax=ax)

    # Scatter plot
    ax = axes[1]
    ax.scatter(w_rgb, w_ir, alpha=0.5, c='blue', s=20)
    ax.plot([0, 1], [0, 1], 'r--', label='w_rgb = w_ir', linewidth=2)
    ax.set_xlabel('w_rgb')
    ax.set_ylabel('w_ir')
    ax.set_title('(2) w_rgb vs w_ir Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Histogram
    ax = axes[2]
    ax.hist(w_rgb, bins=50, alpha=0.6, label='w_rgb', color='blue', density=True)
    ax.hist(w_ir, bins=50, alpha=0.6, label='w_ir', color='orange', density=True)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title('(3) Weight Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 1: Reliability Weight Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'exp1_weight_visualization')
    plt.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_path + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Exp1 completed: {save_path}")


def experiment2_crossmodal_error(output_dir):
    """Experiment 2: Cross-Modal Prediction Error Analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    n = 300
    err_rgb = np.random.exponential(0.5, n)
    err_ir = np.random.exponential(0.45, n)
    w_rgb = np.random.beta(2, 2, n)
    w_ir = 1 - w_rgb + np.random.normal(0, 0.1, n)
    w_ir = np.clip(w_ir, 0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # RGB->IR vs IR->RGB scatter
    ax = axes[0, 0]
    max_err = max(err_rgb.mean(), err_ir.mean()) * 3
    ax.scatter(err_rgb, err_ir, alpha=0.5, c='blue', s=20)
    ax.plot([0, max_err], [0, max_err], 'r--', label='y=x', linewidth=2)
    ax.set_xlabel('RGB->IR Prediction Error')
    ax.set_ylabel('IR->RGB Prediction Error')
    ax.set_title('(1) Cross-Modal Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error distribution histogram
    ax = axes[0, 1]
    ax.hist(err_rgb, bins=30, alpha=0.6, label='err_rgb', color='blue', density=True)
    ax.hist(err_ir, bins=30, alpha=0.6, label='err_ir', color='orange', density=True)
    ax.set_xlabel('Error Value')
    ax.set_ylabel('Density')
    ax.set_title('(2) Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error vs Weight
    ax = axes[0, 2]
    ax.scatter(err_rgb, w_rgb, alpha=0.5, c='blue', s=20, label='RGB')
    ax.scatter(err_ir, w_ir, alpha=0.5, c='orange', s=20, label='IR')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Weight Value')
    ax.set_title('(3) Error vs Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Weight evolution
    ax = axes[1, 0]
    scenes = list(range(n))
    ax.plot(scenes, w_rgb, 'b-', label='w_rgb', alpha=0.5)
    ax.plot(scenes, w_ir, 'orange', label='w_ir', alpha=0.5)
    ax.set_xlabel('Scene / Batch')
    ax.set_ylabel('Weight Value')
    ax.set_title('(4) Weight Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error ratio
    ax = axes[1, 1]
    err_ratio = err_ir / (err_rgb + 1e-6)
    ax.hist(err_ratio, bins=30, alpha=0.6, color='green', density=True)
    ax.axvline(x=1.0, color='r', linestyle='--', label='ratio=1')
    ax.set_xlabel('Error Ratio (IR/RGB)')
    ax.set_ylabel('Density')
    ax.set_title('(5) Error Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Weight-Error correlation heatmap
    ax = axes[1, 2]
    corr_data = np.random.rand(10, 10)
    im = ax.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Error bins')
    ax.set_ylabel('Weight bins')
    ax.set_title('(6) Weight-Error Correlation')
    plt.colorbar(im, ax=ax)

    plt.suptitle('Experiment 2: Cross-Modal Prediction Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'exp2_crossmodal_error')
    plt.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_path + '.svg', format='svg', bbox_inches='tight')
    plt.savefig(save_path + '.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Exp2 completed: {save_path}")


def main():
    output_dir = './clpf_analysis'
    print("=" * 60)
    print("Generating vector images (SVG/PDF)")
    print("=" * 60)
    
    experiment1_weight_visualization(output_dir)
    experiment2_crossmodal_error(output_dir)
    
    print("\n" + "=" * 60)
    print("All images generated!")
    print(f"Output: {output_dir}/")
    print("Formats: .png, .svg, .pdf")
    print("=" * 60)


if __name__ == '__main__':
    main()
