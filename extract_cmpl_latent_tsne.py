"""
CMPL Latent Space Visualization with t-SNE
展示 RGB-IR 潜在空间的分布对比，直观说明结构一致性约束的效果
"""
#python extract_cmpl_latent_tsne.py --model runs_second/yolov12nir-rgb-base-p2di-p3-hcfm12/weights/best.pt --data F:/datasets/mydata.yaml --output ./clpf_analysis --tag '_trained'

import argparse
import os
import glob
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ultralytics.nn.tasks import DetectionModel


def load_data_from_yaml(data_yaml):
    """从 yaml 解析数据路径"""
    import yaml
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    base_path = data.get('path', '')
    train_path = os.path.join(base_path, data.get('train', ''))

    # IR 路径 - yaml 中写的是 image/train（小写）
    train_infrared = data.get('train_infrared', '')
    train_infrared_path = os.path.join(base_path, train_infrared) if train_infrared else train_path

    # 获取图片文件
    rgb_files = []
    ir_files = []

    for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
        rgb_files.extend(glob.glob(os.path.join(train_path, ext)))
        rgb_files.extend(glob.glob(os.path.join(train_path, ext.upper())))
        ir_files.extend(glob.glob(os.path.join(train_infrared_path, ext)))
        ir_files.extend(glob.glob(os.path.join(train_infrared_path, ext.upper())))

    # 去重并排序
    rgb_files = sorted(list(set(rgb_files)))
    ir_files = sorted(list(set(ir_files)))

    print(f"RGB path: {train_path}, files: {len(rgb_files)}")
    print(f"IR path: {train_infrared_path}, files: {len(ir_files)}")

    return rgb_files, ir_files


def extract_latent_features(model, data_yaml, num_samples=2000, device='cuda', img_size=512):
    """
    从 CLPF 模块提取 RGB-IR 潜在空间特征
    """
    model.eval().to(device)

    # 收集所有 CLPF/CLPFRes 模块的索引
    clpf_modules = []
    for i, m in model.named_modules():
        module_name = type(m).__name__
        if module_name == 'CLPF' or module_name == 'CLPFRes':
            clpf_modules.append((i, m))

    if not clpf_modules:

        print("Warning: No CLPF modules found!")
        return None, None

    print(f"Found {len(clpf_modules)} CLPF modules")

    # 加载数据
    rgb_files, ir_files = load_data_from_yaml(data_yaml)
    print(f"Found {len(rgb_files)} RGB images, {len(ir_files)} IR images")

    # 配对 RGB 和 IR 图片
    rgb_ir_pairs = []
    for rgb_path in rgb_files[:min(num_samples, len(rgb_files))]:
        # 提取文件名
        basename = os.path.basename(rgb_path)
        name_without_ext = os.path.splitext(basename)[0]

        # 尝试找对应的 IR 图片
        ir_path = None
        for ir_file in ir_files:
            ir_basename = os.path.basename(ir_file)
            ir_name = os.path.splitext(ir_basename)[0]
            if name_without_ext == ir_name or name_without_ext in ir_name or ir_name in name_without_ext:
                ir_path = ir_file
                break

        if ir_path is None and ir_files:
            idx = len(rgb_ir_pairs) % len(ir_files)
            ir_path = ir_files[idx]

        if ir_path:
            rgb_ir_pairs.append((rgb_path, ir_path))

    if not rgb_ir_pairs:
        print("No paired RGB-IR images found!")
        return None, None

    print(f"Processing {len(rgb_ir_pairs)} RGB-IR pairs...")

    all_z_rgb = []
    all_z_ir = []

    for rgb_path, ir_path in tqdm(rgb_ir_pairs, desc="Extracting features"):
        sample_count_before = len(all_z_rgb)
        try:
            # 读取图片 - 模型期望 6 通道输入 [RGB+IR]
            img_bgr = cv2.imread(rgb_path)
            if img_bgr is None or img_bgr.shape[2] == 0:
                print(f"Failed to read RGB or invalid image: {rgb_path}")
                continue

            # 尝试读取 IR
            if ir_path and os.path.exists(ir_path):
                img_ir_bgr = cv2.imread(ir_path)
                if img_ir_bgr is None or img_ir_bgr.shape[2] == 0:
                    img_ir_bgr = img_bgr.copy()
            else:
                img_ir_bgr = img_bgr.copy()

            # BGR -> RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_ir = cv2.cvtColor(img_ir_bgr, cv2.COLOR_BGR2RGB)

            # 缩放到统一大小
            img_rgb = cv2.resize(img_rgb, (img_size, img_size))
            img_ir = cv2.resize(img_ir, (img_size, img_size))

            # 确保是 float 类型
            img_rgb = img_rgb.astype(np.float32)
            img_ir = img_ir.astype(np.float32)

            # 归一化并转为 tensor [3, H, W]
            img_rgb_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1) / 255.0
            img_ir_tensor = torch.from_numpy(img_ir).permute(2, 0, 1) / 255.0

            # 标准化 (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_rgb_tensor = (img_rgb_tensor - mean) / std
            img_ir_tensor = (img_ir_tensor - mean) / std

            # 合并为 6 通道 [6, H, W]，确保 FP32
            img_combined = torch.cat([img_rgb_tensor, img_ir_tensor], dim=0).unsqueeze(0).float().to(device)

            # 前向传播 - 使用模型的 predict 方法（已修复 CLPF 输入处理）
            with torch.no_grad():
                _ = model.predict(img_combined)

            # 从第一个 CLPF 模块提取特征
            feature_extracted = False
            for idx, module in clpf_modules:
                if hasattr(module, 'recorded_z_rgb') and module.recorded_z_rgb is not None:
                    z_rgb = module.recorded_z_rgb.cpu().numpy()
                    z_ir = module.recorded_z_ir.cpu().numpy()

                    # 全局平均池化
                    z_rgb_flat = z_rgb.mean(axis=(2, 3)).flatten()
                    z_ir_flat = z_ir.mean(axis=(2, 3)).flatten()

                    all_z_rgb.append(z_rgb_flat)
                    all_z_ir.append(z_ir_flat)

                    # 重置记录
                    module.recorded_z_rgb = None
                    module.recorded_z_ir = None
                    feature_extracted = True
                    break

            if not feature_extracted and len(all_z_rgb) == sample_count_before:
                print(f"Warning: No features extracted for {rgb_path}")

        except Exception as e:
            print(f"Error processing {rgb_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_z_rgb:
        print("No latent features extracted!")
        return None, None

    return np.array(all_z_rgb), np.array(all_z_ir)


def plot_tsne_comparison(z_rgb, z_ir, output_dir='./clpf_analysis', tag=''):
    """绘制 t-SNE 对比图"""
    # 合并 RGB 和 IR 特征
    z_combined = np.concatenate([z_rgb, z_ir], axis=0)

    # 标签：0 = RGB, 1 = IR
    labels = np.array([0] * len(z_rgb) + [1] * len(z_ir))

    # 标准化
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_combined)

    # t-SNE 降维
    print("Running t-SNE (this may take a while)...")
    perplexity = min(30, len(z_rgb) - 1)
    if perplexity < 2:
        perplexity = 2
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    z_tsne = tsne.fit_transform(z_scaled)

    # 分离 RGB 和 IR 的 t-SNE 结果
    z_rgb_tsne = z_tsne[:len(z_rgb)]
    z_ir_tsne = z_tsne[len(z_rgb):]

    # ========== 绘图 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    color_rgb = '#E74C3C'
    color_ir = '#3498DB'
    color_both = '#9B59B6'

    # 子图1：RGB vs IR 散点图
    ax1 = axes[0]
    ax1.scatter(z_rgb_tsne[:, 0], z_rgb_tsne[:, 1], c=color_rgb, alpha=0.5, s=15,
                label=f'RGB (n={len(z_rgb)})')
    ax1.scatter(z_ir_tsne[:, 0], z_ir_tsne[:, 1], c=color_ir, alpha=0.5, s=15,
                label=f'IR (n={len(z_ir)})')

    n_connections = min(100, len(z_rgb))
    for i in range(n_connections):
        ax1.plot([z_rgb_tsne[i, 0], z_ir_tsne[i, 0]],
                 [z_rgb_tsne[i, 1], z_ir_tsne[i, 1]],
                 c=color_both, alpha=0.1, linewidth=0.5)

    ax1.set_xlabel('t-SNE Dim 1', fontsize=11)
    ax1.set_ylabel('t-SNE Dim 2', fontsize=11)
    ax1.set_title('(a) RGB-IR Latent Space Distribution\n(with modality pairing lines)', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 子图2：模态间距离分布
    ax2 = axes[1]
    distances = np.linalg.norm(z_rgb - z_ir, axis=1)

    ax2.hist(distances, bins=50, color=color_both, alpha=0.7, edgecolor='white')
    ax2.axvline(distances.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {distances.mean():.4f}')
    ax2.axvline(np.median(distances), color='orange', linestyle='-.', linewidth=2,
                label=f'Median: {np.median(distances):.4f}')

    ax2.set_xlabel('L2 Distance (RGB ↔ IR)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('(b) Cross-Modal Distance Distribution\n(Lower = Better Alignment)', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 子图3：配对点重叠率
    ax3 = axes[2]

    from scipy.spatial.distance import cdist
    dist_matrix = cdist(z_rgb_tsne, z_ir_tsne, metric='euclidean')
    min_distances_rgb_to_ir = dist_matrix.min(axis=1)
    min_distances_ir_to_rgb = dist_matrix.min(axis=0)

    data_to_plot = [min_distances_rgb_to_ir, min_distances_ir_to_rgb]
    bp = ax3.boxplot(data_to_plot, labels=['RGB→IR', 'IR→RGB'], patch_artist=True)

    colors = [color_rgb, color_ir]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax3.set_ylabel('Nearest Neighbor Distance (t-SNE space)', fontsize=11)
    ax3.set_title('(c) Cross-Modal Nearest Neighbor Distance\n(Lower = Stronger Structure)', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'CMPL Latent Space Analysis (t-SNE){tag}', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # 保存
    output_path = os.path.join(output_dir, f'cmpl_latent_tsne{tag}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    pdf_path = output_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")

    plt.close(fig)

    # ========== 计算统计指标 ==========
    print("\n" + "="*60)
    print("CMPL Latent Space Statistics")
    print("="*60)

    print(f"\n[1] Cross-Modal L2 Distance:")
    print(f"    Mean:   {distances.mean():.6f}")
    print(f"    Std:    {distances.std():.6f}")
    print(f"    Median: {np.median(distances):.6f}")
    print(f"    Min:    {distances.min():.6f}")
    print(f"    Max:    {distances.max():.6f}")

    overlap_threshold = 5.0
    overlap_rgb = (min_distances_rgb_to_ir < overlap_threshold).mean()
    overlap_ir = (min_distances_ir_to_rgb < overlap_threshold).mean()
    print(f"\n[2] Nearest Neighbor Overlap Rate (threshold={overlap_threshold}):")
    print(f"    RGB→IR: {overlap_rgb:.2%}")
    print(f"    IR→RGB: {overlap_ir:.2%}")

    from scipy.stats import pearsonr
    corr_rgb_ir, p_rgb = pearsonr(z_rgb.flatten(), z_ir.flatten())
    print(f"\n[3] RGB-IR Correlation:")
    print(f"    Pearson r:  {corr_rgb_ir:.4f} (p={p_rgb:.2e})")

    return {
        'mean_distance': distances.mean(),
        'std_distance': distances.std(),
        'median_distance': np.median(distances),
        'overlap_rgb_to_ir': overlap_rgb,
        'overlap_ir_to_rgb': overlap_ir,
        'pearson_r': corr_rgb_ir,
    }


def main():
    import yaml

    parser = argparse.ArgumentParser(description='CMPL Latent Space t-SNE Visualization')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pt)')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--output', type=str, default='./clpf_analysis', help='Output directory')
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples to extract')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--img-size', type=int, default=512, help='Image size')
    parser.add_argument('--tag', type=str, default='', help='Tag for output files')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 加载模型
    print(f"Loading model: {args.model}")
    ckpt = torch.load(args.model, map_location='cpu')

    # YOLO checkpoint 结构
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model = ckpt['model']
    elif hasattr(ckpt, 'model'):
        model = ckpt.model
    else:
        model = ckpt

    # 提取特征
    # 确保模型有 predict 方法（使用 DetectionModel 的 predict 以支持 CLPF）
    if not hasattr(model, 'predict'):
        print("Model does not have predict method, wrapping with DetectionModel")
        from ultralytics.nn.tasks import DetectionModel
        model = DetectionModel(cfg=args.model.replace('.pt', '.yaml'), verbose=False)
        ckpt_full = torch.load(args.model, map_location='cpu')
        if 'model' in ckpt_full:
            state = ckpt_full['model'].state_dict()
        else:
            state = ckpt_full.state_dict()
        model.load_state_dict(state, strict=False)
    elif hasattr(model, 'module'):
        # 处理 DataParallel/EMA 包装
        model = model.module

    # 确保模型权重为 FP32（模型权重可能是 AMP 训练的 FP16）
    model = model.float()

    print(f"Model type: {type(model).__name__}")

    # 提取特征
    z_rgb, z_ir = extract_latent_features(
        model, args.data, args.samples, args.device, args.img_size
    )

    if z_rgb is None:
        print("Failed to extract latent features.")
        return

    print(f"\nExtracted {len(z_rgb)} RGB samples, {len(z_ir)} IR samples")
    print(f"Feature dimension: {z_rgb.shape[1]}")

    # 绘制 t-SNE 图
    stats = plot_tsne_comparison(z_rgb, z_ir, args.output, args.tag)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
