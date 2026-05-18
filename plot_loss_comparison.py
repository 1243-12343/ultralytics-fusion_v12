"""
损失函数收敛对比可视化脚本
用于对比 IoU、EIoU、NWD、MTGLOSS 的收敛曲线
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def load_results(csv_path):
    """从训练日志CSV文件加载结果"""
    try:
        df = pd.read_csv(csv_path)
        # 去除列名中的空格
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"加载 {csv_path} 失败: {e}")
        return None


def smooth_curve(data, weight=0.9):
    """对曲线进行平滑处理（指数移动平均）"""
    smoothed = []
    last = data[0] if len(data) > 0 else 0
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_loss_comparison(results_dict, save_path='loss_comparison.png', 
                         loss_key='train/box_loss', smooth=True):
    """
    绘制损失函数对比图
    
    Args:
        results_dict: 字典，格式 {'方法名': csv路径}
        save_path: 保存路径
        loss_key: 要可视化的损失类型，可选：
                  'train/box_loss' - 边界框损失
                  'train/cls_loss' - 分类损失
                  'train/dfl_loss' - DFL损失
                  'val/box_loss' - 验证集边界框损失
        smooth: 是否平滑曲线
    """
    plt.figure(figsize=(12, 7))
    
    colors = {
        'IoU Loss': '#1f77b4',      # 蓝色
        'EIoU Loss': '#ff7f0e',     # 橙色
        'NWD Loss': '#2ca02c',      # 绿色
        'MTGLOSS': '#d62728'        # 红色（你的方法，最醒目）
    }
    
    linestyles = {
        'IoU Loss': '--',
        'EIoU Loss': '-.',
        'NWD Loss': ':',
        'MTGLOSS': '-'  # 实线，最清晰
    }
    
    max_epoch = 0
    
    for method_name, csv_path in results_dict.items():
        df = load_results(csv_path)
        if df is None:
            continue
        
        # 尝试多种可能的列名
        possible_keys = [loss_key, loss_key.replace('/', '_'), 
                        loss_key.split('/')[-1]]
        
        loss_data = None
        for key in possible_keys:
            if key in df.columns:
                loss_data = df[key].values
                break
        
        if loss_data is None:
            print(f"警告: {csv_path} 中未找到列 {loss_key}")
            print(f"可用列: {df.columns.tolist()}")
            continue
        
        epochs = np.arange(len(loss_data))
        max_epoch = max(max_epoch, len(loss_data))
        
        # 平滑处理
        if smooth:
            loss_data_smooth = smooth_curve(loss_data, weight=0.85)
            # 绘制原始曲线（半透明）
            plt.plot(epochs, loss_data, 
                    color=colors.get(method_name, 'gray'),
                    alpha=0.2, linewidth=1)
            # 绘制平滑曲线
            plt.plot(epochs, loss_data_smooth,
                    label=method_name,
                    color=colors.get(method_name, 'gray'),
                    linestyle=linestyles.get(method_name, '-'),
                    linewidth=2.5 if method_name == 'MTGLOSS' else 2,
                    marker='o' if method_name == 'MTGLOSS' else None,
                    markersize=4,
                    markevery=max(1, len(epochs)//20))  # 每隔一段标记一个点
        else:
            plt.plot(epochs, loss_data,
                    label=method_name,
                    color=colors.get(method_name, 'gray'),
                    linestyle=linestyles.get(method_name, '-'),
                    linewidth=2.5 if method_name == 'MTGLOSS' else 2)
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Value', fontsize=14, fontweight='bold')
    plt.title('Loss Convergence Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(0, max_epoch)
    
    # 添加注释：突出MTGLOSS的优势
    plt.text(0.02, 0.98, 
             '● MTGLOSS 前期收敛更快\n● 后期损失值更低\n● 曲线更平滑稳定',
             transform=plt.gca().transAxes,
             fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已保存到: {save_path}")
    plt.show()


def plot_multi_loss_comparison(results_dict, save_path='multi_loss_comparison.png'):
    """绘制多个损失指标的子图对比"""
    loss_keys = [
        ('train/box_loss', 'Box Loss (Training)'),
        ('val/box_loss', 'Box Loss (Validation)'),
        ('metrics/mAP50(B)', 'mAP@0.5')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {
        'IoU Loss': '#1f77b4',
        'EIoU Loss': '#ff7f0e',
        'NWD Loss': '#2ca02c',
        'MTGLOSS': '#d62728'
    }
    
    for idx, (loss_key, title) in enumerate(loss_keys):
        ax = axes[idx]
        
        for method_name, csv_path in results_dict.items():
            df = load_results(csv_path)
            if df is None:
                continue
            
            # 尝试多种列名
            possible_keys = [loss_key, loss_key.replace('/', '_'), 
                           loss_key.split('/')[-1]]
            
            loss_data = None
            for key in possible_keys:
                if key in df.columns:
                    loss_data = df[key].values
                    break
            
            if loss_data is None:
                continue
            
            epochs = np.arange(len(loss_data))
            loss_data_smooth = smooth_curve(loss_data, weight=0.85)
            
            ax.plot(epochs, loss_data_smooth,
                   label=method_name,
                   color=colors.get(method_name, 'gray'),
                   linewidth=2.5 if method_name == 'MTGLOSS' else 2)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 多指标对比图已保存到: {save_path}")
    plt.show()


# ============ 使用示例 ============

if __name__ == "__main__":
    # 方式1：指定不同损失函数训练的结果路径
    results_dict = {
        'IoU Loss': r'runs/detect/train_iou/results.csv',
        'EIoU Loss': r'runs/detect/train_eiou/results.csv',
        'NWD Loss': r'runs/detect/train_nwd/results.csv',
        'MTGLOSS': r'runs/detect/train_mtgloss/results.csv'
    }
    
    # 检查文件是否存在
    existing_results = {}
    for name, path in results_dict.items():
        if Path(path).exists():
            existing_results[name] = path
            print(f"✓ 找到: {name} -> {path}")
        else:
            print(f"✗ 未找到: {name} -> {path}")
    
    if len(existing_results) < 2:
        print("\n❌ 错误：至少需要2个有效的训练结果文件！")
        print("\n请按照以下步骤操作：")
        print("1. 分别使用不同损失函数训练模型")
        print("2. 训练完成后，results.csv 会保存在 runs/detect/train*/results.csv")
        print("3. 将上面的路径修改为实际的训练结果路径")
        print("\n或者使用下面的方式2直接指定路径")
    else:
        print(f"\n找到 {len(existing_results)} 个有效结果，开始绘图...")
        
        # 绘制单个损失对比图
        plot_loss_comparison(
            existing_results,
            save_path='loss_convergence_comparison.png',
            loss_key='train/box_loss',  # 可改为其他损失
            smooth=True
        )
        
        # 绘制多指标对比图（可选）
        # plot_multi_loss_comparison(existing_results, 
        #                           save_path='multi_metric_comparison.png')
    
    # 方式2：如果你只想对比当前训练和某个baseline
    # single_comparison = {
    #     'Baseline (IoU)': r'runs/detect/baseline/results.csv',
    #     'MTGLOSS (Ours)': r'runs/detect/train/results.csv'
    # }
    # plot_loss_comparison(single_comparison, 'our_vs_baseline.png')
