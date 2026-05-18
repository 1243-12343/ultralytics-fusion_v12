"""
用不同损失函数训练模型以收集对比数据
运行此脚本会自动训练4次，每次使用不同的损失函数
"""

from ultralytics import YOLO
import yaml
import shutil
from pathlib import Path


def train_with_loss(loss_type, project_name, epochs=100, data_yaml='your_data.yaml'):
    """
    使用指定损失函数训练模型
    
    Args:
        loss_type: 损失函数类型 ('iou', 'eiou', 'nwd', 'mtgloss')
        project_name: 项目名称（保存路径）
        epochs: 训练轮数
        data_yaml: 数据集配置文件
    """
    print(f"\n{'='*60}")
    print(f"开始训练: {loss_type.upper()}")
    print(f"{'='*60}\n")
    
    # 加载模型
    model = YOLO('ultralytics/cfg/models/v12/p2-glfa-nwdloss-siou.yaml')
    
    # 训练参数
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        project=f'runs/detect/{project_name}',
        name=f'train_{loss_type}',
        exist_ok=True,
        # 根据不同损失函数设置参数
        # 注意：这里需要根据你的实际代码修改损失函数配置方式
        # box='nwd' if loss_type == 'nwd' else 'eiou' if loss_type == 'eiou' else 'iou',
        # 如果你的MTGLOSS是默认的，可以不设置额外参数
    )
    
    print(f"\n✅ {loss_type.upper()} 训练完成！")
    print(f"结果保存在: runs/detect/{project_name}/train_{loss_type}/")
    
    return results


def quick_comparison_train(epochs=50):
    """
    快速对比训练（较少epoch，仅用于验证收敛趋势）
    """
    loss_types = ['iou', 'eiou', 'nwd', 'mtgloss']
    
    print("=" * 60)
    print("开始快速对比训练")
    print(f"每个损失函数训练 {epochs} 个 epoch")
    print("=" * 60)
    
    # 请根据你的实际数据集路径修改
    data_yaml = 'path/to/your/data.yaml'
    
    # 检查数据集路径
    if not Path(data_yaml).exists():
        print(f"\n❌ 错误: 数据集配置文件不存在: {data_yaml}")
        print("请修改 data_yaml 为你的实际数据集路径")
        return
    
    results = {}
    for loss_type in loss_types:
        try:
            results[loss_type] = train_with_loss(
                loss_type=loss_type,
                project_name='loss_comparison',
                epochs=epochs,
                data_yaml=data_yaml
            )
        except Exception as e:
            print(f"\n❌ {loss_type} 训练失败: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("所有训练完成！")
    print("=" * 60)
    print("\n训练结果保存在:")
    for loss_type in loss_types:
        csv_path = f'runs/detect/loss_comparison/train_{loss_type}/results.csv'
        if Path(csv_path).exists():
            print(f"  ✓ {loss_type.upper()}: {csv_path}")
        else:
            print(f"  ✗ {loss_type.upper()}: 未生成")
    
    print("\n下一步：运行 plot_loss_comparison.py 来生成对比图")


def copy_existing_results():
    """
    如果你已经有训练好的模型，可以用这个函数收集结果
    """
    print("正在收集已有的训练结果...")
    
    # 定义你已有的训练结果路径
    existing_runs = {
        # 'iou': 'runs/detect/train1/results.csv',
        # 'eiou': 'runs/detect/train2/results.csv',
        # 'nwd': 'runs/detect/train3/results.csv',
        # 'mtgloss': 'runs/detect/train4/results.csv',
    }
    
    # 创建统一的对比目录
    output_dir = Path('runs/detect/loss_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    collected = {}
    for loss_type, csv_path in existing_runs.items():
        if Path(csv_path).exists():
            dest = output_dir / f'train_{loss_type}' / 'results.csv'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(csv_path, dest)
            collected[loss_type] = str(dest)
            print(f"✓ 复制 {loss_type}: {csv_path} -> {dest}")
        else:
            print(f"✗ 未找到 {loss_type}: {csv_path}")
    
    if collected:
        print(f"\n✅ 成功收集 {len(collected)} 个结果")
        print("可以运行 plot_loss_comparison.py 生成对比图")
    else:
        print("\n❌ 未找到任何有效结果")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='损失函数对比训练')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full', 'collect'],
                       help='训练模式: quick=快速验证, full=完整训练, collect=收集已有结果')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数（quick模式默认50，full模式建议300+）')
    parser.add_argument('--data', type=str, default='data.yaml',
                       help='数据集配置文件路径')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        copy_existing_results()
    elif args.mode == 'quick':
        print("⚠️ 快速模式：仅用于验证收敛趋势，不代表最终性能")
        quick_comparison_train(epochs=args.epochs)
    elif args.mode == 'full':
        print("📊 完整训练模式：将训练至收敛")
        quick_comparison_train(epochs=args.epochs)
    
    print("\n" + "="*60)
    print("提示：训练完成后运行以下命令生成对比图：")
    print("  python plot_loss_comparison.py")
    print("="*60)
