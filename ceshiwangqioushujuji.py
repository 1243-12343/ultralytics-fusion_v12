import os
import shutil
import random
from pathlib import Path


def split_dataset(image_folder, label_folder, output_dir, train_ratio=0.8, val_ratio=0.2, copy_files=True):
    """
    按照比例划分数据集，保持图片和标签文件的对应关系

    参数:
        image_folder: 图片文件夹路径
        label_folder: 标签文件夹路径 (txt文件)
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        copy_files: True为复制文件，False为移动文件
    """

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    # 检查比例总和是否为1
    if abs(train_ratio + val_ratio - 1.0) > 0.001:
        print("❌ 错误：训练集和验证集比例之和必须为1")
        return

    # 创建输出目录结构
    train_img_dir = Path(output_dir) / 'train' / 'images'
    train_label_dir = Path(output_dir) / 'train' / 'labels'
    val_img_dir = Path(output_dir) / 'val' / 'images'
    val_label_dir = Path(output_dir) / 'val' / 'labels'

    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件（不含扩展名）
    image_files = []
    if os.path.exists(image_folder):
        for file_path in Path(image_folder).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path.stem)  # 只取文件名（不含扩展名）

    print(f"📊 找到 {len(image_files)} 个图片文件")

    # 检查对应的txt文件是否存在
    valid_files = []
    for img_name in image_files:
        txt_path = Path(label_folder) / f"{img_name}.txt"
        if txt_path.exists():
            valid_files.append(img_name)
        else:
            print(f"⚠️  警告：图片 {img_name} 没有对应的标签文件")

    print(f"📊 有效文件对（图片+标签）：{len(valid_files)} 个")

    if len(valid_files) == 0:
        print("❌ 没有找到有效的文件对，请检查文件夹路径和文件对应关系")
        return

    # 随机打乱文件列表
    random.shuffle(valid_files)

    # 计算划分数量
    train_count = int(len(valid_files) * train_ratio)
    val_count = len(valid_files) - train_count

    # 划分文件
    train_files = valid_files[:train_count]
    val_files = valid_files[train_count:]

    print(f"📁 训练集: {len(train_files)} 个文件")
    print(f"📁 验证集: {len(val_files)} 个文件")

    # 复制/移动文件函数
    def process_files(file_list, img_dest_dir, label_dest_dir):
        for file_name in file_list:
            # 处理图片文件（需要找到原始扩展名）
            img_source = None
            for ext in image_extensions:
                possible_path = Path(image_folder) / f"{file_name}{ext}"
                if possible_path.exists():
                    img_source = possible_path
                    break

            if img_source and img_source.exists():
                # 处理标签文件
                label_source = Path(label_folder) / f"{file_name}.txt"

                if copy_files:
                    shutil.copy2(img_source, img_dest_dir / img_source.name)
                    shutil.copy2(label_source, label_dest_dir / label_source.name)
                else:
                    shutil.move(str(img_source), str(img_dest_dir / img_source.name))
                    shutil.move(str(label_source), str(label_dest_dir / label_source.name))

    # 处理训练集
    print("🔄 处理训练集文件...")
    process_files(train_files, train_img_dir, train_label_dir)

    # 处理验证集
    print("🔄 处理验证集文件...")
    process_files(val_files, val_img_dir, val_label_dir)

    print("✅ 数据集划分完成！")
    print(f"📁 输出目录结构:")
    print(f"   {output_dir}/")
    print(f"   ├── train/")
    print(f"   │   ├── images/ (训练集图片)")
    print(f"   │   └── labels/ (训练集标签)")
    print(f"   └── val/")
    print(f"       ├── images/ (验证集图片)")
    print(f"       └── labels/ (验证集标签)")


def split_dataset_with_test(image_folder, label_folder, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                            copy_files=True):
    """
    划分训练集、验证集和测试集
    """
    # 检查比例总和
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        print("❌ 错误：训练集、验证集和测试集比例之和必须为1")
        return

    # 创建输出目录
    splits = ['train', 'val', 'test']
    dirs = {}
    for split in splits:
        dirs[f"{split}_img_dir"] = Path(output_dir) / split / 'images'
        dirs[f"{split}_label_dir"] = Path(output_dir) / split / 'labels'
        dirs[f"{split}_img_dir"].mkdir(parents=True, exist_ok=True)
        dirs[f"{split}_label_dir"].mkdir(parents=True, exist_ok=True)

    # 获取有效文件对
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    valid_files = []

    if os.path.exists(image_folder):
        for file_path in Path(image_folder).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                img_name = file_path.stem
                txt_path = Path(label_folder) / f"{img_name}.txt"
                if txt_path.exists():
                    valid_files.append(img_name)

    print(f"📊 找到 {len(valid_files)} 个有效文件对")

    if len(valid_files) == 0:
        print("❌ 没有找到有效的文件对")
        return

    # 随机打乱并划分
    random.shuffle(valid_files)
    total = len(valid_files)

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    train_files = valid_files[:train_count]
    val_files = valid_files[train_count:train_count + val_count]
    test_files = valid_files[train_count + val_count:]

    print(f"📁 训练集: {len(train_files)} 个文件")
    print(f"📁 验证集: {len(val_files)} 个文件")
    print(f"📁 测试集: {len(test_files)} 个文件")

    # 处理文件函数
    def process_files(file_list, img_dest_dir, label_dest_dir):
        for file_name in file_list:
            # 找到图片文件
            img_source = None
            for ext in image_extensions:
                possible_path = Path(image_folder) / f"{file_name}{ext}"
                if possible_path.exists():
                    img_source = possible_path
                    break

            if img_source and img_source.exists():
                label_source = Path(label_folder) / f"{file_name}.txt"

                if copy_files:
                    shutil.copy2(img_source, img_dest_dir / img_source.name)
                    shutil.copy2(label_source, label_dest_dir / label_source.name)
                else:
                    shutil.move(str(img_source), str(img_dest_dir / img_source.name))
                    shutil.move(str(label_source), str(label_dest_dir / label_source.name))

    # 处理各数据集
    print("🔄 处理训练集...")
    process_files(train_files, dirs['train_img_dir'], dirs['train_label_dir'])

    print("🔄 处理验证集...")
    process_files(val_files, dirs['val_img_dir'], dirs['val_label_dir'])

    print("🔄 处理测试集...")
    process_files(test_files, dirs['test_img_dir'], dirs['test_label_dir'])

    print("✅ 数据集划分完成！")
    print(f"📁 输出目录: {output_dir}")


# 使用示例
if __name__ == "__main__":
    # 请修改为你的实际路径
    IMAGE_FOLDER = r"F:\sda\train"  # 替换为你的图片文件夹路径
    LABEL_FOLDER = r"F:\sda\label"  # 替换为你的标签文件夹路径
    OUTPUT_DIR = r"F:\sda\out"  # 输出目录

    # 设置随机种子以便复现结果
    random.seed(42)

    print("🚀 开始划分数据集...")

    # 方法1: 只划分训练集和验证集
    split_dataset(
        image_folder=IMAGE_FOLDER,
        label_folder=LABEL_FOLDER,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8,
        val_ratio=0.2,
        copy_files=True  # True复制文件，False移动文件
    )

    # 方法2: 划分训练集、验证集和测试集（取消注释使用）
    # split_dataset_with_test(
    #     image_folder=IMAGE_FOLDER,
    #     label_folder=LABEL_FOLDER,
    #     output_dir=OUTPUT_DIR + "_with_test",
    #     train_ratio=0.7,
    #     val_ratio=0.2,
    #     test_ratio=0.1,
    #     copy_files=True
    # )