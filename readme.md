# YOLOv12 Multimodal Fusion (CAG-Net)

本项目是基于YOLOv12的多模态（可见光+红外）目标检测框架，实现了**CAG-Net**（论文名可能是CMM-DET）的核心模型架构。

## 模块命名对照表

| YAML配置名 | 论文模块名 | 功能描述 |
|-----------|-----------|---------|
| `GLFA` | **MCAA** (Multi-scale Cross-modal Attention) | 多尺度跨模态注意力模块，用于全局与局部特征聚合 |
| `CascadeWaveletFusion` | **CAG** (Cascade Aggregation Fusion) | 级联聚合融合模块，基于小波变换实现双流特征融合 |

## 模型配置

### CMM-DET论文模型 (推荐使用)

```yaml
# 完整论文模型配置
p2-glfa-nwdloss-siou.yaml
```

**模型特点：**
- 双流骨干网络（可见光 + 红外）
- 4阶段级联融合（CAG模块）
- 多尺度特征聚合（MCAA模块）
- P2高分辨率检测层
- NWD Loss + SIoU Loss

### 其他可用配置

| 配置文件 | 说明 |
|---------|------|
| `p2-glfa-nwdloss-siou.yaml` | **完整论文模型** (P2 + GLFA + NWD + SIoU) |
| `p2-glfa-nwdloss-siou_noCWF.yaml` | 去掉CascadeWaveletFusion的变体 |
| `p2-glfa-nwdloss-siou_noGLFA.yaml` | 去掉GLFA的变体 |
| `p2-glfa-nwdloss-siou-nomsaa.yaml` | 去掉多尺度注意力 |
| `p2-glfa.yaml` | 基础GLFA融合模型 |
| `yolov12nir-rgb-base.yaml` | 基础双流融合模型 |

## 快速开始

### 环境配置

```bash
pip install ultralytics
```

### 训练模型

```python
from ultralytics import YOLO

# 加载CMM-DET论文模型
model = YOLO("ultralytics/cfg/models/v12/p2-glfa-nwdloss-siou.yaml")

# 开始训练（需要准备好多模态数据集）
model.train(
    data="path/to/your/mydata.yaml",  # 数据集配置
    epochs=100,
    imgsz=640,
    device=0
)
```

### 使用CLI训练

```bash
yolo detect train model=ultralytics/cfg/models/v12/p2-glfa-nwdloss-siou.yaml data=mydata.yaml epochs=100
```

## 数据集准备

数据集文件夹结构必须按照以下格式：

```
|-Datasets
    |-your_dataset_name
        |-images
            |-train
            |-val
        |-image          # 红外图像文件夹（注意：是image而不是images）
            |-train
            |-val
        |-labels
            |-train
            |-val
```

**数据集配置文件示例 (mydata.yaml)：**

```yaml
path: /path/to/Datasets/your_dataset_name
train: images/train
val: images/val

nc: 6  # 你的类别数
names: ['class1', 'class2', 'class3', 'class4', 'class5', 'class6']
```

## 模块架构说明

### MCAA (GLFA) - Multi-scale Cross-modal Attention

```python
# 代码位置: ultralytics/nn/modules/common_utils.py
class GLFA(nn.Module):
    """
    Lightweight GLFA module with AK convolution integration
    and scale-adaptive dilation
    """
```

**核心特性：**
- 非对称卷积核（AK Convolution）
- 坐标卷积（CoordConv）
- 通道注意力机制
- 尺度自适应空洞率

### CAG (CascadeWaveletFusion) - Cascade Aggregation Fusion

```python
# 代码位置: ultralytics/nn/modules/common_utils.py
class CascadeWaveletFusion(nn.Module):
    """基于小波变换的级联融合模块"""
```

**核心特性：**
- 小波域特征融合
- 多尺度级联处理
- 动态卷积权重

## 核心代码改动

相比原始YOLOv12，本项目的核心改动：

| 文件位置 | 改动说明 |
|---------|---------|
| `ultralytics/nn/modules/common_utils.py` | 新增GLFA、CascadeWaveletFusion等融合模块 |
| `ultralytics/nn/modules/__init__.py` | 模块导出 |
| `ultralytics/cfg/models/v12/` | 新增多模态模型配置文件 |
| `ultralytics/data/base.py` | 支持红外图像读取 |
| `ultralytics/data/augment.py` | HSV增强支持6通道图像 |
| `ultralytics/engine/validator.py` | 支持双模态验证 |
| `ultralytics/engine/predictor.py` | 支持双模态推理 |

## 添加新模块

如需添加新的融合模块，请按以下步骤：

1. **在 `ultralytics/nn/modules/common_utils.py` 中定义模块类**
2. **在 `ultralytics/nn/modules/__init__.py` 中导出**
3. **在 `ultralytics/nn/tasks.py` 的 `parse_model` 函数中注册**

## 引用

如果本项目对你的研究有帮助，请引用：

```
CMM-DET / CAG-Net论文引用信息
```

## 许可证

本项目基于Ultralytics YOLO构建，遵循AGPL-3.0许可证。
