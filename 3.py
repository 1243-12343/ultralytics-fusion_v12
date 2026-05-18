import os
import torch
import torchvision.utils as vutils

from ultralytics import YOLO
from ultralytics.nn.modules.common_utils import GLFA

def save_feature_grid(feat, save_path, b=0, max_channels=16):
    """
    feat: [B, C, H, W] 的 tensor（CPU 上）
    b:   第几张图（一般 0）
    max_channels: 最多可视化多少个通道
    """
    if feat is None:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 取 batch 中第 b 张图 -> [C, H, W]
    img = feat[b].detach().cpu()

    # 只取前 max_channels 个通道
    img = img[:max_channels]          # [N, H, W]

    # 变成 [N,1,H,W]，每个通道当一张灰度图
    img = img.unsqueeze(1)

    grid = vutils.make_grid(
        img,
        nrow=4,                # 每行 4 张
        normalize=True,
        scale_each=True
    )
    vutils.save_image(grid, save_path)
    print("saved:", save_path)


# 1. 加载模型
model = YOLO(r"C:\Users\Administrator\Desktop\xiaorongshiyan\msaa+ta gdf p2\yolov11-earlyfusion52\weights\best.pt")

# 2. 取出内部的 PyTorch 模型
net = model.model  # 对于 YOLOv8/11 通常是这样

# 3. 打开所有 GLFA 模块的 debug，并清空旧特征
for m in net.modules():
    if isinstance(m, GLFA):
        m.debug = True
        m.saved_features = {}

# 4. 正常执行一次推理
results = model(
    source=r"C:\Users\Administrator\Desktop\datasets\M3FD_yolo\images\train\00000.png",
    save=True,
    conf=0.3,
    iou=0.8,
    show_labels=True,
    show_conf=True,
    line_width=1,
)

# 5. 推理结束后，从每个 GLFA 里拿特征图并保存
out_dir = "debug_features"  # 所有特征图保存到这个文件夹

for idx, m in enumerate(net.modules()):
    if isinstance(m, GLFA):
        print(f"==== GLFA 模块 {idx} ====")
        print("keys:", list(m.saved_features.keys()))

        feat_input  = m.saved_features.get("input_before_coord")
        feat_b1     = m.saved_features.get("branch1_out")
        feat_concat = m.saved_features.get("concat_feat")
        feat_final  = m.saved_features.get("final_out_with_residual")

        # 只看 batch=0，前 16 个通道
        save_feature_grid(feat_input,
                          os.path.join(out_dir, f"glfa_{idx}_input_before_coord.png"),
                          b=0, max_channels=16)
        save_feature_grid(feat_b1,
                          os.path.join(out_dir, f"glfa_{idx}_branch1_out.png"),
                          b=0, max_channels=16)
        save_feature_grid(feat_concat,
                          os.path.join(out_dir, f"glfa_{idx}_concat_feat.png"),
                          b=0, max_channels=16)
        save_feature_grid(feat_final,
                          os.path.join(out_dir, f"glfa_{idx}_final_out.png"),
                          b=0, max_channels=16)