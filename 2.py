import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

# 根据你的工程实际导入 GLFA
from ultralytics.nn.modules.common_utils import GLFA


def load_image(img_path, img_size=640, device="cuda:0"):
    """读入一张图片并预处理为 (1, 3, H, W) 的张量"""
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),              # [0,1]
        # 如有需要可加归一化
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # (1, 3, H, W)
    return x.to(device)


def save_feature_maps(feat, save_dir, prefix, max_channels=16):
    """
    将特征图保存为图片：
    - feat: (B, C, H, W)
    - 只保存前 max_channels 个通道的可视化
    """
    os.makedirs(save_dir, exist_ok=True)
    b, c, h, w = feat.shape

    # 只取 batch 中第 1 个样本
    feat = feat[0:1, :min(c, max_channels), :, :]  # (1, C', H, W)

    # 归一化到 0-1，便于保存
    feat_min = feat.min()
    feat_max = feat.max()
    feat_norm = (feat - feat_min) / (feat_max - feat_min + 1e-6)

    # 把每个通道排成一个 grid
    grid = vutils.make_grid(
        feat_norm, nrow=4, normalize=False, scale_each=False
    )
    save_path = os.path.join(save_dir, f"{prefix}.png")
    vutils.save_image(grid, save_path)
    print(f"[保存特征图] {prefix}: {save_path}, 形状 = {feat.shape}")


def main():
    # ======== 配置部分 ========
    img_path = "input.jpg"          # TODO: 换成你的图片路径
    save_dir = "runs/glfa_feats"         # 特征图保存目录
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 这里假设直接对 RGB 图像使用 GLFA，因此 in_channels = 3
    # 如果你在骨干网络中使用 GLFA，应把 in_channels 改成对应特征图通道数
    model = GLFA(
        in_channels=3,
        scale='p3',
        channel_factor=2,
        attention_ratio=8,
        use_coordconv=True,
        act=nn.ReLU
    ).to(device)
    model.eval()

    # ======== 注册 forward hook，捕获中间特征图 ========
    features = {}

    def get_hook(name):
        def hook(module, input, output):
            # output 是该层的输出特征图
            features[name] = output.detach()
        return hook

    # 你关心哪些阶段，就给哪些模块挂钩子
    # 这里示例：branch1、branch2、拼接后的特征、fuse 输出
    # 注意：feat 拼接后的张量在 forward 里是一个中间变量，不能直接 hook，
    #       所以我们改为 hook 对应的模块输出：
    model.branch1.register_forward_hook(get_hook("branch1_out"))
    model.branch2.register_forward_hook(get_hook("branch2_out"))
    model.fuse.register_forward_hook(get_hook("fuse_out"))

    # ======== 读入图片并前向传播 ========
    x = load_image(img_path, img_size=640, device=device)
    print(f"[输入] x 形状: {x.shape}")

    with torch.no_grad():
        out = model(x)

    print(f"[输出] GLFA 最终输出 out 形状: {out.shape}")

    # ======== 保存特征图 ========
    # 这里仅示范保存若干关键特征图，
    # 你可以根据需要增加更多（比如改 forward 返回中间 feat）
    for name, feat in features.items():
        # feat: (B, C, H, W)
        save_feature_maps(feat, save_dir, prefix=name, max_channels=16)

    # 也可以保存最终输出 out 的特征图
    save_feature_maps(out, save_dir, prefix="out_final", max_channels=16)


if __name__ == "__main__":
    main()