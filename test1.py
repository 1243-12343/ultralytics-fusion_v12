#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化 CascadeWaveletFusion (aka CAG) 中 RGB/IR 通道门控权重的热力图
Usage:
    python visualize_cag_gates.py \
        --model ultralytics/cfg/models/v12/p2-glfa-nwdloss-siou.yaml \
        --weights runs/train/exp/weights/best.pt \
        --rgb data/samples/night_rgb.jpg \
        --ir  data/samples/night_ir.jpg \
        --out results/night_demo \
        --stage-idx 9 12 17
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ops


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/v12/p2-glfa-nwdloss-siou.yaml",
        help="可选：模型 YAML（若 weights 已含结构，可省略）",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default=r"C:\Users\Administrator\Desktop\yolov12new\yolov12new\ultralytics-fusion_v12mode1\ultralytics-fusion_v12mode1\ultralytics-fusion_v12\best.pt",
        help="训练好的权重文件 .pt",
    )
    ap.add_argument(
        "--rgb",
        type=str,
        default=r"C:\Users\Administrator\Desktop\datasets\M3FD_yolo\images\train\00000.png",
        help="RGB 可见光图片路径",
    )
    ap.add_argument(
        "--ir",
        type=str,
        default=r"C:\Users\Administrator\Desktop\datasets\M3FD_yolo\image\train\00000.png",
        help="IR/热成像图片路径",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="results/night_demo11",
        help="输出目录",
    )
    ap.add_argument(
        "--stage-idx",
        nargs="+",
        type=int,
        default=[9, 12, 17],
        help="需要抓取的 CascadeWaveletFusion 节点编号（按 YAML 序号）",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap.parse_args()


def load_pair(rgb_path, ir_path, imgsz):
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.resize(rgb, (imgsz, imgsz))
    rgb_np = rgb[:, :, ::-1].copy().transpose(2, 0, 1)
    rgb_t = torch.from_numpy(rgb_np).float() / 255.0

    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    ir = cv2.resize(ir, (imgsz, imgsz))
    ir_rgb = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    ir_np = ir_rgb[:, :, ::-1].copy().transpose(2, 0, 1)
    ir_t = torch.from_numpy(ir_np).float() / 255.0

    return rgb, ir_rgb, rgb_t.unsqueeze(0), ir_t.unsqueeze(0)


def register_gate_hooks(yolo_model, target_ids):
    handles = []
    gate_records = {}

    # yolo_model could be the high-level YOLO wrapper or the raw nn.Sequential
    core_model = getattr(yolo_model, "model", yolo_model)
    module_list = getattr(core_model, "model", core_model)

    def make_hook(node_id):
        def hook(module, inputs, outputs):
            gate_records[node_id] = {
                "feat": outputs.detach().cpu().clone(),
                "feat_shape": outputs.shape,
            }
        return hook

    for node_id, layer in module_list.named_children():
        idx = int(node_id)
        if idx in target_ids and layer.__class__.__name__ == "CascadeWaveletFusion":
            handles.append(layer.register_forward_hook(make_hook(idx)))
    return handles, gate_records


def feature_to_heatmap(feat_tensor, ref_hw):
    """Average along channel dimension and upsample to image size."""
    feat_map = feat_tensor.mean(dim=1, keepdim=True)
    feat_map = torch.nn.functional.interpolate(
        feat_map, size=ref_hw, mode="bilinear", align_corners=False)
    feat_map = feat_map.squeeze().numpy()
    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-6)
    return feat_map


def overlay_heatmap(image_bgr, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heat_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    heat_color = cv2.resize(heat_color, (image_bgr.shape[1], image_bgr.shape[0]))
    blended = cv2.addWeighted(image_bgr, 1 - alpha, heat_color, alpha, 0)
    return blended


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    rgb_img, ir_img, rgb_t, ir_t = load_pair(args.rgb, args.ir, args.imgsz)
    fused_pair = torch.cat([rgb_t, ir_t], dim=1)  # [1,6,H,W], 匹配 Multiin 入口

    # 加载权重（无需再 eval()，直接设置推理模式）
    model = YOLO(args.weights)
    model.overrides["mode"] = "predict"
    model.overrides["data"] = None  # 阻止去加载 mydata.yaml
    model.to(args.device)

    handles, gates = register_gate_hooks(model, set(args.stage_idx))

    core_model = getattr(model, "model", model)
    with torch.no_grad():
        _ = core_model(fused_pair.to(args.device))  # 直接前向一次即可触发 hook

    for h in handles:
        h.remove()

    for stage_id, rec in gates.items():
        feat = rec["feat"]  # [1,C,H,W]
        feat_h, feat_w = rec["feat_shape"][2], rec["feat_shape"][3]

        feat_map = feature_to_heatmap(feat, (feat_h, feat_w))
        feat_overlay = overlay_heatmap(rgb_img, feat_map)

        cv2.imwrite(str(Path(args.out) / f"stage{stage_id}_feat.png"), feat_overlay)

    print(f"可视化完成，结果保存在 {args.out}")


if __name__ == "__main__":
    main()