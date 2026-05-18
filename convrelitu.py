#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""。
可视化 CAG 模块中 GSConv/DWConv 加权特征与模态门控的热力图。
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


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
        default="results/conv_relitu",
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
    ap.add_argument(
        "--force-equal-gate",
        action="store_true",
        help="将 WaveletFusion 门控固定为 0.5/0.5（用于对比无门控效果）",
    )
    return ap.parse_args()


def load_pair(rgb_path, ir_path, imgsz):
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None:
        raise FileNotFoundError("RGB/IR 图像加载失败")
    rgb = cv2.resize(rgb, (imgsz, imgsz))
    ir = cv2.resize(ir, (imgsz, imgsz))
    ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    rgb_t = torch.from_numpy(rgb[:, :, ::-1].copy().transpose(2, 0, 1)).float() / 255.0
    ir_t = torch.from_numpy(ir[:, :, ::-1].copy().transpose(2, 0, 1)).float() / 255.0
    return rgb, ir, rgb_t.unsqueeze(0), ir_t.unsqueeze(0)


def register_gate_hooks(yolo_model, target_ids):
    handles = []
    gate_records = {}

    def safe_clone(tensor):
        return tensor.detach().cpu().clone() if tensor is not None else None

    core_model = getattr(yolo_model, "model", yolo_model)
    module_list = getattr(core_model, "model", core_model)

    def make_hook(node_id):
        def hook(module, inputs, outputs):
            wf = module.wavelet_fusion
            gate_records[node_id] = {
                "feat": outputs.detach().cpu().clone(),
                "ga": safe_clone(getattr(wf, "latest_ga", None)),
                "gb": safe_clone(getattr(wf, "latest_gb", None)),
                "branch_a": safe_clone(getattr(wf, "latest_branch_a", None)),
                "branch_b": safe_clone(getattr(wf, "latest_branch_b", None)),
                "feat_shape": outputs.shape,
            }
        return hook

    for node_id, layer in module_list.named_children():
        idx = int(node_id)
        if idx in target_ids and layer.__class__.__name__ == "CascadeWaveletFusion":
            handles.append(layer.register_forward_hook(make_hook(idx)))
    return handles, gate_records


def feature_to_heatmap(feat_tensor, ref_hw):
    feat_map = feat_tensor.mean(dim=1, keepdim=True)
    feat_map = torch.nn.functional.interpolate(
        feat_map, size=ref_hw, mode="bilinear", align_corners=False)
    feat_map = feat_map.squeeze().numpy()
    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-6)
    return feat_map


def dominance_heatmap(ga_tensor, gb_tensor, ref_hw):
    diff = (ga_tensor - gb_tensor).mean(dim=1, keepdim=True)
    diff = torch.nn.functional.interpolate(
        diff, size=ref_hw, mode="bilinear", align_corners=False)
    diff = diff.squeeze().numpy()
    max_abs = np.max(np.abs(diff)) + 1e-6
    diff_norm = diff / max_abs
    diff_vis = (diff_norm + 1) / 2  # [-1,1] -> [0,1]
    return diff_vis


def overlay_heatmap(image_bgr, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heat_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    heat_color = cv2.resize(heat_color, (image_bgr.shape[1], image_bgr.shape[0]))
    blended = cv2.addWeighted(image_bgr, 1 - alpha, heat_color, alpha, 0)
    return blended


def overlay_dominance(image_bgr, dom_map, alpha=0.5):
    dom_uint8 = np.clip(dom_map * 255, 0, 255).astype(np.uint8)
    dom_uint8 = cv2.resize(dom_uint8, (image_bgr.shape[1], image_bgr.shape[0]))
    color = np.zeros_like(image_bgr)
    color[..., 2] = dom_uint8  # red -> RGB 占优
    color[..., 0] = 255 - dom_uint8  # blue -> IR 占优
    blended = cv2.addWeighted(image_bgr, 1 - alpha, color, alpha, 0)
    return blended


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    rgb_img, ir_img, rgb_t, ir_t = load_pair(args.rgb, args.ir, args.imgsz)
    fused_pair = torch.cat([rgb_t, ir_t], dim=1)

    model = YOLO(args.weights)
    model.overrides["mode"] = "predict"
    model.overrides["data"] = None
    model.to(args.device)

    if args.force_equal_gate:
        core = getattr(model, "model", model)
        for layer in getattr(core, "model", core):
            if layer.__class__.__name__ == "CascadeWaveletFusion":
                layer.wavelet_fusion.force_equal_gate = True

    handles, gates = register_gate_hooks(model, set(args.stage_idx))

    core_model = getattr(model, "model", model)
    with torch.no_grad():
        _ = core_model(fused_pair.to(args.device))

    for h in handles:
        h.remove()

    for stage_id, rec in gates.items():
        feat = rec["feat"]
        feat_h, feat_w = rec["feat_shape"][2], rec["feat_shape"][3]

        feat_map = feature_to_heatmap(feat, (feat_h, feat_w))
        feat_overlay = overlay_heatmap(rgb_img, feat_map)
        cv2.imwrite(str(Path(args.out) / f"stage{stage_id}_feat.png"), feat_overlay)

        ga = rec.get("ga")
        gb = rec.get("gb")
        ba = rec.get("branch_a")
        bb = rec.get("branch_b")

        if ga is not None and gb is not None:
            ga_map = feature_to_heatmap(ga, (feat_h, feat_w))
            gb_map = feature_to_heatmap(gb, (feat_h, feat_w))
            dom_map = dominance_heatmap(ga, gb, (feat_h, feat_w))

            cv2.imwrite(str(Path(args.out) / f"stage{stage_id}_ga.png"), overlay_heatmap(rgb_img, ga_map))
            cv2.imwrite(str(Path(args.out) / f"stage{stage_id}_gb.png"), overlay_heatmap(rgb_img, gb_map))
            cv2.imwrite(str(Path(args.out) / f"stage{stage_id}_dominance.png"), overlay_dominance(rgb_img, dom_map))

        if ba is not None:
            ba_map = feature_to_heatmap(ba, (feat_h, feat_w))
            cv2.imwrite(str(Path(args.out) / f"stage{stage_id}_gsconv_gate.png"), overlay_heatmap(rgb_img, ba_map))

        if bb is not None:
            bb_map = feature_to_heatmap(bb, (feat_h, feat_w))
            cv2.imwrite(str(Path(args.out) / f"stage{stage_id}_dwconv_gate.png"), overlay_heatmap(rgb_img, bb_map))

    print(f"可视化完成，结果保存在 {args.out}")


if __name__ == "__main__":
    main()
