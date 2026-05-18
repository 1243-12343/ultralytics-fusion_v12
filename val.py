from ultralytics import YOLO
import torch
import time
from ultralytics.utils.torch_utils import model_info
from ultralytics.utils import yaml_load
import thop
import warnings

warnings.filterwarnings("ignore")


# -------------------- 可调参数 --------------------
IMGSZ = 512
BATCH_VAL = 4
SMALL_AREA_THR = 32.0  # COCO small: area < SMALL_AREA_THR^2 像素（原图尺度）

# 纯推理 FPS（不含 dataloader / NMS），与验证日志里的 Speed 口径不同
RUN_PURE_INFERENCE_BENCH = True
FPS_WARMUP = 20
FPS_ITERS = 100
FPS_USE_FP16 = True  # 与常见论文推理设定一致；无 GPU 会自动关掉


def resolve_val_split(data_yaml_path, preferred_split="val"):
    """Pick a split that exists in the dataset yaml (Ultralytics expects train/val/test paths)."""
    cfg = yaml_load(data_yaml_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid dataset yaml: {data_yaml_path}")
    if cfg.get(preferred_split):
        return preferred_split
    for alt in ("val", "test", "train"):
        if alt != preferred_split and cfg.get(alt):
            print(
                f"WARNING: dataset yaml has no '{preferred_split}' path; "
                f"using split='{alt}' instead. Add '{preferred_split}: <path>' to the yaml if needed."
            )
            return alt
    raise FileNotFoundError(
        f"Dataset yaml must define at least one of train/val/test with image paths. "
        f"File: {data_yaml_path}\nKeys found: {list(cfg.keys())}"
    )


def print_fps_from_validation(metrics):
    """Validation 过程中统计的 per-image 耗时（含预处理、推理、后处理分段）。"""
    speed = getattr(metrics, "speed", None)
    if not isinstance(speed, dict) or not speed:
        print("\n================ Speed / FPS (from val) =================")
        print("No speed dict on metrics (unexpected).")
        print("=========================================================\n")
        return

    preprocess_ms = float(speed.get("preprocess", 0.0) or 0.0)
    inference_ms = float(speed.get("inference", 0.0) or 0.0)
    postprocess_ms = float(speed.get("postprocess", 0.0) or 0.0)
    total_ms = preprocess_ms + inference_ms + postprocess_ms
    inference_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    total_fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    print("\n================ Speed / FPS (from val) =================")
    print("Per-image ms (Ultralytics validator). Batch affects dataloader/amortization.")
    print(f"Preprocess:   {preprocess_ms:.3f} ms/image")
    print(f"Inference:    {inference_ms:.3f} ms/image -> {inference_fps:.2f} FPS")
    print(f"Postprocess:  {postprocess_ms:.3f} ms/image")
    print(f"End-to-end:   {total_ms:.3f} ms/image -> {total_fps:.2f} FPS")
    print("=========================================================\n")


def print_coco_small_metrics(metrics, area_side_thr: float):
    """COCO-style small: GT bbox 原图像素面积 < area_side_thr^2。"""
    rd = getattr(metrics, "results_dict", None)
    if not isinstance(rd, dict):
        print("\n================ COCO Small Objects =================")
        print("metrics.results_dict missing.")
        print("=====================================================\n")
        return

    k50 = "metrics/mAP50_small(B)"
    k95 = "metrics/mAP50-95_small(B)"
    kn = "metrics/small_instances(B)"

    print("\n================ COCO Small Objects =================")
    print(f"Definition: GT bbox area < {area_side_thr:g}^2 pixels (original image space)")
    if k50 in rd:
        print(f"Small instances: {int(rd.get(kn, 0))}")
        print(f"mAP50_small:     {float(rd.get(k50, 0.0)):.4f}")
        print(f"mAP50-95_small:  {float(rd.get(k95, 0.0)):.4f}")
    else:
        print("Small-object keys not in results_dict.")
        print("Ensure you use this repo's DetectionValidator (small-area metrics wired in detect/val.py).")
    print("=====================================================\n")


def benchmark_pure_inference_fps(model, imgsz: int, warmup: int, iters: int, use_fp16: bool):
    """仅模型前向（单张 6 通道），不包含 NMS / 读图。适合论文 reporting。"""
    device = next(model.model.parameters()).device
    cuda = device.type == "cuda"
    use_fp16 = use_fp16 and cuda

    x = torch.randn(1, 6, imgsz, imgsz, device=device)
    if use_fp16:
        x = x.half()

    m = model.model
    was_training = m.training
    m.eval()
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                m(x)
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                m(x)
            if cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
    finally:
        if was_training:
            m.train()

    elapsed = t1 - t0
    fps = iters / elapsed if elapsed > 0 else 0.0
    ms = 1000.0 * elapsed / iters if iters > 0 else 0.0

    print("\n================ Pure inference FPS =================")
    print(f"Input: 1x6x{imgsz}x{imgsz}, device={device}, FP16={use_fp16}")
    print(f"Warmup/iters: {warmup}/{iters}")
    print(f"Latency: {ms:.3f} ms/batch  |  FPS: {fps:.2f} images/s")
    print("(Model forward only; no NMS, no I/O.)")
    print("=====================================================\n")


# 模型与数据 YAML
model_yaml_path = r"C:\Users\Administrator\Desktop\yolov12new\yolov12new\ultralytics-fusion_v12mode1\ultralytics-fusion_v12mode1\ultralytics-fusion_v12\runs_second\yolov12nir-rgb-base-p2di-p3-hcfm12\weights\best.pt"
data_yaml_path = r"C:\Users\Administrator\Desktop\datasets\mydata.yaml"
#C:\Users\Administrator\Desktop\datasets\mydata.yaml  D:\flir\flir\flir\mydata.yaml
if __name__ == "__main__":
    model = YOLO(model_yaml_path)

    print("\nModel Summary:")
    model_info(model.model, verbose=True)

    input_shape = (1, 6, IMGSZ, IMGSZ)
    input_tensor = torch.randn(input_shape, device=next(model.model.parameters()).device)
    macs, params = thop.profile(model.model, inputs=(input_tensor,), verbose=False)
    yolo_gflops = macs * 2
    print("\nDetailed FLOPs calculation:")
    print(f"Input shape: {input_shape}")
    print(f"Total MACs (THOP): {macs / 1e9:.2f} GMACs")
    print(f"Total FLOPs (YOLO-style): {yolo_gflops / 1e9:.2f} GFLOPs")
    print(f"Total params: {params / 1e6:.2f}M")

    if RUN_PURE_INFERENCE_BENCH:
        benchmark_pure_inference_fps(
            model, IMGSZ, FPS_WARMUP, FPS_ITERS, FPS_USE_FP16
        )

    val_split = resolve_val_split(data_yaml_path, preferred_split="test")

    metrics = model.val(
        data=data_yaml_path,
        split=val_split,
        imgsz=IMGSZ,
        batch=BATCH_VAL,
        project="runs/val",
        name="exp",
        rmopp=False,
        rmopp_gamma1=1.15,
        rmopp_uncertainty_weight=0.3,
        rmopp_mode="soft",
        rmopp_max_penalty=0.20,
        small_area_thr=SMALL_AREA_THR,
    )

    print_fps_from_validation(metrics)
    print_coco_small_metrics(metrics, SMALL_AREA_THR)

    # -------------------- Weak-class reference metric --------------------
    weak_map95_threshold = 0.40
    weak_focus_weight = 0.40

    if hasattr(metrics, "maps") and hasattr(metrics, "results_dict"):
        maps95 = [float(v) for v in metrics.maps]
        names = getattr(metrics, "names", model.names)
        if isinstance(names, (list, tuple)):
            names = {i: n for i, n in enumerate(names)}

        weak_ids = [i for i, v in enumerate(maps95) if v < weak_map95_threshold]
        overall_map95 = float(
            metrics.results_dict.get("metrics/mAP50-95(B)", sum(maps95) / max(len(maps95), 1))
        )
        weak_avg_map95 = (
            (sum(maps95[i] for i in weak_ids) / len(weak_ids)) if weak_ids else overall_map95
        )
        reference_score = (1.0 - weak_focus_weight) * overall_map95 + weak_focus_weight * weak_avg_map95

        map50_per_class = [float("nan")] * len(maps95)
        if hasattr(metrics, "box") and hasattr(metrics.box, "all_ap"):
            ap_class_index = [int(v) for v in getattr(metrics, "ap_class_index", [])]
            all_ap = list(getattr(metrics.box, "all_ap", []))
            for k, c in enumerate(ap_class_index):
                if 0 <= c < len(map50_per_class) and k < len(all_ap):
                    map50_per_class[c] = float(all_ap[k][0])

        print("\n================ Weak-Class Diagnostic ================")
        print(f"Overall mAP50-95: {overall_map95:.4f}")
        print(f"Weak threshold (mAP50-95): {weak_map95_threshold:.2f}")
        print(f"Weak-focus weight: {weak_focus_weight:.2f}")
        print(f"Reference score (for tuning): {reference_score:.4f}")

        if weak_ids:
            print("Underperforming classes:")
            for c in sorted(weak_ids, key=lambda x: maps95[x]):
                name = names.get(c, str(c))
                map50_text = "NA" if map50_per_class[c] != map50_per_class[c] else f"{map50_per_class[c]:.4f}"
                print(f"  - {name:<12} mAP50={map50_text}  mAP50-95={maps95[c]:.4f}")
        else:
            print("No underperforming class under current threshold.")
        print("=======================================================\n")
