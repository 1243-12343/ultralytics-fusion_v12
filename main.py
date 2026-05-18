from ultralytics import YOLO
from torch.optim import Adam
import torch
from ultralytics.utils.torch_utils import model_info, get_flops
import thop

# # 关闭确定性算法设置
# torch.use_deterministic_algorithms(False)

if __name__ == '__main__':
    ############## 这是train的代码 ##############
    # 关闭确定性算法设置
    torch.use_deterministic_algorithms(False)

    # yolov11
    model = YOLO(r"C:\Users\Administrator\Desktop\yolov12new\yolov12new\ultralytics-fusion_v12mode1\ultralytics-fusion_v12mode1\ultralytics-fusion_v12\ultralytics\cfg\models\v12\yolov12nir-rgb-clpfres-p2di-p3-hcfm.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/11/yolo11-twoCSP.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/11/yolo11-twoCSP-CTF.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/11/yolo11-twoCSP-CFE.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/11/yolo11-twoCSP-CTF-CFE.yaml")  # 初始化模型

    # yolov8
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-earlyfusion.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-CTF.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-CFE.yaml")  # 初始化模型
    # model = YOLO(r"ultralytics/cfg/models/v8/yolov8-twoCSP-CTF-CFE.yaml")  # 初始化模型

    #强制显示模型信息和GFLOPs（使用 YOLO 官方方法）
    print("\n" + "=" * 60)
    print("Model Summary (YOLO Official):")
    print("=" * 60)
    # model.info(verbose=True) 显示模型结构
    model.info(verbose=True)

    # 使用 thop 手动计算 GFLOPs（模拟 YOLO 官方计算方式）
    input_shape = (1, 6, 512, 512)  # 批次大小为1，6个通道，512x512分辨率
    input_tensor = torch.randn(input_shape).to(next(model.model.parameters()).device)
    macs, params = thop.profile(model.model, inputs=(input_tensor,), verbose=False)
    gflops = macs * 2

    print(f"\n{'='*60}")
    print(f"FLOPs Calculation (YOLO-style with thop):")
    print(f"{'='*60}")
    print(f"Input shape: {input_shape}")
    print(f"Total Params: {params / 1e6:.2f}M")
    print(f"Total MACs: {macs / 1e9:.2f} GMACs")
    print(f"Total FLOPs: {gflops / 1e9:.2f} GFLOPs")
    print(f"{'='*60}")



    # Set training configuration
    model.train(
        data=r"F:\datasets\mydata.yaml",
        #D:\flir\flir\flir\mydata.yaml  F:\datasets\mydata.yaml
        batch=8,
        imgsz=512,
        epochs=150,
        project='runs_second',
        name='yolov12nir-rgb-base-p2di-p3-hcfm',
        workers=1,
        cache=False,
        optimizer="SGD",
        single_cls=False,  # Set to True if single class detection
        device=0,  # GPU device (use 0 for first GPU)
        # Disable mixed precision training
        verbose=True,  # Enable verbose output
        profile=True,  # Enable profiling to show GFLOPs
        deterministic=False,  # Disable deterministic algorithms
        clpf_loss_weight=0.05,
        mf_sigreg_loss_weight=1.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        resume=False,
        amp=True,
    )  # 训练

    ############## 这是val和predict的代码 ##############
    # model = YOLO(r"runs\yolov11-earlyfusion9\weights\best.pt")
    # model.val(data=r"ultralytics/cfg/datasets/mydata.yaml", batch=1, save_json=True, save_txt=False)  # 验证
    # # model.predict(source=r"datasets/kaist_7601/images/test/set06_V000_I00019.jpg", save=True)  #   检测

# import torch
# from ultralytics import YOLO
# torch.use_deterministic_algorithms(False)
# # 清理缓存
# torch.cuda.empty_cache()
#
# model = YOLO('ultralytics/cfg/models/v12/yolov12n-twoCSP.yaml')
# model.train(data='ultralytics/cfg/datasets/mydata.yaml', workers=0, epochs=1, batch=12,resume= True)

# 训练结束后清理缓存
# torch.cuda.empty_cache()