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
    model = YOLO(r"C:\Users\Administrator\Desktop\yolov12new\yolov12new\ultralytics-fusion_v12mode1\ultralytics-fusion_v12mode1\ultralytics-fusion_v12\ultralytics\cfg\models\v12\yolov12nir-rgb-base.yaml")  # 初始化模型
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

    # 强制显示模型信息和GFLOPs
    # print("\nModel Summary:")
    # model_info(model.model, verbose=True)  # 显示详细模型信息
    #
    # # 手动计算FLOPs
    # input_shape = (1, 6, 640, 640)  # 批次大小为1，6个通道，640x640分辨率
    # input_tensor = torch.randn(input_shape)
    # flops, params = thop.profile(model.model, inputs=(input_tensor,), verbose=False)
    # print(f"\nDetailed FLOPs calculation:")
    # print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")
    # print(f"Total params: {params / 1e6:.2f}M")

    # Set training configuration
    model.train(
        data=r"D:\flir\flir\flir\mydata.yaml",
        batch=8,
        epochs=1000,
        project='runs',
        name='yolov11-earlyfusion',
        workers=0,
        optimizer="SGD",
        single_cls=False,  # Set to True if single class detection
        device=0,  # GPU device (use 0 for first GPU)
        # Disable mixed precision training
        verbose=True,  # Enable verbose output
        profile=True,  # Enable profiling to show GFLOPs
        deterministic=False,  # Disable deterministic algorithms
        resume=True,
        amp=False,
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