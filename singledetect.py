# from ultralytics import YOLO
# from torch.optim import Adam
# import torch
# from ultralytics.utils.torch_utils import model_info, get_flops
# import thop
#
# model = YOLO(r"C:\Users\Administrator\Desktop\xiaorongshiyan\cag+ta gdf\yolov11-earlyfusion45\weights\best.pt")
# # model.val(data=r"ultralytics/cfg/datasets/mydata.yaml", batch=1, save_json=True, save_txt=False)  # 验证
# path
# # 完整的预测参数设置
# model.predict(
#     source=r"C:\Users\Administrator\Desktop\datasets\M3FD_yolo\images\train\03000.png",
#     save=True,
#     conf=0.60,      # 置信度阈值
#     iou=0.45,       # NMS的IOU阈值
#     show_labels=True,    # 显示标签
#     show_conf=True,      # 显示置信度分数
#     line_width=2,   # 边界框线宽
#     max_det=100,    # 每张图像最大检测数量
#     augment=False   # 是否使用测试时数据增强
# )

from ultralytics import YOLO

model = YOLO(r"C:\Users\Administrator\Desktop\第一篇论文\xiaorongshiyan\msaa+ta gdf p2\yolov11-earlyfusion52\weights\best.pt")
#
# # 指定具体的四张图片路径
image_paths = [
    r"C:\Users\Administrator\Desktop\datasets\M3FD_yolo\images\train\01444.png",
    r"C:\Users\Administrator\Desktop\datasets\M3FD_yolo\images\train\02212.png",
    r"C:\Users\Administrator\Desktop\datasets\M3FD_yolo\images\train\00000.png",
    r"C:\Users\Administrator\Desktop\datasets\M3FD_yolo\images\train\03000.png"
]

model(
    source=r"C:\Users\Administrator\Desktop\datasets\M3FD_yolo\images\train\03000.png",  # 图片路径列表
    save=True,
    conf=0.4,
    iou=0.8,
    show_labels=False,
    show_conf=True,
    line_width=1,
    # max_det=100,
    # augment=False
)