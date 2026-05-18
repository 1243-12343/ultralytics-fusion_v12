# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
# import os
#
# # BILIBILI UP 魔傀面具
# # 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()
# # 预测框粗细和颜色修改问题可看<新手推荐学习视频.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点
#
# if __name__ == '__main__':
#     # 加载模型
#     model = YOLO(r'C:\Users\Administrator\Desktop\ultralytics-fusion_v12mode1\ultralytics-fusion_v12mode1\quanzhong\150\best.pt')
#
#     # 设置数据源路径
#     source_path = r'C:\Users\Administrator\Desktop\ultralytics-fusion_v12mode1\ultralytics-fusion_v12mode1\datasetsscompetition\diedao\images\train'
#
#     # 检查路径是否存在
#     if not os.path.exists(source_path):
#         print(f"错误：路径不存在 {source_path}")
#         exit()
#
#     # 统计图片数量
#     image_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
#     print(f"找到 {len(image_files)} 张图片")
#
#     # 开始检测
#     print("开始检测...")
#     results = model.predict(source=source_path,
#                            imgsz=640,
#                            project='runs/detect',
#                            name='exp',
#                            save=True,
#                            conf=0.25,        # 置信度阈值
#                            iou=0.7,          # NMS的IoU阈值
#                            save_txt=True,    # 保存检测结果为txt文件
#                            save_conf=True,   # 在txt文件中保存置信度
#                            # agnostic_nms=True,
#                            # visualize=True, # visualize model features maps
#                            # line_width=2, # line width of the bounding boxes
#                            # show_conf=True, # show prediction confidence
#                            # show_labels=True, # show prediction labels
#                            # save_crop=True, # save cropped images with results
#                          )
#
#     print("检测完成！")
#     print(f"结果保存在: runs/detect/exp/")


import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os

# BILIBILI UP 魔傀面具
# 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()
# 预测框粗细和颜色修改问题可看<新手推荐学习视频.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

if __name__ == '__main__':
    # 加载模型
    model = YOLO(
        r'C:\Users\Administrator\Desktop\xiaorongshiyan\msaa+ta gdf p2\yolov11-earlyfusion52\weights\best.pt')

    # 设置数据源路径
    source_path = r'C:\Users\Administrator\Desktop\datasets\M3FD_yolo\images\train'

    # 检查路径是否存在
    if not os.path.exists(source_path):
        print(f"错误：路径不存在 {source_path}")
        exit()

    # 统计图片数量
    image_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    print(f"找到 {len(image_files)} 张图片")

    # 开始检测
    print("开始检测...")
    results = model.predict(source=source_path,
                            imgsz=640,
                            project='runs/detect',
                            name='exp',
                            save=True,
                            conf=0.25,  # 置信度阈值
                            iou=0.7,  # NMS的IoU阈值
                            save_txt=True,  # 保存检测结果为txt文件
                            save_conf=True,  # 在txt文件中保存置信度
                            # agnostic_nms=True,
                            line_width=1,
                            # visualize=True, # visualize model features maps
                            # line_width=2, # line width of the bounding boxes
                            # show_conf=True, # show prediction confidence
                            # show_labels=True, # show prediction labels
                            # save_crop=True, # save cropped images with results
                            )

    print("检测完成！")
    print(f"结果保存在: runs/detect/exp/")