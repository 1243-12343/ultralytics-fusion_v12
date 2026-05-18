模型配置文件：ultralytics-fusion\ultralytics\cfg\models\v12\yolov12n-twoCSP.yaml
数据集配置文件：ultralytics/cfg/datasets/mydata.yaml  
根据自己情况修改'path'路径

## 数据集准备
数据集文件夹需严格按下面命名：
```
|-Datasets
    |-llvip
        |-images
        |-image  # 额外的图片文件夹，放红外图
        |-labels
```

## 训练/验证/检测
运行main.py即可

## Yolov12训添加一个新模块时需要更改的代码
1. 在ultralytics/nn/models/common_utils.py等中添加新模块，例如名称为`MyModule`
2. 在ultralytics/nn/models/__init__.py中添加`MyModule`
3. 在ultralytics/nn/tasks.py的parse_model函数中中添加`MyModule`,以及相应的输入输出通道规则

## 相较于原始yolov12，针对双模态输入作出更改的代码部分：
1. (train bug)在ultralytics/models/yolo/detect/train.py中的plot_training_samples()函数中，添加了对红外图的显示
2. 在ultralytics/data/base.py的load_image()函数中，添加了对红外图像的读取
3. 修改default.yaml中的ch参数，以及mydata.yaml中的路径，以适应双模态输入
4. 在ultralytics/data/augment.py的类RandomHSV中，对融合后的6通道图像进行拆开，分别对RGB和红外图像进行HSV变换，再合并;以及LetterBox类中边框添加
5. (val bug)在ultralytics/engine/validator.py的类BaseValidator中的157行，将model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))的3改为self.args.ch，以适应双模态输入
6. 在ultralytics/models/yolo/detect/val.py的类DetectionValidator中的plot_val_samples()和plot_predictions()函数中添加对红外图像的结果绘制
7. (pre bug)在ultralytics/engine/predictor.py的类BasePredictor中的234行，将self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))的3改为self.args.ch，以适应双模态输入
8. 在ultralytics/engine/predictor.py的类BasePredictor中的265行，加入对可见光和红外图像结果的返回
9. 在ultralytics/engine/predictor.py的类BasePredictor中的286行，加入对可见光和红外图像推理结果图像的保存
10. 修改ultralytics/data/loaders.py的类LoadImagesAndVideos类中的__next__()函数，将红外图像读取并拼接到RGB图像上
11. 在ultralytics/models/yolo/detect/predict.py的类DetectionPredictor中，修改postprocess函数，将结果框画分别画在两个模态的图上



