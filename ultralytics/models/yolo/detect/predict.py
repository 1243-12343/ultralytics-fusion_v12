# Ultralytics YOLO 🚀, AGPL-3.0 license
import numpy as np

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            rmopp=getattr(self.args, "rmopp", False),
            rmopp_gamma1=getattr(self.args, "rmopp_gamma1", 1.0),
            rmopp_uncertainty_weight=getattr(self.args, "rmopp_uncertainty_weight", 0.5),
            rmopp_mode=getattr(self.args, "rmopp_mode", "soft"),
            rmopp_max_penalty=getattr(self.args, "rmopp_max_penalty", 0.25),
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        ir_result = []

        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # 先推理可见光
            vis_orig_img = orig_img[..., 3:]    # 此时传入进来的im0的前三通道是红外，后三通道是可见光
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], vis_orig_img.shape)
            results.append(Results(vis_orig_img, path=img_path, names=self.model.names, boxes=pred))
            # 再推理红外
            ir_path = img_path.split('images')
            ir_path = str(ir_path[0] + 'image' + ir_path[1])
            if orig_img.shape[-1] >= 4:
                ir_img = orig_img[..., :3]
                ir_result.append(Results(ir_img, path=ir_path, names=self.model.names, boxes=pred))
        return results, ir_result
