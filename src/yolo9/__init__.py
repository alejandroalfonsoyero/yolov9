from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import cv2
from yolo9.models.common import DetectMultiBackend
from yolo9.utils.general import LOGGER, non_max_suppression
from yolo9.utils.torch_utils import select_device


class CocoModels(Enum):
    YOLO9_C = "yolov9-c"
    YOLO9_E = "yolov9-e"
    YOLO9_M = "yolov9-m"
    YOLO9_S = "yolov9-s"


class YOLO9:
    def __init__(
        self,
        model: CocoModels,
        device: str,
        classes: Dict[int, float],  # class id -> confidence threshold
        dnn: bool = False,
        half: bool = False,
        batch_size: int = 1,  # batch size
        iou_threshold: float = 0.45,
        max_det: int = 1000,
    ):
        weights_dir = Path(__file__).parent / 'weights'
        weights_dir.mkdir(exist_ok=True)
        data = Path(__file__).parent / 'data' / 'coco.yaml'

        self.weights_path = weights_dir / f'{model.value}.pt'
        self.device = select_device(device)
        self.conf_thres = min(classes.values()) if classes else 0.25
        self.iou_thres = iou_threshold
        self.max_det = max_det
        self.classes = classes
        if not self.weights_path.exists():
            LOGGER.info(f"Downloading {self.weights_path.name} from GitHub releases...")
            import requests
            url = f"https://github.com/alejandroalfonsoyero/yolov9/releases/download/v1.0.2/{model.value}.pt"
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(self.weights_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            LOGGER.info(f"Downloaded {self.weights_path.name} successfully")

        self.model = DetectMultiBackend(self.weights_path, device=self.device, dnn=dnn, data=data, fp16=half)
        self.img_size = 640
        self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else batch_size, 3, self.img_size, self.img_size))

    def detect(self, image: np.ndarray) -> list:
        im = cv2.resize(image, (self.img_size, self.img_size))
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0-255 to 0-1
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        im = im.permute(0, 3, 1, 2)  # BHWC to BCHW

        pred = self.model(im)
        pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, max_det=self.max_det)[0]

        detections = []
        for det in pred:
            confidence, class_id = det[4].tolist(), int(det[5].tolist())
            if class_id not in self.classes:
                continue
            if confidence < self.classes[class_id]:
                continue
            x1, y1, x2, y2 = (det[0:4] / self.img_size).tolist()
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            detections.append((polygon, confidence, class_id))

        return detections
