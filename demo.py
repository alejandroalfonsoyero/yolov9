import cv2
from yolo9 import YOLO9, CocoModels
import numpy as np

if __name__ == "__main__":
    yolo = YOLO9(
        model=CocoModels.YOLO9_L,
        device="cpu",
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        classes={0: 0.5, 17: 0.5}
    )

    img = cv2.imread("man_and_horse.webp")
    detections = yolo.detect(img)

    for detection in detections:
        polygon, confidence, class_id = detection
        # Convert polygon points to integer coordinates for drawing
        pts = []
        for x, y in polygon:
            pts.append((int(x * img.shape[1]), int(y * img.shape[0])))
        
        cv2.polylines(
            img,
            pts=[np.array(pts)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=1,
        )

        # Add class id and confidence text
        text = f"class: {class_id} conf: {confidence:.2f}"
        cv2.putText(img, text, (pts[0][0], pts[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image
    cv2.imwrite("output.jpg", img)
