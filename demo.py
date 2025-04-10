import cv2
from yolo9 import YOLO9, ModelWeight
import numpy as np

if __name__ == "__main__":
    yolo = YOLO9(
        model=ModelWeight.YOLO9_M_CARPLATE,
        device="cpu",
        iou_threshold=0.45,
        max_det=1000,
        classes={0: 0.1, 1: 0.1, 2: 0.1}
    )

    img = cv2.imread("carplates.webp")
    detections = yolo.detect(img)

    for polygon, confidence, class_id, class_name in detections:
        # Convert polygon points to integer coordinates for drawing
        pts = []
        textx, texty =  -1, -1
        for x, y in polygon:
            x_, y_ = int(x * img.shape[1]), int(y * img.shape[0])
            pts.append((x_, y_))
            if textx == -1 or x_ < textx:
                textx = x_
            if texty == -1 or y_ < texty:
                texty = y_
        texty = 0 if texty < 10 else texty - 10
        
        cv2.polylines(
            img,
            pts=[np.array(pts)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        # Add class id and confidence text
        text = f"{class_name} {confidence:.2f}"
        cv2.putText(img, text, (textx, texty), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image
    cv2.imwrite("output.jpg", img)
