from ultralytics import YOLO
import cv2
import math 
import numpy as np
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 540)

# model
model = YOLO("best (2).pt")

# object classes
classNames = ["kertas","tembaga" ]

# List of colors for each class
class_colors = np.random.randint(0, 256, size=(len(classNames), 3), dtype=np.uint8)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for i, r in enumerate(results):
        boxes = r.boxes

        for j, box in enumerate(boxes):
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # put box in cam
            color = tuple(map(int, class_colors[j % len(class_colors)]))  # Use a random color for each class
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = (x1, y1 - 10)  # Adjust the position for displaying coordinates above the box
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2

            # Display class name
            cv2.putText(img, classNames[cls], org, font, fontScale, (255, 255, 255), thickness)

            # Display coordinates
            coordinates_text = f"({x1}, {y1}) - ({x2}, {y2})"
            cv2.putText(img, coordinates_text, (x1, y1 - 30), font, fontScale, (255, 255, 255), thickness)

            # Display confidence
            confidence_text = f"Confidence: {confidence}"
            cv2.putText(img, confidence_text, (x1, y1 - 50), font, fontScale, (255, 255, 255), thickness)

            # Draw crosshair at the center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.drawMarker(img, (center_x, center_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            # circle to mark the area
            cv2.circle(img, (center_x, center_y), 100, color, thickness=2) 

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()