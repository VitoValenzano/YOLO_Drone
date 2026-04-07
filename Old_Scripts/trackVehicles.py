from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load model
model = YOLO("yolov8s.pt")

# Load video
cap = cv2.VideoCapture("DJI_20260203115126_0001_D.MP4")

# Store trajectories
track_history = defaultdict(list)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking
    results = model.track(
        frame,
        persist=True,
        classes=[2,3,5,7],  # vehicles only
        tracker="bytetrack.yaml",
        imgsz=1280
    )

    annotated = results[0].plot()

    # Extract trajectories
    for box in results[0].boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])
        x1, y1, x2, y2 = box.xyxy[0]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        track_history[track_id].append((cx, cy))

        # Draw trajectory trail
        for i in range(1, len(track_history[track_id])):
            cv2.line(
                annotated,
                track_history[track_id][i-1],
                track_history[track_id][i],
                (0,255,0), 2
            )

    display = cv2.resize(annotated, (1280, 720)) # For changing window size
    cv2.imshow("Vehicle Tracking", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
