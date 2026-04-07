from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time
import csv
import json


# ----------------------------
# Configuration
# ----------------------------

VIDEO_PATH = "DJI_20260203115126_0001_D.MP4"
MODEL_PATH = "yolov8s.pt"

MIN_TRACK_LENGTH = 10   # frames before considering valid movement

# ----------------------------
# Entry / Exit Zones 
# ----------------------------


with open("zones.json") as f:
    zone_data = json.load(f)

ENTRY_ZONES = { k.split("_")[1]: v for k,v in zone_data.items() if k.startswith("entry") }
EXIT_ZONES  = { k.split("_")[1]: v for k,v in zone_data.items() if k.startswith("exit") }


# ----------------------------
# Movement Mapping (Your Diagram)
# ----------------------------

MOVEMENT_MAP = {
    ("S","E"):10, ("S","N"):8, ("S","W"):3,
    ("W","S"):11, ("W","E"):2, ("W","N"):5,
    ("N","W"):12, ("N","S"):4, ("N","E"):7,
    ("E","N"):9, ("E","W"):6, ("E","S"):1
}

# ----------------------------
# Helper Functions
# ----------------------------

def inside(poly, point):
    return cv2.pointPolygonTest(np.array(poly, np.int32), point, False) >= 0

# ----------------------------
# Initialization
# ----------------------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

track_history = defaultdict(list)
vehicle_paths = {}
counted_ids = set()
movement_counts = defaultdict(int)

log_file = open("vehicle_log.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["timestamp", "track_id", "entry", "exit", "movement_id"])

# ----------------------------
# Main Processing Loop
# ----------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280,720))


    results = model.track(
        frame,
        persist=True,
        classes=[2,3,5,7],  # vehicles only
        tracker="bytetrack.yaml",
        imgsz=1280,
        verbose=False
    )

    annotated = results[0].plot()

    # Draw zones
    for poly in ENTRY_ZONES.values():
        cv2.polylines(annotated, [np.array(poly)], True, (255,0,0), 2)
    for poly in EXIT_ZONES.values():
        cv2.polylines(annotated, [np.array(poly)], True, (0,0,255), 2)

    for box in results[0].boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])
        x1, y1, x2, y2 = box.xyxy[0]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        track_history[track_id].append((cx, cy))

        if track_id not in vehicle_paths:
            vehicle_paths[track_id] = {"entry":None, "exit":None}

        # Entry detection
        if vehicle_paths[track_id]["entry"] is None:
            for k, poly in ENTRY_ZONES.items():
                if inside(poly, (cx,cy)):
                    vehicle_paths[track_id]["entry"] = k

        # Exit detection
        if vehicle_paths[track_id]["entry"] and vehicle_paths[track_id]["exit"] is None:
            for k, poly in EXIT_ZONES.items():
                if inside(poly, (cx,cy)):
                    vehicle_paths[track_id]["exit"] = k

        # Draw trail
        for i in range(1, len(track_history[track_id])):
            cv2.line(
                annotated,
                track_history[track_id][i-1],
                track_history[track_id][i],
                (0,255,0), 2
            )

        # Final movement classification
        path = vehicle_paths[track_id]
        if (path["entry"] and path["exit"] and
            track_id not in counted_ids and
            len(track_history[track_id]) > MIN_TRACK_LENGTH):

            movement = MOVEMENT_MAP.get((path["entry"], path["exit"]))

            if movement is not None:
                counted_ids.add(track_id)
                movement_counts[movement] += 1
                timestamp = time.time()

                writer.writerow([timestamp, track_id, path["entry"], path["exit"], movement])

    # Display counts
    y = 30
    for m in sorted(movement_counts):
        cv2.putText(
            annotated,
            f"{m}: {movement_counts[m]}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,255),
            2
        )
        y += 22

    display = cv2.resize(annotated, (1280,720))
    cv2.imshow("Intersection Traffic Analyzer", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
