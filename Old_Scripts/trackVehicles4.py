from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time
import csv
import json


start_time = time.time()

# ----------------------------
# Configuration
# ----------------------------

VIDEO_PATH = "DJI_20260203115126_0001_D.MP4"
MODEL_PATH = "yolov8s.pt"

ZONE_FLASH_TIME = 0.4
zone_flash_times = {}

MIN_TRACK_LENGTH = 10   # frames before considering valid movement


# ----------------------------
# Load Zones
# ----------------------------

with open("zones.json") as f:
    zone_data = json.load(f)

ENTRY_ZONES = { k.split("_")[1]: v for k,v in zone_data.items() if k.startswith("entry") }
EXIT_ZONES  = { k.split("_")[1]: v for k,v in zone_data.items() if k.startswith("exit") }


# ----------------------------
# Movement Mapping
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


def draw_zones(frame):
    now = time.time()
    overlay = frame.copy()

    for name, poly in ENTRY_ZONES.items():
        poly_np = np.array(poly, dtype=np.int32)
        color = (255,0,0)

        if name in zone_flash_times and now - zone_flash_times[name] < ZONE_FLASH_TIME:
            cv2.fillPoly(overlay, [poly_np], color)
            cv2.polylines(frame, [poly_np], True, (255,255,255), 3)
        else:
            cv2.polylines(frame, [poly_np], True, color, 2)

        cx, cy = poly_np.mean(axis=0).astype(int)
        cv2.putText(frame, f"IN {name}", (cx-25, cy),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for name, poly in EXIT_ZONES.items():
        poly_np = np.array(poly, dtype=np.int32)
        color = (0,0,255)

        if name in zone_flash_times and now - zone_flash_times[name] < ZONE_FLASH_TIME:
            cv2.fillPoly(overlay, [poly_np], color)
            cv2.polylines(frame, [poly_np], True, (255,255,255), 3)
        else:
            cv2.polylines(frame, [poly_np], True, color, 2)

        cx, cy = poly_np.mean(axis=0).astype(int)
        cv2.putText(frame, f"OUT {name}", (cx-30, cy),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


# ----------------------------
# Initialization
# ----------------------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

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
        classes=[2,3,5,7],
        tracker="bytetrack.yaml",
        imgsz=1280,
        verbose=False
    )

    annotated = frame.copy()

    if results[0].boxes is None:
        draw_zones(annotated)
        cv2.imshow("Intersection Traffic Analyzer", annotated)
        continue

    for box in results[0].boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if track_id not in vehicle_paths:
            vehicle_paths[track_id] = {
                "entry": None,
                "exit": None,
                "frames": 0
            }

        vehicle_paths[track_id]["frames"] += 1

        # ENTRY detection
        if vehicle_paths[track_id]["entry"] is None:
            for k, poly in ENTRY_ZONES.items():
                if inside(poly, (cx,cy)):
                    vehicle_paths[track_id]["entry"] = k
                    zone_flash_times[k] = time.time()

        # EXIT detection
        if (vehicle_paths[track_id]["entry"] and
            vehicle_paths[track_id]["exit"] is None):
            for k, poly in EXIT_ZONES.items():
                if inside(poly, (cx,cy)):
                    vehicle_paths[track_id]["exit"] = k
                    zone_flash_times[k] = time.time()

        # Bounding box only (NO trails)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.circle(annotated, (cx,cy), 3, (0,0,255), -1)
        cv2.putText(annotated, f"ID {track_id}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Final movement classification
        path = vehicle_paths[track_id]
        if (path["entry"] and path["exit"] and
            track_id not in counted_ids and
            path["frames"] > MIN_TRACK_LENGTH):

            movement = MOVEMENT_MAP.get((path["entry"], path["exit"]))

            if movement is not None:
                counted_ids.add(track_id)
                movement_counts[movement] += 1

                timestamp = time.time() - start_time

                writer.writerow([
                    round(timestamp, 3),
                    track_id,
                    path["entry"],
                    path["exit"],
                    movement
                ])

    draw_zones(annotated)

    # Display movement counts
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

    cv2.imshow("Intersection Traffic Analyzer", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
