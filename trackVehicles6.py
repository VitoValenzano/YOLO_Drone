from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import csv
import json

# ----------------------------
# Configuration
# ----------------------------

MODEL_PATH = "yolov8s.pt"
NUM_SEGMENTS = 12
FPS = 30  # video FPS

ZONE_FLASH_TIME = 0.4
MIN_TRACK_LENGTH = 10

zone_flash_times = {}

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


def draw_zones(frame, ENTRY_ZONES, EXIT_ZONES):
    overlay = frame.copy()

    for name, poly in ENTRY_ZONES.items():
        poly_np = np.array(poly, dtype=np.int32)
        color = (255, 0, 0)

        cv2.polylines(frame, [poly_np], True, color, 2)

        cx, cy = poly_np.mean(axis=0).astype(int)
        cv2.putText(
            frame,
            f"IN {name}",
            (cx - 25, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    for name, poly in EXIT_ZONES.items():
        poly_np = np.array(poly, dtype=np.int32)
        color = (0, 0, 255)

        cv2.polylines(frame, [poly_np], True, color, 2)

        cx, cy = poly_np.mean(axis=0).astype(int)
        cv2.putText(
            frame,
            f"OUT {name}",
            (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


# ----------------------------
# Initialization
# ----------------------------

model = YOLO(MODEL_PATH)

movement_counts = defaultdict(int)
global_frame_count = 0   # continuous across all segments

log_file = open("vehicle_log.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["timestamp", "segment", "track_id", "entry", "exit", "movement_id"])


# ----------------------------
# Segment Processing Loop
# ----------------------------

for segment_idx in range(1, NUM_SEGMENTS + 1):

    video_path = f"seg{segment_idx}.mp4"
    zones_path = f"zones{segment_idx}.json"

    print(f"\n▶ Processing {video_path} with {zones_path}")

    with open(zones_path) as f:
        zone_data = json.load(f)

    ENTRY_ZONES = {
        k.split("_")[1]: v
        for k, v in zone_data.items()
        if k.startswith("entry")
    }

    EXIT_ZONES = {
        k.split("_")[1]: v
        for k, v in zone_data.items()
        if k.startswith("exit")
    }

    cap = cv2.VideoCapture(video_path)

    vehicle_paths = {}
    counted_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        global_frame_count += 1
        frame = cv2.resize(frame, (1280, 720))

        results = model.track(
            frame,
            persist=True,
            classes=[2, 3, 5, 7],
            tracker="bytetrack.yaml",
            imgsz=1280,
            verbose=False
        )

        annotated = frame.copy()

        if results[0].boxes is not None:

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

                # ----------------------------
                # ENTRY detection
                # ----------------------------
                if vehicle_paths[track_id]["entry"] is None:
                    for k, poly in ENTRY_ZONES.items():
                        if inside(poly, (cx, cy)):
                            vehicle_paths[track_id]["entry"] = k
                            break

                # ----------------------------
                # EXIT detection
                # ----------------------------
                if (
                    vehicle_paths[track_id]["entry"]
                    and vehicle_paths[track_id]["exit"] is None
                ):
                    for k, poly in EXIT_ZONES.items():
                        if inside(poly, (cx, cy)):
                            vehicle_paths[track_id]["exit"] = k
                            break

                # ----------------------------
                # Visualization
                # ----------------------------
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )

                cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

                path = vehicle_paths[track_id]

                # ----------------------------
                # Movement classification
                # ----------------------------
                if (
                    path["entry"]
                    and path["exit"]
                    and track_id not in counted_ids
                    and path["frames"] > MIN_TRACK_LENGTH
                ):

                    movement = MOVEMENT_MAP.get(
                        (path["entry"], path["exit"])
                    )

                    if movement is not None:
                        counted_ids.add(track_id)
                        movement_counts[movement] += 1

                        # TRUE VIDEO TIME
                        timestamp = global_frame_count / FPS

                        writer.writerow([
                            round(timestamp, 3),
                            segment_idx,
                            track_id,
                            path["entry"],
                            path["exit"],
                            movement
                        ])

        draw_zones(annotated, ENTRY_ZONES, EXIT_ZONES)

        # ----------------------------
        # Display movement totals
        # ----------------------------
        y = 30
        for m in sorted(movement_counts):
            cv2.putText(
                annotated,
                f"{m}: {movement_counts[m]}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            y += 22

        cv2.imshow("Intersection Traffic Analyzer", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            log_file.close()
            cv2.destroyAllWindows()
            quit()

    cap.release()

log_file.close()
cv2.destroyAllWindows()