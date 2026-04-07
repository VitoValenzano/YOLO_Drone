import cv2
import json
import numpy as np

VIDEO_PATH = "seg10.MP4"
OUTPUT_FILE = "zones10.json"
DISPLAY_W = 1280
DISPLAY_H = 720

zones = {}
current_points = []
current_label = None

instructions = """
Controls:
Click   : add polygon vertex
ENTER   : finish current polygon
N,S,E,W : ENTRY zones
n,s,e,w : EXIT zones
U       : undo last point
Q       : quit & save
"""

def mouse_callback(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

if not ret:
    print("Failed to load video.")
    exit()

# Resize first frame to fixed resolution
frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

cv2.namedWindow("Calibrate Zones")
cv2.setMouseCallback("Calibrate Zones", mouse_callback)

print(instructions)

while True:
    display = frame.copy()

    for label, poly in zones.items():
        cv2.polylines(display, [np.array(poly, dtype=np.int32)], True, (0,255,0), 2)
        cx = sum(p[0] for p in poly) // len(poly)
        cy = sum(p[1] for p in poly) // len(poly)
        cv2.putText(display, label, (cx,cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)


    for p in current_points:
        cv2.circle(display, p, 4, (0,0,255), -1)

    y0 = 20
    for line in instructions.split("\n"):
        cv2.putText(display, line, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y0 += 20

    cv2.imshow("Calibrate Zones", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('u'):
        if current_points:
            current_points.pop()

    elif key in [ord('n'),ord('s'),ord('e'),ord('w'),
                 ord('N'),ord('S'),ord('E'),ord('W')]:

        zone_type = "entry" if chr(key).isupper() else "exit"
        direction = chr(key).upper()

        current_label = f"{zone_type}_{direction}"
        current_points.clear()
        print(f"Drawing {current_label}...")

    elif key == 13 and current_label:  # ENTER
        if len(current_points) >= 3:
            zones[current_label] = current_points.copy()
            print(f"Saved {current_label}")
            current_points.clear()
            current_label = None

cap.release()
cv2.destroyAllWindows()

with open(OUTPUT_FILE, "w") as f:
    json.dump(zones, f, indent=2)

print("Zones saved to zones.json")
