import cv2
import numpy as np

polygon_points = []
selected_frame = None

def click_event(event, x, y, flags, param):
    global polygon_points, selected_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        cv2.circle(selected_frame, (x, y), 3, (0, 255, 0), -1)
        if len(polygon_points) > 1:
            cv2.line(selected_frame, polygon_points[-2], polygon_points[-1], (255, 0, 0), 2)

        cv2.imshow("Select Points", selected_frame)

# Load the video
video_path = "datasets/videos_raw/00005.MTS"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, selected_frame = cap.read()
if not ret:
    print("Error: Could not read frame from video.")
    cap.release()
    exit()

cv2.namedWindow("Select Points")
cv2.setMouseCallback("Select Points", click_event)

cv2.imshow("Select Points", selected_frame)
cv2.waitKey(0)

# Proceed if the user has selected at least 3 points
if len(polygon_points) > 2:
    mask = np.zeros_like(selected_frame, dtype=np.uint8)
    pts = np.array(polygon_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = cv2.bitwise_and(frame, mask)
        cv2.imshow("Processed Video", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    np.savetxt("crop_points.txt", polygon_points)
else:
    print("Pelo menos 3 pontos para um polígono")

cap.release()
cv2.destroyAllWindows()