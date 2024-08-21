import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

video_name = 'datasets/videos_raw/00005.MTS'

cv2.namedWindow('imagem', cv2.WINDOW_NORMAL)
cv2.resizeWindow('imagem', (1920, 1080))
cap = cv2.VideoCapture(video_name)

video_height = 1080
video_width = 1440


points_crop = np.loadtxt("crop_points.txt")
points_crop = np.array(points_crop[:4], dtype=np.float32)
points_new = np.array([[0, 0], [video_width, 0], [video_width, video_height], [0, video_height]], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(points_crop, points_new)
x_min, y_min = np.min(points_crop, axis=0)
x_max, y_max = np.max(points_crop, axis=0)

ret, frame = cap.read()

model = YOLO("models/last.pt")
track_history = defaultdict(lambda: [])

while ret:
    if cv2.waitKey(1) == ord('q') or not ret:
        break
    ret, frame = cap.read()
    cropped_frame = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
    # result = cv2.warpPerspective(cropped_frame, matrix, (1920, 1080))
    results = model.track(cropped_frame, conf=0.000001, persist=True)
    if results[0].boxes is None or results[0].boxes.id is None:
        cropped_frame = cv2.resize(cropped_frame, (1280, 720))
        cv2.imshow('imagem', cropped_frame)
    else:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 100:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 0, 0), thickness=3)
        annotated_frame = cv2.resize(annotated_frame, (1280, 720))
        cv2.imshow('imagem', annotated_frame)

cap.release()
cv2.destroyAllWindows()