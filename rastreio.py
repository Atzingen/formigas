import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict

# ajuste o caminho do vídeo como necessário.
video_name = 'datasets/videos_raw/ants.mp4'

cv2.namedWindow('imagem', cv2.WINDOW_NORMAL)
cv2.resizeWindow('imagem', (1280, 720))
cap = cv2.VideoCapture(video_name)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(f'output_{video_name}', fourcc, 30.0, (1280, 720))

model = YOLO("models/best.pt")
track_history = defaultdict(lambda: [])
ret, frame = cap.read()

while ret:
    results = model.track(frame, conf=0.001, persist=True) # conf baixa pq o modelo é ruim.
    if results[0].boxes is None or results[0].boxes.id is None:
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('imagem', frame)
        output.write(frame)
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
        output.write(annotated_frame)
    if cv2.waitKey(1) == ord('q') or not ret:
        break
    ret, frame = cap.read()
        
cap.release()
cv2.destroyAllWindows()
output.release()