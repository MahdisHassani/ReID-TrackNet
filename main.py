from detector.detector import YOLODetector
from tracker.tracker import Tracker
from utils.video import load_video
from utils.visualization import draw_boxes
from utils.metrics import MOTMetrics

import cv2

video_path = "data/video.avi"

cap = load_video(video_path)

detector = YOLODetector()
tracker = Tracker()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

metrics = MOTMetrics()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame)

    metrics.update(detections, tracks)

    frame = draw_boxes(frame, tracks)
    out.write(frame)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

metrics.compute()
cap.release()
out.release()
cv2.destroyAllWindows()