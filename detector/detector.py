from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8s.pt", conf_thres=0.5):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]

        boxes = []

        if results.boxes is None:
            return boxes

        for box in results.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if cls != 0:
                continue

            if conf < self.conf_thres:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([x1, y1, x2, y2])

        return boxes