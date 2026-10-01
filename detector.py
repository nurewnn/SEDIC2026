import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ULTRALYTICS_CONFIG_DIR = PROJECT_DIR / "Ultralytics"
MPL_CONFIG_DIR = PROJECT_DIR / ".matplotlib"
ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
MPL_CONFIG_DIR.mkdir(exist_ok=True)

os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

from ultralytics import YOLO
from utils.colours import THREAT_COLOURS, THREAT_LEVEL
import cv2, numpy as np

class GuardianDetector:
    def __init__(self, model_path: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame: np.ndarray, conf: float | None = None) -> list[dict]:
        """
        Returns a list of detection dicts for one frame.
        Each dict: {class_name, confidence, bbox, colour, threat_level, vessel_id}
        iou=0.5 removes overlapping duplicate boxes on the same vessel.
        vessel_id is a sequential index (no tracking) for single-frame calls.
        """
        results = self.model.predict(
            frame,
            conf=self.conf if conf is None else conf,
            iou=0.5,        # suppress double detections on same vessel
            imgsz=1280,
            verbose=False,
        )[0]

        detections = []
        for idx, box in enumerate(results.boxes):
            cls_id   = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_name":   cls_name,
                "confidence":   round(float(box.conf[0]), 3),
                "bbox":         (x1, y1, x2, y2),
                "colour":       THREAT_COLOURS.get(cls_name, (180, 180, 180)),
                "threat_level": THREAT_LEVEL.get(cls_name, "UNKNOWN"),
                "vessel_id":    cls_id,   # fallback to class_id so Gantt groups by class
            })
        return detections

    def track(self, frame: np.ndarray, conf: float | None = None) -> list[dict]:
        """
        Returns a list of detection dicts WITH persistent object tracking.
        Uses Ultralytics built-in ByteTrack. If it fails, falls back to predict().
        """
        try:
            results = self.model.track(
                frame,
                conf=self.conf if conf is None else conf,
                iou=0.5,
                persist=True,
                imgsz=1280,
                verbose=False,
            )[0]
            
            detections = []
            if results.boxes is None or results.boxes.id is None:
                return detections

            for box in results.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                detections.append({
                    "class_name":   cls_name,
                    "confidence":   round(float(box.conf[0]), 3),
                    "bbox":         (x1, y1, x2, y2),
                    "colour":       THREAT_COLOURS.get(cls_name, (180, 180, 180)),
                    "threat_level": THREAT_LEVEL.get(cls_name, "UNKNOWN"),
                    "vessel_id":    track_id,
                })
            return detections
        except Exception:
            return self.predict(frame, conf)

    def reset_tracker(self):
        """Reset the tracking state for a new session."""
        self.model.predictor = None

    def diagnostics(self, frame: np.ndarray, conf: float = 0.01) -> dict:
        """Returns low-threshold model output for troubleshooting."""
        detections = self.predict(frame, conf=conf)
        return {
            "probe_confidence": conf,
            "image_shape":      tuple(frame.shape),
            "model_classes":    self.model.names,
            "candidate_count":  len(detections),
            "candidates":       detections[:20],
        }

    def annotate(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draws bounding boxes + labels onto a copy of the frame."""
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            colour = d["colour"]
            label  = f"{d['class_name']}  {d['confidence']:.0%}"
            threat = d["threat_level"]

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            # Label background pill
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Threat badge (top-right of box)
            cv2.putText(out, threat, (x2 - 110, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        return out