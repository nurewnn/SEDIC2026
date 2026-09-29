import csv, json, uuid
from datetime import datetime
from pathlib import Path

FIELDS = [
    "session_id", "timestamp", "frame_id", "vessel_id",
    "class_name", "threat_level", "confidence",
    "x1", "y1", "x2", "y2", "detection_duration",
]

class DetectionLogger:
    """Writes a high-fidelity CSV detection log with session and tracking data.

    Fields:
        session_id          -- unique per DetectionLogger instance (uuid4 hex)
        timestamp           -- UTC ISO-8601 with milliseconds
        frame_id            -- frame index (0 for single images)
        vessel_id           -- object-tracking ID (int) or sequential index
        class_name          -- detected vessel class
        threat_level        -- CIVILIAN / SMALL CRAFT / MONITOR / PRIORITY / HIGH PRIORITY
        confidence          -- model confidence 0.000-1.000
        x1, y1, x2, y2      -- bounding-box pixel coordinates
        detection_duration  -- seconds this vessel_id was continuously observed
    """

    def __init__(self, out_path: str = "outputs/detection_log.csv",
                 session_id: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.path = Path(out_path)
        self.path.parent.mkdir(exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        # Track first-seen frame per vessel_id for duration calculation
        self._first_seen: dict = {}
        self._fps = 1.0  # default; overridden by set_fps()

    def set_fps(self, fps: float):
        """Set the video FPS for detection_duration calculation."""
        self._fps = fps if fps and fps > 0 else 1.0

    def log(self, frame_id: int, detections: list,
            fps: float | None = None):
        """Log detections for a single frame.

        Each detection dict may include:
            class_name, confidence, bbox, threat_level,
            vessel_id (from tracker) -- defaults to sequential index
        """
        if fps is not None:
            self.set_fps(fps)

        ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

        for idx, d in enumerate(detections):
            x1, y1, x2, y2 = d["bbox"]
            vessel_id = d.get("vessel_id", idx)

            # Track first-seen frame for duration
            vid_key = vessel_id
            if vid_key not in self._first_seen:
                self._first_seen[vid_key] = frame_id
            duration_frames = frame_id - self._first_seen[vid_key]
            # Convert frame difference to seconds (0 for single-frame / image)
            detection_duration = round(duration_frames / self._fps, 3) if self._fps else 0

            self._writer.writerow({
                "session_id":          self.session_id,
                "timestamp":            ts,
                "frame_id":            frame_id,
                "vessel_id":           vessel_id,
                "class_name":          d["class_name"],
                "threat_level":        d["threat_level"],
                "confidence":          d["confidence"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "detection_duration":  detection_duration,
            })

    def close(self):
        self._file.close()

    def to_json(self) -> str:
        rows = []
        with open(self.path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return json.dumps(rows, indent=2)