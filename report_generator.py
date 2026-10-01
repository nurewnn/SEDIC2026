"""
ReportGenerator - Intelligence Pipeline for SEDIC 2026 Grand Finale.

Reads a detection CSV (produced by DetectionLogger), performs statistical
analysis with Pandas, generates charts with Matplotlib, writes an executive
summary via Ollama (local LLM, with template fallback), and stitches
everything into a mission-ready PDF using FPDF.

The report is designed to be printed and handed to a commanding officer.
It is formal, concise, and fits on 2-5 pages.

Vessels (unique tracked objects) are always distinguished from detections
(per-frame observations) throughout the report.
"""

import os
import re
import json
import uuid
import threading
import concurrent.futures
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF

# -- Chart styling (white background for print) --------------------------------
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   "#333333",
    "xtick.color":       "#666666",
    "ytick.color":       "#666666",
    "text.color":        "#222222",
    "axes.titlecolor":   "#1a6b8a",
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "figure.dpi":        150,
    "font.family":       "sans-serif",
})

ACCENT      = "#1a6b8a"   # dark teal (printable)
ACCENT_SOFT = "#3a8db0"
GREEN       = "#2d9d5f"
RED         = "#cc3333"
ORANGE      = "#e88a3a"
YELLOW      = "#d4a017"

def _hex_to_rgb(h):
    """Convert hex color string '#rrggbb' to (r, g, b) tuple."""
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


THREAT_COLORS = {
    "HIGH PRIORITY": RED,
    "PRIORITY":      ORANGE,
    "MONITOR":       YELLOW,
    "SMALL CRAFT":   "#b8a020",
    "CIVILIAN":      GREEN,
}

# Import THREAT_LEVEL from utils/colours.py (do NOT modify that file)
try:
    from utils.colours import THREAT_LEVEL as _TL
    THREAT_LEVEL = dict(_TL)
except Exception:
    THREAT_LEVEL = {
        "container_ship": "CIVILIAN", "tanker": "CIVILIAN", "cargo": "CIVILIAN",
        "passenger_ferry": "CIVILIAN", "tugboat": "CIVILIAN",
        "yacht": "SMALL CRAFT", "speedboat": "SMALL CRAFT", "fishing_boat": "SMALL CRAFT",
        "patrol_boat": "MONITOR",
        "foreign_military_ship": "HIGH PRIORITY", "local_military_ship": "PRIORITY",
    }


class ReportGenerator:
    """Turns a detection CSV into a mission-ready PDF report."""

    def __init__(self, csv_path, session_label="", model_path="",
                 conf_thresh=0.25, iou_thresh=0.5, tracker_name="ByteTrack",
                 model_names=None, video_info=None, use_llm=True,
                 llm_model=None, frame_crops_dir=None,
                 input_type='video', annotated_image_path=None,
                 image_resolution=None, annotated_frames=None):
        self.csv_path = Path(csv_path)
        self.session_label = session_label or self.csv_path.stem
        self.model_path = model_path or ""
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.tracker_name = tracker_name or "ByteTrack"
        self.video_info = video_info or {}
        self.use_llm = use_llm
        self.frame_crops_dir = Path(frame_crops_dir) if frame_crops_dir else None
        self.input_type = input_type
        self.annotated_image_path = annotated_image_path
        self.image_resolution = image_resolution
        self.annotated_frames = annotated_frames or []  # list of (path, label) tuples

        # Resolve model_names
        if model_names is not None:
            self.model_names = dict(model_names)
        elif model_path:
            self.model_names = self._load_model_names(model_path)
        else:
            self.model_names = {}

        # Resolve llm_model
        if llm_model:
            self.llm_model = llm_model
        else:
            self.llm_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

        self.df = None
        self.stats = {}
        self.chart_paths = {}
        self.llm_analysis = None
        self._chart_dir = Path("outputs/charts")
        self._chart_dir.mkdir(parents=True, exist_ok=True)

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _load_model_names(model_path):
        """Try loading model.names from a YOLO model via ultralytics."""
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            return dict(model.names)
        except Exception:
            return {}

    @staticmethod
    def _sanitize(text):
        """Replace Unicode chars with ASCII, encode latin-1 replace."""
        if not text:
            return ""
        replacements = {
            "\u2014": "-", "\u2013": "-",
            "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"',
            "\u2026": "...", "\u00b0": " deg",
        }
        for uni, asc in replacements.items():
            text = text.replace(uni, asc)
        return text.encode("latin-1", "replace").decode("latin-1")

    # -- Data loading ----------------------------------------------------------

    def load_csv(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        return self.df

    # -- Statistics -------------------------------------------------------------

    def compute_stats(self):
        """Compute ALL numbers. Distinguish vessels from detections."""
        if self.df is None:
            self.load_csv()
        df = self.df
        s = {}
        s["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        s["model_weights_name"] = Path(self.model_path).name if self.model_path else "N/A"
        s["num_classes"] = len(self.model_names)
        s["llm_model"] = self.llm_model
        s["conf_threshold"] = self.conf_thresh
        s["iou_threshold"] = self.iou_thresh
        s["tracker_name"] = self.tracker_name
        s["input_type"] = self.input_type
        if self.image_resolution:
            w_res, h_res = self.image_resolution
            s["image_resolution"] = f"{w_res}x{h_res}"
        else:
            s["image_resolution"] = None
        vi = self.video_info
        if vi:
            s["video_fps"] = vi.get("fps", 0)
            s["video_width"] = vi.get("width", 0)
            s["video_height"] = vi.get("height", 0)
            s["video_total_frames"] = vi.get("total_frames", 0)
            s["video_duration_s"] = vi.get("duration_s", 0)
        else:
            s["video_fps"] = 0
            s["video_width"] = 0
            s["video_height"] = 0
            s["video_total_frames"] = 0
            s["video_duration_s"] = 0

        if df is None or df.empty:
            s["total_detections"] = 0
            s["unique_vessels"] = 0
            s["frames_processed"] = 0
            s["session_id"] = "N/A"
            s["per_vessel"] = []
            s["per_class"] = []
            s["high_priority"] = 0
            s["priority"] = 0
            s["monitor"] = 0
            s["small_craft"] = 0
            s["civilian"] = 0
            s["threat_counts_vessel"] = {}
            s["threat_counts_detection"] = {}
            s["threat_one_liner"] = "0 threat contacts detected (0 HIGH PRIORITY, 0 PRIORITY)"
            s["class_list_model"] = list(self.model_names.values()) if self.model_names else []
            s["unmapped_classes"] = []
            s["missing_military_classes"] = []
            s["low_conf_classes"] = []
            s["short_tracking"] = False
            s["avg_confidence"] = 0
            s["median_confidence"] = 0
            s["min_confidence"] = 0
            s["max_confidence"] = 0
            s["avg_duration"] = 0
            s["max_duration"] = 0
            s["min_duration"] = 0
            s["input_type"] = self.input_type
            if self.image_resolution:
                w_res, h_res = self.image_resolution
                s["image_resolution"] = f"{w_res}x{h_res}"
            else:
                s["image_resolution"] = None
            self.stats = s
            return s

        # Basic counts -- vessels vs detections
        s["total_detections"] = len(df)
        s["unique_vessels"] = int(df["vessel_id"].nunique()) if "vessel_id" in df.columns else 0
        s["frames_processed"] = int(df["frame_id"].nunique()) if "frame_id" in df.columns else 0
        s["session_id"] = str(df["session_id"].iloc[0]) if "session_id" in df.columns else "N/A"

        # Overall confidence
        conf_col = df["confidence"] if "confidence" in df.columns else pd.Series([0.0])
        s["avg_confidence"] = round(float(conf_col.mean()), 3)
        s["median_confidence"] = round(float(conf_col.median()), 3)
        s["min_confidence"] = round(float(conf_col.min()), 3)
        s["max_confidence"] = round(float(conf_col.max()), 3)

        # Per-vessel stats
        per_vessel = []
        if "vessel_id" in df.columns:
            for vid in sorted(df["vessel_id"].unique()):
                vdf = df[df["vessel_id"] == vid]
                cls_name = str(vdf["class_name"].iloc[0]) if "class_name" in vdf.columns else "unknown"
                tl = str(vdf["threat_level"].iloc[0]) if "threat_level" in vdf.columns else THREAT_LEVEL.get(cls_name, "UNKNOWN")
                vc = vdf["confidence"] if "confidence" in vdf.columns else pd.Series([0.0])
                ff = int(vdf["frame_id"].min()) if "frame_id" in vdf.columns else 0
                lf = int(vdf["frame_id"].max()) if "frame_id" in vdf.columns else 0
                ft = str(vdf["timestamp"].iloc[0]) if "timestamp" in vdf.columns else "N/A"
                lt = str(vdf["timestamp"].iloc[-1]) if "timestamp" in vdf.columns else "N/A"
                dur = float(vdf["detection_duration"].max()) if "detection_duration" in vdf.columns else 0.0
                avg_c = round(float(vc.mean()), 3)
                min_c = round(float(vc.min()), 3)
                max_c = round(float(vc.max()), 3)
                med_c = round(float(vc.median()), 3)
                low_conf = (avg_c < 0.50) or (min_c < 0.40)
                per_vessel.append({
                    "vessel_id": int(vid),
                    "class_name": cls_name,
                    "threat_level": tl,
                    "detection_count": int(len(vdf)),
                    "first_frame": ff,
                    "last_frame": lf,
                    "first_timestamp": ft,
                    "last_timestamp": lt,
                    "duration_s": round(dur, 2),
                    "avg_conf": avg_c,
                    "min_conf": min_c,
                    "max_conf": max_c,
                    "median_conf": med_c,
                    "low_confidence": low_conf,
                })
        per_vessel.sort(key=lambda v: v["vessel_id"])
        s["per_vessel"] = per_vessel

        # Per-class stats
        per_class = []
        if "class_name" in df.columns:
            for cls in sorted(df["class_name"].unique()):
                cdf = df[df["class_name"] == cls]
                cc = cdf["confidence"] if "confidence" in cdf.columns else pd.Series([0.0])
                vcount = int(cdf["vessel_id"].nunique()) if "vessel_id" in cdf.columns else 0
                avg_c = round(float(cc.mean()), 3)
                min_c = round(float(cc.min()), 3)
                max_c = round(float(cc.max()), 3)
                low = (avg_c < 0.50) or (min_c < 0.40)
                per_class.append({
                    "class_name": str(cls),
                    "detection_count": int(len(cdf)),
                    "vessel_count": vcount,
                    "avg_conf": avg_c,
                    "min_conf": min_c,
                    "max_conf": max_c,
                    "low_confidence": low,
                })
        per_class.sort(key=lambda c: -c["detection_count"])
        s["per_class"] = per_class

        # Threat counts -- vessel-based and detection-based
        threat_vessel = {}
        threat_detection = {}
        if "threat_level" in df.columns:
            for tl in df["threat_level"].unique():
                tl_str = str(tl)
                vcount = df[df["threat_level"] == tl]["vessel_id"].nunique() if "vessel_id" in df.columns else 0
                threat_vessel[tl_str] = int(vcount)
                dcount = len(df[df["threat_level"] == tl])
                threat_detection[tl_str] = int(dcount)
        for tl_name in ["HIGH PRIORITY", "PRIORITY", "MONITOR", "SMALL CRAFT", "CIVILIAN"]:
            threat_vessel.setdefault(tl_name, 0)
            threat_detection.setdefault(tl_name, 0)
        s["threat_counts_vessel"] = threat_vessel
        s["threat_counts_detection"] = threat_detection

        s["high_priority"] = threat_vessel.get("HIGH PRIORITY", 0)
        s["priority"] = threat_vessel.get("PRIORITY", 0)
        s["monitor"] = threat_vessel.get("MONITOR", 0)
        s["small_craft"] = threat_vessel.get("SMALL CRAFT", 0)
        s["civilian"] = threat_vessel.get("CIVILIAN", 0)

        # Threat one-liner
        hp = s["high_priority"]
        pr = s["priority"]
        s["threat_one_liner"] = f"{hp + pr} threat contacts detected ({hp} HIGH PRIORITY, {pr} PRIORITY)"

        # Class list from model_names
        s["class_list_model"] = list(self.model_names.values()) if self.model_names else []

        # Unmapped classes -- in detection data but not in THREAT_LEVEL
        detected_classes = set(df["class_name"].unique()) if "class_name" in df.columns else set()
        s["unmapped_classes"] = [c for c in detected_classes if c not in THREAT_LEVEL]

        # Missing military classes -- not in model_names
        military = ["foreign_military_ship", "local_military_ship"]
        model_class_names = set(self.model_names.values()) if self.model_names else set()
        s["missing_military_classes"] = [m for m in military if m not in model_class_names]

        # Low-confidence classes list
        s["low_conf_classes"] = [c["class_name"] for c in per_class if c["low_confidence"]]

        # Short tracking note
        if per_vessel:
            avg_dur = float(sum(v["duration_s"] for v in per_vessel)) / len(per_vessel)
            s["avg_duration"] = round(avg_dur, 2)
            s["max_duration"] = round(max(v["duration_s"] for v in per_vessel), 2)
            s["min_duration"] = round(min(v["duration_s"] for v in per_vessel), 2)
            s["short_tracking"] = avg_dur < 2.0
        else:
            s["avg_duration"] = 0
            s["max_duration"] = 0
            s["min_duration"] = 0
            s["short_tracking"] = False

        self.stats = s
        return s

    # -- LLM analysis ----------------------------------------------------------

    def _stats_for_llm(self):
        """Build a JSON-serializable subset of stats for the LLM payload."""
        s = self.stats
        keep = {}
        for k, v in s.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                keep[k] = v
            elif isinstance(v, (list, dict)):
                try:
                    json.dumps(v)
                    keep[k] = v
                except (TypeError, ValueError):
                    pass
        return keep

    def _validate_llm_json(self, data):
        """Validate the LLM-returned JSON against the required schema."""
        if not isinstance(data, dict):
            return False
        for k in ["executive_summary", "threat_assessment", "limitations", "recommendations", "captions"]:
            if k not in data:
                return False
        if not isinstance(data["limitations"], list):
            return False
        if not isinstance(data["recommendations"], list):
            return False
        caps = data.get("captions", {})
        if not isinstance(caps, dict):
            return False
        for k in ["track_chart", "confidence_hist", "class_table", "annotated_frames"]:
            if k not in caps:
                return False
        return True

    def generate_llm_analysis(self, stats=None):
        """ONE Ollama call. JSON output. Temperature 0.2. Timeout 30s. keep_alive 5m."""
        if stats is None:
            stats = self.stats
        if not stats:
            return self.template_fallback(stats)
        if not self.use_llm:
            return self.template_fallback(stats)

        stats_json = json.dumps(self._stats_for_llm(), indent=2, default=str)
        prompt = (
            "You are a maritime domain awareness analyst. Based on the detection "
            "session statistics below, write a JSON analysis. Use ONLY the numbers "
            "in the JSON. Do not invent vessels, classes, locations, or facts. "
            "Always distinguish vessels (unique tracked objects) from detections "
            "(per-frame observations). If confidence is low, say verification is "
            "needed. Output valid JSON only.\n\n"
            f"Statistics:\n{stats_json}\n\n"
            'Output this exact JSON schema:\n'
            '{"executive_summary": "2-3 sentences", "threat_assessment": '
            '"short paragraph with caveats", "limitations": ["...", "..."], '
            '"recommendations": ["...", "..."], "captions": {"track_chart": '
            '"1-2 sentences", "confidence_hist": "1-2 sentences", "class_table": '
            '"1-2 sentences", "annotated_frames": "1-2 sentences"}}'
        )

        try:
            import httpx
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2},
                "keep_alive": "5m",
            }
            resp = httpx.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            if not self._validate_llm_json(parsed):
                return self.template_fallback(stats)
            parsed["executive_summary"] = self._sanitize(parsed.get("executive_summary", ""))
            parsed["threat_assessment"] = self._sanitize(parsed.get("threat_assessment", ""))
            parsed["limitations"] = [self._sanitize(str(x)) for x in parsed.get("limitations", [])]
            parsed["recommendations"] = [self._sanitize(str(x)) for x in parsed.get("recommendations", [])]
            caps = parsed.get("captions", {})
            for ck in ["track_chart", "confidence_hist", "class_table", "annotated_frames"]:
                caps[ck] = self._sanitize(str(caps.get(ck, "")))
            parsed["captions"] = caps
            return parsed
        except Exception:
            return self.template_fallback(stats)

    def template_fallback(self, stats=None):
        """Same schema as LLM. Built with f-strings from stats."""
        if stats is None:
            stats = self.stats
        s = stats
        total = s.get("total_detections", 0)
        unique = s.get("unique_vessels", 0)
        hp = s.get("high_priority", 0)
        pr = s.get("priority", 0)
        avg_c = s.get("avg_confidence", 0)
        frames = s.get("frames_processed", 0)
        monitor = s.get("monitor", 0)
        sc = s.get("small_craft", 0)
        civ = s.get("civilian", 0)
        is_image = (self.input_type == 'image')

        # Executive summary
        if is_image:
            if total == 0:
                exec_sum = "No vessel detections were recorded from the still image."
            elif total == 1:
                exec_sum = (
                    f"1 detection was recorded from a still image. "
                    f"Average model confidence was {avg_c:.1%}."
                )
            else:
                exec_sum = (
                    f"{total} detections were recorded from a still image. "
                    f"Average model confidence was {avg_c:.1%}."
                )
        elif total == 0:
            exec_sum = ("No vessel contacts were recorded during this session. "
                        "The system maintained operational readiness throughout.")
        else:
            exec_sum = (
                f"During this session, {total} detections were recorded across "
                f"{unique} unique vessels over {frames} processed frame(s). "
                f"Average model confidence was {avg_c:.1%}. "
            )
            if hp > 0:
                exec_sum += f"{hp} HIGH PRIORITY vessel(s) identified - immediate action required. "
            elif pr > 0:
                exec_sum += f"{pr} PRIORITY vessel(s) identified - monitor and report. "
            else:
                exec_sum += "No priority-level threat contacts detected. "
            exec_sum += "The system maintained operational readiness throughout."

        # Threat assessment
        if is_image:
            if hp == 0 and pr == 0:
                ta = ("No priority-level threat contacts detected. No military vessels were "
                      "identified in this image. Note: military recall is not guaranteed and "
                      "absence of detection does not confirm absence of military assets.")
            else:
                ta = (f"{hp} HIGH PRIORITY and {pr} PRIORITY vessel(s) identified. "
                      f"Threat assessment requires verification with raw detection data. "
                      f"Monitor: {monitor}, Small craft: {sc}, Civilian: {civ}. "
                      f"Note: military recall is not guaranteed.")
            if s.get("low_conf_classes"):
                ta += f" Low-confidence classes ({', '.join(s['low_conf_classes'])}) require manual review."
        elif hp == 0 and pr == 0:
            ta = ("No priority-level threat contacts detected. No military vessels were "
                  "identified in this session. Note: military recall is not guaranteed and "
                  "absence of detection does not confirm absence of military assets.")
        else:
            ta = (f"{hp} HIGH PRIORITY and {pr} PRIORITY vessel(s) identified. "
                  f"Threat assessment requires verification with raw detection data. "
                  f"Monitor: {monitor}, Small craft: {sc}, Civilian: {civ}. "
                  f"Note: military recall is not guaranteed.")
        if (not is_image) and s.get("low_conf_classes"):
            ta += f" Low-confidence classes ({', '.join(s['low_conf_classes'])}) require manual review."

        # Limitations
        limitations = []
        if is_image:
            limitations.append(
                "Still image: vessel identity and tracking cannot be verified."
            )
            limitations.append(
                "Cross-reference with AIS data recommended."
            )
            if s.get("low_conf_classes"):
                limitations.append(
                    f"Low-confidence detections for classes: {', '.join(s['low_conf_classes'])}. "
                    "Manual verification recommended."
                )
            source = (self.session_label or "").lower()
            if any(w in source for w in ["cinema", "studio", "synthetic", "demo", "sample"]):
                limitations.append("Source image may not be real maritime surveillance imagery.")
            if total > 0 and total < 50:
                limitations.append("Small sample size - statistics may not be representative.")
            if s.get("missing_military_classes"):
                limitations.append(
                    f"Missing military classes in model: {', '.join(s['missing_military_classes'])}. "
                    "Military detection capability may be limited."
                )
            if s.get("unmapped_classes"):
                limitations.append(
                    f"Unmapped classes (no threat level defined): {', '.join(s['unmapped_classes'])}."
                )
        else:
            if s.get("low_conf_classes"):
                limitations.append(
                    f"Low-confidence detections for classes: {', '.join(s['low_conf_classes'])}. "
                    "Manual verification recommended."
                )
            if s.get("short_tracking"):
                limitations.append(
                    f"Short average tracking duration ({s.get('avg_duration', 0):.1f}s). "
                    "Vessel tracks may be fragmented."
                )
            source = (self.session_label or "").lower()
            if any(w in source for w in ["cinema", "studio", "synthetic", "demo", "sample"]):
                limitations.append("Source footage may not be real maritime surveillance footage.")
            if total > 0 and total < 50:
                limitations.append("Small sample size - statistics may not be representative.")
            if s.get("missing_military_classes"):
                limitations.append(
                    f"Missing military classes in model: {', '.join(s['missing_military_classes'])}. "
                    "Military detection capability may be limited."
                )
            if s.get("unmapped_classes"):
                limitations.append(
                    f"Unmapped classes (no threat level defined): {', '.join(s['unmapped_classes'])}."
                )
        if not limitations:
            limitations.append("No significant data quality limitations identified.")

        # Recommendations
        recommendations = []
        if is_image:
            recommendations.append("Cross-reference detections with AIS data where available.")
            recommendations.append("Test on additional images for statistical reliability.")
            if hp > 0:
                recommendations.append("Investigate and track all HIGH PRIORITY contacts immediately.")
            if pr > 0:
                recommendations.append("Monitor PRIORITY contacts and report to command.")
            if s.get("low_conf_classes"):
                recommendations.append("Conduct manual review of all low-confidence detections.")
            if s.get("missing_military_classes"):
                recommendations.append("Update model with military class weights to improve recall.")
        else:
            if hp > 0:
                recommendations.append("Investigate and track all HIGH PRIORITY contacts immediately.")
            if pr > 0:
                recommendations.append("Monitor PRIORITY contacts and report to command.")
            if s.get("low_conf_classes"):
                recommendations.append("Conduct manual review of all low-confidence detections.")
            if s.get("short_tracking"):
                recommendations.append("Review tracker configuration to improve track continuity.")
            if s.get("missing_military_classes"):
                recommendations.append("Update model with military class weights to improve recall.")
            recommendations.append("Cross-reference detections with AIS data where available.")
            recommendations.append("Archive detection log for post-mission audit trail.")
        if not recommendations:
            recommendations.append("Continue standard monitoring procedures.")

        # Captions
        if is_image:
            captions = {
                "track_chart": "N/A for still image input",
                "confidence_hist": (
                    f"Confidence distribution across {total} detection(s). "
                    f"Median: {s.get('median_confidence', 0):.1%}, "
                    f"threshold: {s.get('conf_threshold', 0.25):.0%}."
                ) if total > 0 else "No detections to display.",
                "class_table": (
                    f"Per-class breakdown for {len(s.get('per_class', []))} detected class(es). "
                    "Low-confidence classes are flagged for review."
                ),
                "annotated_frames": (
                    f"Annotated image with detection overlays for {total} detection(s). "
                    "Confidence scores shown per detection."
                ),
            }
        else:
            captions = {
                "track_chart": (
                    f"Per-vessel track chart showing {unique} vessel(s) across {frames} frame(s). "
                    "Each bar represents one vessel's detection span."
                ),
                "confidence_hist": (
                    f"Confidence distribution across {total} detections. "
                    f"Median: {s.get('median_confidence', 0):.1%}, threshold: {s.get('conf_threshold', 0.25):.0%}."
                ),
                "class_table": (
                    f"Per-class breakdown for {len(s.get('per_class', []))} detected class(es). "
                    "Low-confidence classes are flagged for review."
                ),
                "annotated_frames": (
                    "Annotated vessel crop images with detection overlay. "
                    "Confidence scores shown per crop."
                ),
            }

        return {
            "executive_summary": self._sanitize(exec_sum),
            "threat_assessment": self._sanitize(ta),
            "limitations": [self._sanitize(l) for l in limitations],
            "recommendations": [self._sanitize(r) for r in recommendations],
            "captions": {k: self._sanitize(v) for k, v in captions.items()},
        }

    # -- Charts ----------------------------------------------------------------

    def generate_charts(self):
        """Generate charts: Gantt track chart, confidence histogram, class bar."""
        if self.df is None:
            self.load_csv()
        charts = {}
        df = self.df
        chart_dir = self._chart_dir
        chart_dir.mkdir(parents=True, exist_ok=True)
        is_image = (self.input_type == 'image')

        # Class color map for Gantt
        class_colors = {}
        palette = [ACCENT, ACCENT_SOFT, GREEN, ORANGE, YELLOW, RED, "#7c4daf",
                   "#4daf73", "#a8602b", "#8855aa", "#3b7bd1"]
        all_classes = (list(df["class_name"].unique())
                       if df is not None and not df.empty and "class_name" in df.columns
                       else [])
        for i, cls in enumerate(all_classes):
            class_colors[cls] = palette[i % len(palette)]

        # -- Chart 1: Per-vessel track chart (Gantt-style) -- skip for images
        if is_image:
            charts["track_chart"] = None
        else:
            fig, ax = plt.subplots(figsize=(6, 3))
            if (df is not None and not df.empty
                    and "vessel_id" in df.columns and "frame_id" in df.columns):
                vessels = sorted(df["vessel_id"].unique())
                for idx, vid in enumerate(vessels):
                    vdf = df[df["vessel_id"] == vid]
                    cls = str(vdf["class_name"].iloc[0]) if "class_name" in vdf.columns else "unknown"
                    ff = int(vdf["frame_id"].min())
                    lf = int(vdf["frame_id"].max())
                    span = max(lf - ff, 1)
                    color = class_colors.get(cls, ACCENT)
                    ax.barh(idx, span, left=ff, height=0.6, color=color,
                            edgecolor="white", linewidth=0.3)
                    label = f"#{int(vid)} {cls.replace('_', ' ')}"
                    ax.text(lf + max(span * 0.05, 1), idx, label,
                            va="center", ha="left", fontsize=5.5, color="#333333")
                ax.set_yticks(range(len(vessels)))
                ax.set_yticklabels([f"#{int(v)}" for v in vessels], fontsize=6)
                ax.set_xlabel("Frame ID", fontsize=7)
            else:
                ax.text(0.5, 0.5, "No detections", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#999999")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title("Per-Vessel Track Duration", fontweight="bold", fontsize=9, pad=4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.invert_yaxis()
            fig.tight_layout()
            path = str(chart_dir / "track_chart.png")
            fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=150, pad_inches=0.02)
            plt.close(fig)
            charts["track_chart"] = path

        # -- Chart 2: Confidence histogram -- for images with <5 detections, use per-detection bar chart
        fig2, ax2 = plt.subplots(figsize=(5, 2.5))
        if (df is not None and not df.empty
                and "confidence" in df.columns and len(df) > 0):
            if is_image and len(df) < 5:
                # Per-detection bar chart: one bar per detection
                confs = list(df["confidence"])
                labels = [f"#{i+1}" for i in range(len(confs))]
                bar_colors = [RED if c < self.conf_thresh else ACCENT for c in confs]
                bars2 = ax2.bar(range(len(confs)), confs, color=bar_colors,
                                edgecolor="white", width=0.6)
                ax2.set_xticks(range(len(confs)))
                ax2.set_xticklabels(labels, fontsize=6)
                ax2.set_ylim(0, 1.0)
                ax2.set_xlabel("Detection #", fontsize=7)
                ax2.set_ylabel("Confidence", fontsize=7)
                for bar2, cv in zip(bars2, confs):
                    ax2.text(bar2.get_x() + bar2.get_width() / 2,
                             bar2.get_height() + 0.02,
                             f"{cv:.1%}", ha="center", va="bottom",
                             fontsize=6, color="#333333")
            else:
                ax2.hist(df["confidence"], bins=20, color=ACCENT, edgecolor="white", alpha=0.85)
                median_val = float(df["confidence"].median())
                conf_thr = self.conf_thresh
                ax2.axvline(median_val, color=ORANGE, linestyle="--", linewidth=1.2,
                            label=f"Median: {median_val:.1%}")
                ax2.axvline(conf_thr, color=RED, linestyle=":", linewidth=1.2,
                            label=f"Threshold: {conf_thr:.0%}")
                ax2.legend(fontsize=6, frameon=False, labelcolor="#444444")
                ax2.set_xlabel("Confidence", fontsize=7)
                ax2.set_ylabel("Count", fontsize=7)
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=10, color="#999999")
            ax2.set_xlabel("Confidence", fontsize=7)
            ax2.set_ylabel("Count", fontsize=7)
        ax2.set_title("Confidence Distribution", fontweight="bold", fontsize=9, pad=4)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(labelsize=6)
        fig2.tight_layout()
        path2 = str(chart_dir / "conf_hist.png")
        fig2.savefig(path2, bbox_inches="tight", facecolor="white", dpi=150, pad_inches=0.02)
        plt.close(fig2)
        charts["conf_hist"] = path2

        # -- Chart 3: Per-class bar chart --
        fig3, ax3 = plt.subplots(figsize=(5, 2.5))
        if df is not None and not df.empty and "class_name" in df.columns:
            class_counts = df["class_name"].value_counts().sort_values(ascending=True)
            bars = ax3.barh(class_counts.index, class_counts.values, color=ACCENT,
                            edgecolor="white", height=0.55)
            ax3.set_xlabel("Count", fontsize=7)
            ax3.tick_params(labelsize=6)
            for bar in bars:
                w = bar.get_width()
                ax3.text(w + 0.1, bar.get_y() + bar.get_height() / 2,
                         str(int(w)), va="center", fontsize=5.5, color="#444444")
        else:
            ax3.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax3.transAxes, fontsize=10, color="#999999")
        ax3.set_title("Detections by Class", fontweight="bold", fontsize=9, pad=4)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        fig3.tight_layout()
        path3 = str(chart_dir / "class_bar.png")
        fig3.savefig(path3, bbox_inches="tight", facecolor="white", dpi=150, pad_inches=0.02)
        plt.close(fig3)
        charts["class_bar"] = path3

        self.chart_paths = charts
        return charts

    # -- PDF helpers ------------------------------------------------------------

    def _section_header(self, pdf, title, y=None):
        """Draw section header with teal text and thin underline. Big gap before and after."""
        # Gap before the section (unless we're at the top of a page)
        if pdf.get_y() > 25:
            pdf.ln(8)
            
        # Check if we need a page break (heuristic: 30mm for header + some content)
        if pdf.get_y() > pdf.h - pdf.b_margin - 30:
            pdf.add_page()
            
        if y is not None:
            pdf.set_xy(15, y)
        else:
            pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(26, 107, 138)
        pdf.cell(0, 7, title)
        pdf.ln(1)
        pdf.set_draw_color(26, 107, 138)
        pdf.set_line_width(0.3)
        y_line = pdf.get_y()
        pdf.line(15, y_line, 195, y_line)
        pdf.ln(8)

    def _footer(self, pdf):
        """Draw footer at bottom of current page."""
        pdf.set_auto_page_break(auto=False)
        pdf.set_xy(15, 285)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(180, 185, 190)
        model_name = self.stats.get("model_weights_name", "N/A")
        conf_pct = int(self.conf_thresh * 100) if isinstance(self.conf_thresh, (int, float)) else 25
        footer_text = (f"AI-generated analysis: verify with raw detection log | "
                       f"Model: {model_name} | Conf threshold: {conf_pct}%")
        pdf.cell(0, 4, footer_text, align="L")
        pdf.set_auto_page_break(auto=True, margin=18)

    def _place_image(self, pdf, path, x, y, w, max_h=None):
        """Place an image at (x, y) with width w. Returns height in mm.
        If max_h is set, image is scaled to fit within both w and max_h (uniform sizing)."""
        from PIL import Image as PILImage
        img = PILImage.open(path)
        aspect = img.height / img.width
        h = w * aspect
        # If max_h is set and the image is too tall, scale down to fit
        if max_h is not None and h > max_h:
            h = max_h
            w = h / aspect if aspect > 0 else w
            
        # Check if image goes past the bottom margin
        if y + h > pdf.h - pdf.b_margin:
            pdf.add_page()
            y = pdf.get_y()
            
        pdf.image(path, x=x, y=y, w=w)
        pdf.set_xy(x, y + h)
        return h

    # -- PDF builder ------------------------------------------------------------

    def build_pdf(self):
        if not self.stats:
            self.compute_stats()
        if not self.chart_paths:
            self.generate_charts()
        if self.llm_analysis is None:
            self.llm_analysis = self.generate_llm_analysis(self.stats)

        llm = self.llm_analysis
        s = self.stats
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)

        # ============================================================
        # PAGE 1: Header + Session Metadata + Executive Summary + Key Metrics
        # ============================================================
        pdf.add_page()

        # -- Header (tight, no big gap) --
        pdf.set_xy(15, 15)
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(26, 107, 138)
        pdf.cell(0, 10, "PROJECT GUARDIAN")
        pdf.set_xy(15, 27)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(80, 100, 120)
        pdf.cell(0, 6, "Maritime Domain Awareness - Mission Report")
        pdf.set_draw_color(26, 107, 138)
        pdf.set_line_width(0.5)
        pdf.line(15, 36, 195, 36)

        # -- Session metadata block --
        pdf.set_xy(15, 40)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(120, 130, 140)
        vid_fps = s.get("video_fps", 0)
        vid_res = f"{s.get('video_width', 0)}x{s.get('video_height', 0)}"
        conf_pct = int(s.get("conf_threshold", 0.25) * 100) if isinstance(s.get("conf_threshold"), (int, float)) else 25
        is_image = (self.input_type == 'image')
        if is_image:
            img_res = s.get("image_resolution", "N/A")
            meta_lines = [
                f"Session ID: {s.get('session_id', 'N/A')}",
                f"Generated: {s.get('timestamp', 'N/A')}",
                f"Source: {self._sanitize(self.session_label)}",
                f"Model weights: {s.get('model_weights_name', 'N/A')}",
                f"Classes: {s.get('num_classes', 0)}",
                f"LLM model: {s.get('llm_model', 'N/A')}",
                f"Conf threshold: {conf_pct}%",
                f"IoU threshold: {s.get('iou_threshold', 0.5)}",
                f"Input: still image, {img_res} px",
            ]
        else:
            meta_lines = [
                f"Session ID: {s.get('session_id', 'N/A')}",
                f"Generated: {s.get('timestamp', 'N/A')}",
                f"Source: {self._sanitize(self.session_label)}",
                f"Model weights: {s.get('model_weights_name', 'N/A')}",
                f"Classes: {s.get('num_classes', 0)}",
                f"LLM model: {s.get('llm_model', 'N/A')}",
                f"Conf threshold: {conf_pct}%",
                f"IoU threshold: {s.get('iou_threshold', 0.5)}",
                f"Tracker: {s.get('tracker_name', 'ByteTrack')}",
                f"Video: {vid_res} @ {vid_fps} fps, {s.get('video_total_frames', 0)} frames, {s.get('video_duration_s', 0)}s",
            ]
        for line in meta_lines:
            pdf.cell(0, 4.5, line)
            pdf.ln(4.5)

        # -- Executive Summary --
        pdf.ln(2)
        self._section_header(pdf, "EXECUTIVE SUMMARY")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 55, 65)
        pdf.multi_cell(180, 5.5, llm.get("executive_summary", ""))
        pdf.ln(6)

        # -- Key Metrics --
        self._section_header(pdf, "KEY METRICS")
        total = s.get("total_detections", 0)
        unique = s.get("unique_vessels", 0)
        frames = s.get("frames_processed", 0)
        avg_c = s.get("avg_confidence", 0)
        min_c = s.get("min_confidence", 0)
        max_c = s.get("max_confidence", 0)
        med_c = s.get("median_confidence", 0)
        threat_ol = s.get("threat_one_liner",
                          "0 threat contacts detected (0 HIGH PRIORITY, 0 PRIORITY)")

        metrics = [
            f"Vessels: {unique}  |  Detections: {total}  |  Frames: {frames}",
            threat_ol,
            (f"Confidence - Avg: {avg_c:.1%}  |  Median: {med_c:.1%}  |  "
             f"Min: {min_c:.1%}  |  Max: {max_c:.1%}"),
            (f"Threat breakdown (vessels) - HIGH PRIORITY: {s.get('high_priority', 0)}  |  "
             f"PRIORITY: {s.get('priority', 0)}  |  MONITOR: {s.get('monitor', 0)}  |  "
             f"SMALL CRAFT: {s.get('small_craft', 0)}  |  CIVILIAN: {s.get('civilian', 0)}"),
        ]
        if s.get("avg_duration", 0) > 0:
            metrics.append(
                f"Tracking duration - Avg: {s['avg_duration']:.1f}s  |  "
                f"Max: {s.get('max_duration', 0):.1f}s  |  "
                f"Min: {s.get('min_duration', 0):.1f}s")
        start_y = pdf.get_y()
        for i, m in enumerate(metrics):
            bg = (238, 243, 247) if i % 2 == 0 else (248, 250, 253)
            pdf.set_fill_color(*bg)
            y = start_y + i * 8
            pdf.rect(15, y, 180, 7, "F")
            pdf.set_xy(18, y + 1.5)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(50, 55, 65)
            pdf.cell(174, 4, m)
        pdf.ln(5)  # gap after key metrics before next section

        # -- Vessel Table / Detection Table --
        if is_image:
            self._section_header(pdf, "DETECTION TABLE")

            per_vessel = s.get("per_vessel", [])
            if self.df is not None and not self.df.empty:
                det_rows = []
                for i, row in self.df.iterrows():
                    cls_name = str(row.get("class_name", "unknown"))
                    tl = str(row.get("threat_level", THREAT_LEVEL.get(cls_name, "UNKNOWN")))
                    conf = float(row.get("confidence", 0.0))
                    x1 = float(row.get("x1", 0)); y1 = float(row.get("y1", 0))
                    x2 = float(row.get("x2", 0)); y2 = float(row.get("y2", 0))
                    bbox_str = f"{int(x2-x1)}x{int(y2-y1)}"
                    low_conf = (conf < 0.50)
                    det_rows.append({
                        "index": i + 1,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": bbox_str,
                        "threat_level": tl,
                        "low_conf": low_conf,
                    })

                # Table header
                hdr_y = pdf.get_y()
                pdf.set_fill_color(230, 238, 242)
                col_w = [12, 40, 24, 28, 36, 24]
                total_w = sum(col_w)
                pdf.rect(15, hdr_y, total_w, 8, "F")
                pdf.set_xy(16, hdr_y + 1.5)
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(26, 107, 138)
                headers = ["#", "Class", "Conf", "Bbox (WxH)", "Threat Level", "Review"]
                for hw, htext in zip(col_w, headers):
                    pdf.cell(hw, 5, htext, align="C")
                pdf.ln(8)

                # Table rows
                pdf.set_font("Helvetica", "", 8)
                row_h = 8
                for r in det_rows:
                    row_bg = (245, 248, 250) if r["index"] % 2 == 1 else (250, 252, 254)
                    y = pdf.get_y()
                    pdf.set_fill_color(*row_bg)
                    pdf.rect(15, y, total_w, row_h, "F")
                    pdf.set_xy(16, y + 2)
                    pdf.set_text_color(50, 55, 65)
                    review_val = r["low_conf"]
                    review_text = "LOW CONF" if review_val else "OK"
                    review_color = _hex_to_rgb(RED) if review_val else _hex_to_rgb(GREEN)
                    vals = [
                        (str(r["index"]), "C"),
                        (r["class_name"].replace("_", " ").title(), "L"),
                        (f"{r['confidence']:.1%}", "C"),
                        (r["bbox"], "C"),
                        (r["threat_level"], "C"),
                    ]
                    for j, (val, align) in enumerate(vals):
                        cw = col_w[j]
                        pdf.cell(cw, 4, val, align=align)
                    pdf.set_text_color(*review_color)
                    pdf.set_font("Helvetica", "B", 7)
                    pdf.cell(col_w[-1], 4, review_text, align="C")
                    pdf.set_font("Helvetica", "", 8)
                    pdf.set_text_color(50, 55, 65)
                    pdf.ln(row_h)
                pdf.ln(2)
            else:
                per_vessel = []
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(120, 120, 120)
                pdf.cell(0, 8, "No detections in this image.")
                pdf.ln(8)
        else:
            self._section_header(pdf, "VESSEL TABLE")

        per_vessel = s.get("per_vessel", [])
        if per_vessel:
            crops_dir = self.frame_crops_dir
            has_crops = crops_dir is not None and crops_dir.exists()

            # Table header
            hdr_y = pdf.get_y()
            pdf.set_fill_color(230, 238, 242)
            if has_crops:
                col_w = [15, 12, 28, 14, 28, 16, 18, 16, 22, 16]
            else:
                col_w = [12, 30, 16, 28, 18, 18, 16, 24, 18]
            total_w = sum(col_w)
            pdf.rect(15, hdr_y, total_w, 8, "F")
            pdf.set_xy(16, hdr_y + 1.5)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(26, 107, 138)
            if has_crops:
                headers = ["", "ID", "Class", "Det", "Frame Range", "Dur(s)",
                           "Avg Conf", "Min Conf", "Threat Level", "Review"]
            else:
                headers = ["ID", "Class", "Det", "Frame Range", "Dur(s)",
                           "Avg Conf", "Min Conf", "Threat Level", "Review"]
            for hw, htext in zip(col_w, headers):
                pdf.cell(hw, 5, htext, align="C")
            pdf.ln(8)

            # Table rows
            pdf.set_font("Helvetica", "", 8)
            row_h = 10
            for i, v in enumerate(per_vessel):
                row_bg = (245, 248, 250) if i % 2 == 0 else (250, 252, 254)
                y = pdf.get_y()
                pdf.set_fill_color(*row_bg)
                pdf.rect(15, y, total_w, row_h, "F")

                col_idx = 0
                if has_crops:
                    crop_path = crops_dir / f"vessel_{v['vessel_id']}.jpg"
                    if crop_path.exists():
                        try:
                            from PIL import Image as PILImage
                            img = PILImage.open(crop_path)
                            aspect = img.height / img.width
                            tw = 13
                            th = tw * aspect
                            if th > row_h - 2:
                                th = row_h - 2
                                tw = th / aspect if aspect > 0 else 13
                            ty = y + (row_h - th) / 2
                            tx = 16 + (15 - tw) / 2
                            pdf.image(str(crop_path), x=tx, y=ty, w=tw)
                        except Exception:
                            pass
                    col_idx = 1

                pdf.set_xy(15 + sum(col_w[:col_idx]) + 1, y + 2.5)
                pdf.set_text_color(50, 55, 65)
                review_val = v.get("low_confidence", False)
                review_text = "LOW CONF" if review_val else "OK"
                review_color = _hex_to_rgb(RED) if review_val else _hex_to_rgb(GREEN)

                values = [
                    (str(v["vessel_id"]), "C"),
                    (v["class_name"].replace("_", " ").title(), "L"),
                    (str(v["detection_count"]), "C"),
                    (f"{v['first_frame']}-{v['last_frame']}", "C"),
                    (f"{v['duration_s']:.1f}", "C"),
                    (f"{v['avg_conf']:.1%}", "C"),
                    (f"{v['min_conf']:.1%}", "C"),
                    (str(v["threat_level"]), "C"),
                ]
                for j, (val, align) in enumerate(values):
                    cw = col_w[col_idx + j]
                    if align == "C":
                        pdf.cell(cw, 5, val, align="C")
                    else:
                        pdf.cell(cw, 5, val, align="L")
                cw_rev = col_w[-1]
                pdf.set_text_color(*review_color)
                pdf.set_font("Helvetica", "B", 7)
                pdf.cell(cw_rev, 5, review_text, align="C")
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(50, 55, 65)
                pdf.ln(row_h)

            pdf.ln(2)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 8, "No vessels detected in this session.")
            pdf.ln(8)

        # -- Annotated Frames (if crops available) -- (video mode only)
        if not is_image:
            crops_dir = self.frame_crops_dir
            if crops_dir and crops_dir.exists():
                crop_files = sorted(crops_dir.glob("vessel_*.jpg"))
                if crop_files:
                    pdf.add_page()
                    self._section_header(pdf, "ANNOTATED FRAMES")
                    caps = llm.get("captions", {})
                    if caps.get("annotated_frames"):
                        pdf.set_font("Helvetica", "", 9)
                        pdf.set_text_color(80, 90, 100)
                        pdf.multi_cell(180, 5, caps["annotated_frames"])
                        pdf.ln(3)

                    shown = crop_files[:6]
                    img_w = 80
                    row_y = pdf.get_y()
                    for idx, cf in enumerate(shown):
                        col = idx % 2
                        if col == 0 and idx > 0:
                            row_y = pdf.get_y() + 4
                        x = 15 + col * 90
                        m = re.search(r"vessel_(\d+)", cf.name)
                        vid = int(m.group(1)) if m else 0
                        vinfo = None
                        for v in per_vessel:
                            if v["vessel_id"] == vid:
                                vinfo = v
                                break
                        label = f"#{vid}"
                        if vinfo:
                            label = (f"#{vid} {vinfo['class_name'].replace('_', ' ')} "
                                     f"{vinfo['avg_conf']:.0%}")
                        try:
                            h = self._place_image(pdf, str(cf), x=x, y=row_y, w=img_w)
                            pdf.set_xy(x, row_y + h + 0.5)
                            pdf.set_font("Helvetica", "", 8)
                            pdf.set_text_color(80, 90, 100)
                            pdf.cell(img_w, 4, label, align="C")
                            if col == 1:
                                row_y = row_y + h + 8
                        except Exception:
                            pass

        # -- Annotated Image section (image or video mode) --
        # For image mode: single annotated image
        # For video mode: up to 3 key frames (first, peak, last)
        all_annotated = []
        ll = "AI-generated" if self.use_llm else "Template (LLM unavailable)"
        if is_image:
            if self.annotated_image_path and Path(str(self.annotated_image_path)).exists():
                total_det = s.get("total_detections", 0)
                cap = (f"Annotated image with #ID class conf% labels for {total_det} detection(s). ({ll})")
                all_annotated.append((str(self.annotated_image_path), cap))
        else:
            # Video mode: use annotated_frames list (each has its own label)
            for fpath, flabel in self.annotated_frames:
                if Path(fpath).exists():
                    all_annotated.append((fpath, f"{flabel}. ({ll})"))
            # Fallback: if no annotated_frames but annotated_image_path exists
            if not all_annotated and self.annotated_image_path and Path(str(self.annotated_image_path)).exists():
                total_det = s.get("total_detections", 0)
                cap = (f"Peak threat frame - annotated with #ID class conf% labels for {total_det} detection(s). ({ll})")
                all_annotated.append((str(self.annotated_image_path), cap))

        if all_annotated:
            self._section_header(pdf, "ANNOTATED FRAMES" if len(all_annotated) > 1 else "ANNOTATED IMAGE")
            
            if len(all_annotated) == 1:
                img_w = 170
                img_max_h = 100
                ann_path, caption = all_annotated[0]
                if Path(ann_path).exists():
                    try:
                        h = self._place_image(pdf, ann_path, x=20, y=pdf.get_y(), w=img_w, max_h=img_max_h)
                        pdf.set_y(pdf.get_y() + 4)
                        pdf.set_font("Helvetica", "", 8)
                        pdf.set_text_color(100, 110, 120)
                        pdf.multi_cell(180, 4.5, caption)
                        pdf.ln(4)
                    except Exception:
                        pass
            else:
                img_w = 85
                img_max_h = 60
                row_y = pdf.get_y()
                max_row_y = row_y
                valid_images = [a for a in all_annotated if Path(a[0]).exists()]
                for idx, (ann_path, caption) in enumerate(valid_images):
                    col = idx % 2
                    if col == 0 and idx > 0:
                        row_y = max_row_y + 4
                        pdf.set_y(row_y)
                    
                    x = 15 + col * (img_w + 5)
                    try:
                        # Pre-check page break for new rows
                        if col == 0 and row_y + img_max_h + 15 > pdf.h - pdf.b_margin:
                            pdf.add_page()
                            row_y = pdf.get_y()
                            max_row_y = row_y
                            
                        h = self._place_image(pdf, ann_path, x=x, y=row_y, w=img_w, max_h=img_max_h)
                        pdf.set_xy(x, row_y + h + 1)
                        pdf.set_font("Helvetica", "", 8)
                        pdf.set_text_color(100, 110, 120)
                        pdf.multi_cell(img_w, 4.5, caption, align="C")
                        max_row_y = max(max_row_y, pdf.get_y())
                    except Exception:
                        pass
                pdf.set_y(max_row_y + 4)

        # ============================================================
        # Track Chart + Confidence Histogram (flows after vessel table)
        # ============================================================

        # -- Per-Vessel Track Chart -- skip for images
        caps = llm.get("captions", {})
        if is_image:
            pass  # No track chart for still images
        else:
            self._section_header(pdf, "PER-VESSEL TRACK CHART")
            track_path = self.chart_paths.get("track_chart")
            if track_path and Path(track_path).exists():
                h = self._place_image(pdf, track_path, x=15, y=pdf.get_y(), w=180)
                pdf.set_y(pdf.get_y() + 4)
            if caps.get("track_chart"):
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(100, 110, 120)
                pdf.multi_cell(180, 4.5, caps["track_chart"])
                pdf.ln(2)

        # -- Confidence Histogram --
        self._section_header(pdf, "CONFIDENCE HISTOGRAM")
        hist_path = self.chart_paths.get("conf_hist")
        if hist_path and Path(hist_path).exists():
            h = self._place_image(pdf, hist_path, x=30, y=pdf.get_y(), w=150)
            pdf.set_y(pdf.get_y() + 4)
        if caps.get("confidence_hist"):
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(100, 110, 120)
            pdf.multi_cell(180, 4.5, caps["confidence_hist"])
            pdf.ln(2)

        # ============================================================
        # Per-Class Table + Threat Assessment (flows after histogram)
        # ============================================================

        # -- Per-Class Table --
        self._section_header(pdf, "PER-CLASS TABLE")
        per_class = s.get("per_class", [])
        detected_classes = set()
        if per_class:
            for c in per_class:
                detected_classes.add(c["class_name"])
        model_class_names = list(self.model_names.values()) if self.model_names else []
        undetected = [c for c in model_class_names if c not in detected_classes]

        if per_class or undetected:
            hdr_y = pdf.get_y()
            pdf.set_fill_color(230, 238, 242)
            col_w = [48, 18, 20, 24, 22, 22, 26]
            total_w = sum(col_w)
            pdf.rect(15, hdr_y, total_w, 8, "F")
            pdf.set_xy(16, hdr_y + 1.5)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(26, 107, 138)
            headers = ["Class", "Vessels", "Detections", "Avg Conf",
                        "Min Conf", "Max Conf", "Review"]
            for hw, htext in zip(col_w, headers):
                pdf.cell(hw, 5, htext, align="C")
            pdf.ln(8)

            pdf.set_font("Helvetica", "", 8)
            row_h = 8
            row_idx = 0
            for c in per_class:
                row_bg = (245, 248, 250) if row_idx % 2 == 0 else (250, 252, 254)
                y = pdf.get_y()
                pdf.set_fill_color(*row_bg)
                pdf.rect(15, y, total_w, row_h, "F")
                pdf.set_xy(16, y + 2)
                pdf.set_text_color(50, 55, 65)
                review_val = c.get("low_confidence", False)
                review_text = "LOW CONF" if review_val else "OK"
                review_color = _hex_to_rgb(RED) if review_val else _hex_to_rgb(GREEN)
                vals = [
                    (c["class_name"].replace("_", " ").title(), "L"),
                    (str(c["vessel_count"]), "C"),
                    (str(c["detection_count"]), "C"),
                    (f"{c['avg_conf']:.1%}", "C"),
                    (f"{c['min_conf']:.1%}", "C"),
                    (f"{c['max_conf']:.1%}", "C"),
                ]
                for j, (val, align) in enumerate(vals):
                    cw = col_w[j]
                    pdf.cell(cw, 4, val, align=align)
                pdf.set_text_color(*review_color)
                pdf.set_font("Helvetica", "B", 7)
                pdf.cell(col_w[-1], 4, review_text, align="C")
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(50, 55, 65)
                pdf.ln(row_h)
                row_idx += 1

            # Undetected classes (0 counts) -- video mode shows full table; image mode lists in one line
            if is_image:
                if undetected:
                    pdf.ln(2)
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(150, 150, 150)
                    undetected_display = [c.replace("_", " ").title() for c in sorted(undetected)]
                    pdf.multi_cell(180, 4.5,
                        f"Not detected: {', '.join(undetected_display)}")
                    pdf.set_text_color(50, 55, 65)
            else:
                for cls_name in sorted(undetected):
                    row_bg = (245, 248, 250) if row_idx % 2 == 0 else (250, 252, 254)
                    y = pdf.get_y()
                    pdf.set_fill_color(*row_bg)
                    pdf.rect(15, y, total_w, row_h, "F")
                    pdf.set_xy(16, y + 2)
                    pdf.set_text_color(180, 180, 180)
                    vals = [
                        (cls_name.replace("_", " ").title(), "L"),
                        ("0", "C"), ("0", "C"),
                        ("N/A", "C"), ("N/A", "C"), ("N/A", "C"),
                    ]
                    for j, (val, align) in enumerate(vals):
                        cw = col_w[j]
                        pdf.cell(cw, 4, val, align=align)
                    pdf.set_text_color(200, 200, 200)
                    pdf.set_font("Helvetica", "", 7)
                    pdf.cell(col_w[-1], 4, "N/A", align="C")
                    pdf.set_font("Helvetica", "", 8)
                    pdf.set_text_color(50, 55, 65)
                    pdf.ln(row_h)
                    row_idx += 1

            # Threat Level Legend sub-heading (image mode)
            if is_image:
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(26, 107, 138)
                pdf.cell(0, 5, "Threat Level Legend")
                pdf.ln(5)
                pdf.set_font("Helvetica", "", 7)
                pdf.set_text_color(80, 90, 100)
                tl_legend_parts = []
                tl_groups = {}
                for cls_name2, tl2 in THREAT_LEVEL.items():
                    tl_groups.setdefault(tl2, []).append(cls_name2)
                for tl_name2 in ["HIGH PRIORITY", "PRIORITY", "MONITOR", "SMALL CRAFT", "CIVILIAN"]:
                    if tl_name2 in tl_groups:
                        tl_legend_parts.append(f"{tl_name2}: {', '.join(tl_groups[tl_name2])}")
                tl_legend_text = " | ".join(tl_legend_parts)
                pdf.multi_cell(180, 4, tl_legend_text)
                pdf.ln(2)

            if s.get("unmapped_classes"):
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 8)
                red_rgb = _hex_to_rgb(RED)
                pdf.set_text_color(*red_rgb)
                pdf.multi_cell(180, 4.5,
                    f"Unmapped classes (no threat level defined): "
                    f"{', '.join(s['unmapped_classes'])}")
                pdf.set_text_color(50, 55, 65)

            if s.get("missing_military_classes"):
                pdf.ln(1)
                pdf.set_font("Helvetica", "B", 8)
                orange_rgb = _hex_to_rgb(ORANGE)
                pdf.set_text_color(*orange_rgb)
                pdf.multi_cell(180, 4.5,
                    f"Missing military classes in model: "
                    f"{', '.join(s['missing_military_classes'])}")
                pdf.set_text_color(50, 55, 65)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 8, "No class data available.")
            pdf.ln(8)

        # -- Threat Assessment --
        pdf.ln(6)
        self._section_header(pdf, "THREAT ASSESSMENT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 55, 65)
        pdf.multi_cell(180, 5, llm.get("threat_assessment", ""))
        pdf.ln(3)

        # Threat legend
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(80, 90, 100)
        legend_parts = []
        tl_groups = {}
        for cls_name, tl in THREAT_LEVEL.items():
            tl_groups.setdefault(tl, []).append(cls_name)
        for tl_name in ["HIGH PRIORITY", "PRIORITY", "MONITOR", "SMALL CRAFT", "CIVILIAN"]:
            if tl_name in tl_groups:
                legend_parts.append(f"{tl_name}: {', '.join(tl_groups[tl_name])}")
        legend_text = " | ".join(legend_parts)
        pdf.multi_cell(180, 4, legend_text)
        pdf.ln(2)

        # ============================================================
        # Limitations + Recommendations (flows after threat assessment)
        # ============================================================

        # -- Limitations & Data Quality --
        self._section_header(pdf, "LIMITATIONS & DATA QUALITY")
        limitations = llm.get("limitations", [])
        # Add roboflow source caveat if applicable
        source_label = (self.session_label or "").lower()
        if "roboflow" in source_label:
            limitations = list(limitations) + [
                "Source may be from the training/validation set; confidence may be optimistic."
            ]
        if limitations:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(50, 55, 65)
            for lim in limitations:
                pdf.set_x(18)
                pdf.cell(4, 5.5, "-", align="C")
                pdf.multi_cell(172, 5.5, lim)
                pdf.ln(1)
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 6, "No significant limitations identified.")
            pdf.ln(6)
        # Label: AI-generated or Template
        lim_label = "AI-generated (llama3.1:8b)" if self.use_llm else "Template (LLM unavailable)"
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(160, 160, 170)
        pdf.cell(0, 4, lim_label, align="L")
        pdf.ln(4)

        # -- Recommended Actions --
        pdf.ln(6)
        self._section_header(pdf, "RECOMMENDED ACTIONS")
        recommendations = llm.get("recommendations", [])
        if recommendations:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(50, 55, 65)
            for rec in recommendations:
                pdf.set_x(18)
                pdf.cell(4, 5.5, "-", align="C")
                pdf.multi_cell(172, 5.5, rec)
                pdf.ln(1)
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 6, "No specific recommendations at this time.")
            pdf.ln(6)
        rec_label = "AI-generated (llama3.1:8b)" if self.use_llm else "Template (LLM unavailable)"
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(160, 160, 170)
        pdf.cell(0, 4, rec_label, align="L")
        pdf.ln(4)

        self._footer(pdf)

        return bytes(pdf.output(dest="S"))

    def build_pdf_bytes(self):
        """Return the PDF as bytes (alias for build_pdf)."""
        return self.build_pdf()
