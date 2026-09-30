"""
ReportGenerator - Intelligence Pipeline for SEDIC 2026 Grand Finale.

Reads a detection CSV (produced by DetectionLogger), performs statistical
analysis with Pandas, generates charts with Matplotlib, writes an executive
summary via Ollama (local LLM, with template fallback), and stitches
everything into a mission-ready PDF using FPDF.

The report is designed to be printed and handed to a commanding officer.
It is formal, concise, and fits on 1-2 pages.
"""

import os
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
DARK_BG     = "#f5f5f5"   # light grey for edges

THREAT_COLORS = {
    "HIGH PRIORITY": RED,
    "PRIORITY":      ORANGE,
    "MONITOR":       YELLOW,
    "SMALL CRAFT":   "#b8a020",
    "CIVILIAN":      GREEN,
}


class ReportGenerator:
    """Turns a detection CSV into a mission-ready PDF report."""

    def __init__(self, csv_path: str, session_label: str = ""):
        self.csv_path = Path(csv_path)
        self.session_label = session_label or self.csv_path.stem
        self.df = None
        self.stats = {}
        self.chart_paths = {}
        self._chart_dir = Path("outputs/charts")
        self._chart_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        return self.df

    def compute_stats(self):
        if self.df is None:
            self.load_csv()
        df = self.df
        if df.empty:
            self.stats = {"total_detections": 0}
            return self.stats
        total = len(df)
        unique_vessels = df["vessel_id"].nunique() if "vessel_id" in df.columns else 0
        frames_processed = df["frame_id"].nunique() if "frame_id" in df.columns else 0
        session_id = df["session_id"].iloc[0] if "session_id" in df.columns else "N/A"
        class_counts = df["class_name"].value_counts().to_dict()
        threat_counts = df["threat_level"].value_counts().to_dict()
        high_priority = threat_counts.get("HIGH PRIORITY", 0)
        priority = threat_counts.get("PRIORITY", 0)
        civilian = threat_counts.get("CIVILIAN", 0)
        monitor = threat_counts.get("MONITOR", 0)
        small_craft = threat_counts.get("SMALL CRAFT", 0)
        threat_total = high_priority + priority
        threat_ratio = round(threat_total / total * 100, 1) if total else 0
        avg_conf = round(float(df["confidence"].mean()), 3) if "confidence" in df.columns else 0
        min_conf = round(float(df["confidence"].min()), 3) if "confidence" in df.columns else 0
        max_conf = round(float(df["confidence"].max()), 3) if "confidence" in df.columns else 0
        per_class_conf = {}
        if "confidence" in df.columns:
            for cls in df["class_name"].unique():
                subset = df[df["class_name"] == cls]
                per_class_conf[cls] = {
                    "count": len(subset),
                    "avg_conf": round(float(subset["confidence"].mean()), 3),
                    "min_conf": round(float(subset["confidence"].min()), 3),
                    "max_conf": round(float(subset["confidence"].max()), 3),
                }
        duration_stats = {}
        if "detection_duration" in df.columns and "vessel_id" in df.columns:
            per_vessel = df.groupby("vessel_id")["detection_duration"].max()
            if not per_vessel.empty:
                duration_stats = {
                    "avg_duration": round(float(per_vessel.mean()), 2),
                    "max_duration": round(float(per_vessel.max()), 2),
                    "min_duration": round(float(per_vessel.min()), 2),
                }
        self.stats = {
            "session_id": session_id, "total_detections": total,
            "unique_vessels": unique_vessels, "frames_processed": frames_processed,
            "class_counts": class_counts, "threat_counts": threat_counts,
            "high_priority": high_priority, "priority": priority,
            "civilian": civilian, "monitor": monitor, "small_craft": small_craft,
            "threat_total": threat_total, "threat_ratio": threat_ratio,
            "avg_confidence": avg_conf, "min_confidence": min_conf, "max_confidence": max_conf,
            "per_class_conf": per_class_conf, "duration_stats": duration_stats,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }
        return self.stats

    def generate_charts(self):
        if self.df is None:
            self.load_csv()
        if self.df.empty:
            return {}
        df = self.df
        charts = {}
        # Chart 1: Threat distribution (donut, compact)
        threat_cols = ["HIGH PRIORITY", "PRIORITY", "MONITOR", "SMALL CRAFT", "CIVILIAN"]
        threat_vals = [self.stats["threat_counts"].get(k, 0) for k in threat_cols]
        threat_labels = [k for k, v in zip(threat_cols, threat_vals) if v > 0]
        threat_data = [v for v in threat_vals if v > 0]
        colors = [THREAT_COLORS.get(k, "#888888") for k in threat_labels]
        if threat_data:
            fig, ax = plt.subplots(figsize=(3.5, 2))
            wedges, texts, autotexts = ax.pie(
                threat_data, labels=threat_labels, autopct="%1.0f%%",
                colors=colors, startangle=90, textprops={"fontsize": 5.5},
                wedgeprops={"edgecolor": "white", "linewidth": 1, "width": 0.45},
                pctdistance=0.7, labeldistance=1.12)
            for t in autotexts:
                t.set_color("#000000"); t.set_fontsize(5.5); t.set_fontweight("bold")
            ax.set_title("Threat Distribution", fontweight="bold", fontsize=7, pad=1)
            fig.tight_layout()
            path = str(self._chart_dir / "threat_pie.png")
            fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=150, pad_inches=0.02)
            plt.close(fig); charts["threat_pie"] = path
        # Chart 2: Per-class counts (horizontal bar, compact)
        class_counts = df["class_name"].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(3.5, 2))
        bars = ax.barh(class_counts.index, class_counts.values, color=ACCENT,
                       edgecolor="white", height=0.55)
        ax.set_xlabel("Count", fontsize=6)
        ax.set_title("Detections by Class", fontweight="bold", fontsize=7, pad=2)
        ax.tick_params(labelsize=5.5)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(int(w)), va="center", fontsize=5.5, color="#444444")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        path = str(self._chart_dir / "class_bar.png")
        fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=150, pad_inches=0.02)
        plt.close(fig); charts["class_bar"] = path
        # Chart 3: Confidence distribution (histogram, compact)
        if "confidence" in df.columns and len(df["confidence"]) >= 2:
            fig, ax = plt.subplots(figsize=(4.5, 1.8))
            ax.hist(df["confidence"], bins=min(15, len(df)), color=ACCENT,
                    edgecolor="white", alpha=0.85)
            ax.set_xlabel("Confidence", fontsize=6); ax.set_ylabel("Count", fontsize=6)
            ax.tick_params(labelsize=5.5)
            mean_val = df["confidence"].mean()
            ax.axvline(mean_val, color=ORANGE, linestyle="--", linewidth=1,
                       label=f"Mean: {mean_val:.1%}")
            ax.legend(fontsize=5, frameon=False, labelcolor="#444444")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            fig.tight_layout()
            path = str(self._chart_dir / "conf_hist.png")
            fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=150, pad_inches=0.02)
            plt.close(fig); charts["conf_hist"] = path
        # Chart 4: Detection timeline (only for video with >1 frame)
        if "frame_id" in df.columns and df["frame_id"].nunique() > 1:
            per_frame = df.groupby("frame_id").size()
            fig, ax = plt.subplots(figsize=(5.5, 2))
            ax.plot(per_frame.index, per_frame.values, color=ACCENT, linewidth=1, alpha=0.9)
            ax.fill_between(per_frame.index, per_frame.values, alpha=0.12, color=ACCENT)
            ax.set_xlabel("Frame", fontsize=7); ax.set_ylabel("Detections", fontsize=7)
            ax.set_title("Detection Timeline", fontweight="bold", fontsize=9, pad=6)
            ax.tick_params(labelsize=6.5)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            fig.tight_layout()
            path = str(self._chart_dir / "timeline.png")
            fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=150)
            plt.close(fig); charts["timeline"] = path
        self.chart_paths = charts
        return charts

    def write_summary(self):
        s = self.stats
        if s.get("total_detections", 0) == 0:
            return ("No vessel contacts were recorded during this session. "
                    "The system maintained operational readiness throughout.")
        ctx = (
            f"Session ID: {s['session_id']}\n"
            f"Total detections: {s['total_detections']}\n"
            f"Unique vessels tracked: {s['unique_vessels']}\n"
            f"Frames processed: {s['frames_processed']}\n"
            f"High Priority threats: {s['high_priority']}\n"
            f"Priority threats: {s['priority']}\n"
            f"Civilian vessels: {s['civilian']}\n"
            f"Threat contact ratio: {s['threat_ratio']}%\n"
            f"Average confidence: {s['avg_confidence']:.1%}\n"
            f"Class breakdown: {s['class_counts']}\n"
        )
        try:
            import httpx
            resp = httpx.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
                    "prompt": (
                        "You are a maritime domain awareness duty officer. "
                        "Write a formal 3-4 sentence operational summary of the "
                        "following detection session for a commanding officer. "
                        "Be factual, concise, and professional. Use military tone. "
                        "Flag any HIGH PRIORITY contacts immediately. "
                        "Do not repeat these instructions. Output only the summary.\n\n"
                        f"{ctx}"
                    ),
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 200},
                },
                timeout=30.0,
            )
            data = resp.json()
            summary = data.get("response", "").strip()
            # Strip markdown formatting (**, *, #, etc.)
            import re
            summary = re.sub(r'\*+', '', summary)
            summary = re.sub(r'^#+\s*', '', summary, flags=re.MULTILINE)
            # Strip any preamble the LLM might still add
            for prefix in ["Here is", "Certainly", "Sure", "Summary:", "Executive Summary:",
                           "Operational Summary", "OPERATIONAL SUMMARY"]:
                if summary.strip().startswith(prefix.strip()):
                    lines = summary.strip().split("\n")
                    if len(lines) > 1:
                        summary = "\n".join(lines[1:]).strip()
                    break
            summary = summary.strip()
            if summary:
                return summary
        except Exception:
            pass
        high = s["high_priority"]; prio = s["priority"]
        total = s["total_detections"]; unique = s["unique_vessels"]
        avg_c = s["avg_confidence"]
        if high > 0:
            threat_str = f"{high} HIGH PRIORITY contact(s) identified - immediate action required."
        elif prio > 0:
            threat_str = f"{prio} PRIORITY contact(s) identified - monitor and report to command."
        else:
            threat_str = "No priority-level contacts identified. All vessels classified as civilian or small craft."
        return (
            f"During this session, {total} contact(s) were recorded across "
            f"{unique} unique vessel(s) over {s['frames_processed']} processed "
            f"frame(s). {threat_str} "
            f"Average model confidence was {avg_c:.1%}. "
            f"The system maintained operational readiness throughout."
        )

    @staticmethod
    def _sanitize(text):
        replacements = {
            "\u2014": "-", "\u2013": "-",
            "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"',
            "\u2026": "...", "\u00b0": " deg",
        }
        for uni, asc in replacements.items():
            text = text.replace(uni, asc)
        return text.encode("latin-1", "replace").decode("latin-1")

    def _section_header(self, pdf, title, y=None):
        if y is not None:
            pdf.set_xy(15, y)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(26, 107, 138)
        pdf.cell(0, 7, title)
        pdf.ln(1)
        pdf.set_draw_color(26, 107, 138)
        pdf.set_line_width(0.3)
        y_line = pdf.get_y()
        pdf.line(15, y_line, 195, y_line)
        pdf.ln(3)

    def _metric_row_at(self, pdf, x, y, w, label, value, alt=False):
        bg = (240, 245, 248) if alt else (248, 250, 252)
        pdf.set_fill_color(*bg)
        pdf.rect(x, y, w, 6.5, "F")
        pdf.set_xy(x + 3, y + 1)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(80, 90, 100)
        pdf.cell(w - 35, 5, label)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(40, 50, 60)
        pdf.cell(29, 5, value, align="R")

    def _metric_row_big(self, pdf, x, y, w, label, value, alt=False):
        """Larger metric row for the 3-page layout."""
        bg = (238, 243, 247) if alt else (248, 250, 253)
        pdf.set_fill_color(*bg)
        pdf.rect(x, y, w, 9, "F")
        pdf.set_xy(x + 5, y + 2)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(80, 90, 100)
        pdf.cell(w - 40, 5, label)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(40, 50, 60)
        pdf.cell(30, 5, value, align="R")

    def _place_image(self, pdf, path, x, y, w):
        """Place an image at (x, y) with width w. Returns the height in mm.
        Also advances the FPDF Y cursor to below the image."""
        from PIL import Image as PILImage
        img = PILImage.open(path)
        aspect = img.height / img.width
        h = w * aspect
        pdf.image(path, x=x, y=y, w=w)
        pdf.set_xy(x, y + h)
        return h

    def build_pdf(self):
        if not self.stats:
            self.compute_stats()
        if not self.chart_paths:
            self.generate_charts()
        summary = self._sanitize(self.write_summary())
        s = self.stats
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=18)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1: Header + Executive Summary + Key Metrics
        # ══════════════════════════════════════════════════════════════════════
        pdf.add_page()
        # Header
        pdf.set_xy(15, 20)
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(26, 107, 138)
        pdf.cell(0, 10, "PROJECT GUARDIAN")
        pdf.set_xy(15, 32)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(80, 100, 120)
        pdf.cell(0, 6, "Maritime Domain Awareness - Mission Report")
        pdf.set_draw_color(26, 107, 138)
        pdf.set_line_width(0.5)
        pdf.line(15, 42, 195, 42)
        # Session metadata
        pdf.set_xy(15, 47)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(120, 130, 140)
        meta = (f"Session: {s.get('session_id', 'N/A')}    |    "
                f"Generated: {s.get('timestamp', 'N/A')}    |    "
                f"Source: {self._sanitize(self.session_label)}")
        pdf.cell(0, 5, meta)
        # Executive Summary
        self._section_header(pdf, "EXECUTIVE SUMMARY", y=60)
        pdf.ln(4)  # gap between header and summary text
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 55, 65)
        pdf.multi_cell(180, 6, summary)
        pdf.ln(8)
        # Key Metrics (single-column list, readable)
        self._section_header(pdf, "KEY METRICS")
        pdf.ln(4)  # gap between header and metrics table
        avg_str = f"{s.get('avg_confidence', 0):.1%}" if isinstance(s.get("avg_confidence"), (int, float)) else "N/A"
        min_str = f"{s.get('min_confidence', 0):.1%}" if isinstance(s.get("min_confidence"), (int, float)) else "N/A"
        max_str = f"{s.get('max_confidence', 0):.1%}" if isinstance(s.get("max_confidence"), (int, float)) else "N/A"
        all_metrics = [
            ("Total Contacts",          str(s.get("total_detections", 0))),
            ("Unique Vessels Tracked",   str(s.get("unique_vessels", 0))),
            ("Frames Processed",        str(s.get("frames_processed", 0))),
            ("HIGH PRIORITY Contacts",  str(s.get("high_priority", 0))),
            ("PRIORITY Contacts",       str(s.get("priority", 0))),
            ("Civilian Vessels",         str(s.get("civilian", 0))),
            ("Threat Contact Ratio",    f"{s.get('threat_ratio', 0)}%"),
            ("Avg Confidence",          avg_str),
            ("Min / Max Confidence",    f"{min_str} / {max_str}"),
        ]
        if s.get("duration_stats"):
            ds = s["duration_stats"]
            all_metrics.append(("Avg Tracking Duration", f"{ds['avg_duration']}s"))
            all_metrics.append(("Max Tracking Duration", f"{ds['max_duration']}s"))
        start_y = pdf.get_y()
        row_h = 10
        for i, (label, value) in enumerate(all_metrics):
            self._metric_row_big(pdf, 15, start_y + i * row_h, 180, label, value, alt=(i % 2 == 0))

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 2: Charts (full-size, well-spaced)
        # ══════════════════════════════════════════════════════════════════════
        pdf.add_page()
        # Page 2 header
        pdf.set_xy(15, 15)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(26, 107, 138)
        pdf.cell(0, 7, "ANALYTICS")
        pdf.set_draw_color(26, 107, 138)
        pdf.set_line_width(0.3)
        pdf.line(15, 24, 195, 24)
        pdf.ln(15)  # gap between header and charts

        # Threat pie + class bar side-by-side (same size)
        pie_path = self.chart_paths.get("threat_pie")
        bar_path = self.chart_paths.get("class_bar")
        charts_y = pdf.get_y()
        placed_h = 0
        if pie_path and Path(pie_path).exists():
            h = self._place_image(pdf, pie_path, x=15, y=charts_y, w=88)
            placed_h = max(placed_h, h)
        if bar_path and Path(bar_path).exists():
            h2 = self._place_image(pdf, bar_path, x=108, y=charts_y, w=88)
            placed_h = max(placed_h, h2)
        pdf.set_y(charts_y + placed_h + 12)

        # Confidence histogram (centered, larger)
        hist_path = self.chart_paths.get("conf_hist")
        if hist_path and Path(hist_path).exists():
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(26, 107, 138)
            pdf.cell(0, 6, "Confidence Distribution")
            pdf.ln(8)  # advance past the title text
            hy = pdf.get_y()
            h3 = self._place_image(pdf, hist_path, x=25, y=hy, w=160)
            pdf.set_y(hy + h3 + 10)

        # Timeline (full width, only for video)
        tl_path = self.chart_paths.get("timeline")
        if tl_path and Path(tl_path).exists():
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(26, 107, 138)
            pdf.cell(0, 6, "Detection Timeline")
            pdf.ln(8)  # advance past the title text
            ty = pdf.get_y()
            h4 = self._place_image(pdf, tl_path, x=15, y=ty, w=180)
            pdf.set_y(ty + h4 + 5)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 3: Per-class breakdown table
        # ══════════════════════════════════════════════════════════════════════
        if s.get("per_class_conf"):
            pdf.add_page()
            # Page 3 header
            pdf.set_xy(15, 15)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(26, 107, 138)
            pdf.cell(0, 7, "PER-CLASS BREAKDOWN")
            pdf.set_draw_color(26, 107, 138)
            pdf.set_line_width(0.3)
            pdf.line(15, 24, 195, 24)
            pdf.ln(8)

            # Table header
            hdr_y = pdf.get_y()
            pdf.set_fill_color(230, 238, 242)
            pdf.rect(15, hdr_y, 180, 9, "F")
            pdf.set_xy(18, hdr_y + 1.5)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(26, 107, 138)
            pdf.cell(70, 6, "Vessel Class")
            pdf.cell(25, 6, "Count", align="C")
            pdf.cell(25, 6, "Avg Conf", align="C")
            pdf.cell(25, 6, "Min Conf", align="C")
            pdf.cell(25, 6, "Max Conf", align="C")
            pdf.ln(9)

            # Table rows
            pdf.set_font("Helvetica", "", 9)
            row_h = 8
            for i, (cls, vals) in enumerate(
                sorted(s["per_class_conf"].items(), key=lambda x: -x[1]["count"])
            ):
                row_bg = (245, 248, 250) if i % 2 == 0 else (250, 252, 254)
                y = pdf.get_y()
                pdf.set_fill_color(*row_bg)
                pdf.rect(15, y, 180, row_h, "F")
                pdf.set_xy(18, y + 1.5)
                pdf.set_text_color(50, 55, 65)
                pdf.cell(70, 5, cls.replace("_", " ").title())
                pdf.cell(25, 5, str(vals["count"]), align="C")
                pdf.cell(25, 5, f"{vals['avg_conf']:.1%}", align="C")
                pdf.cell(25, 5, f"{vals['min_conf']:.1%}", align="C")
                pdf.cell(25, 5, f"{vals['max_conf']:.1%}", align="C")
                pdf.ln(row_h)

        # Footer on last page
        pdf.set_auto_page_break(auto=False)
        pdf.set_xy(15, 285)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(180, 185, 190)
        pdf.cell(0, 4, "Project Guardian - SEDIC 2026 | Maritime Domain Awareness",
                 align="C", link="https://github.com/nurewnn/SEDIC2026")
        pdf.set_auto_page_break(auto=True, margin=18)
        return bytes(pdf.output(dest="S"))

    def build_pdf_bytes(self):
        return self.build_pdf()
