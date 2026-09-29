"""
ReportGenerator — Intelligence Pipeline for SEDIC 2026 Grand Finale.

Reads a detection CSV (produced by DetectionLogger), performs statistical
analysis with Pandas, generates charts with Matplotlib, writes an executive
summary via Ollama (local LLM, with template fallback), and stitches
everything into a mission-ready PDF using FPDF.

Usage:
    from report_generator import ReportGenerator
    rg = ReportGenerator("outputs/video_log.csv")
    pdf_bytes = rg.build_pdf()
    # pdf_bytes is raw bytes — write to file or serve via st.download_button
"""

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from fpdf import FPDF


# ── Chart styling (matches the GUI dark theme) ────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#070f18",
    "axes.facecolor":    "#0d1825",
    "axes.edgecolor":    "#1e3a4a",
    "axes.labelcolor":   "#b8d0dc",
    "xtick.color":       "#7192a5",
    "ytick.color":       "#7192a5",
    "text.color":        "#edf7fb",
    "axes.titlecolor":   "#35d7f3",
    "font.size":         9,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "figure.dpi":        150,
})

ACCENT     = "#35d7f3"
ACCENT_SOFT = "#9beeff"
GREEN      = "#29e58c"
RED        = "#ff555d"
ORANGE     = "#ffae4a"
DARK_BG    = "#0d1825"


class ReportGenerator:
    """Turns a detection CSV into a mission-ready PDF report."""

    def __init__(self, csv_path: str, session_label: str = ""):
        self.csv_path = Path(csv_path)
        self.session_label = session_label or self.csv_path.stem
        self.df: pd.DataFrame | None = None
        self.stats: dict = {}
        self.chart_paths: dict = {}   # name → file path
        self._chart_dir = Path("outputs/charts")
        self._chart_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    def load_csv(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        return self.df

    # ── 2. Statistics ──────────────────────────────────────────────────────────
    def compute_stats(self) -> dict:
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

        avg_conf = round(df["confidence"].mean(), 3) if "confidence" in df.columns else 0
        min_conf = round(df["confidence"].min(), 3) if "confidence" in df.columns else 0
        max_conf = round(df["confidence"].max(), 3) if "confidence" in df.columns else 0

        # Per-class confidence
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

        # Detection duration stats (if present)
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
            "session_id":          session_id,
            "total_detections":     total,
            "unique_vessels":      unique_vessels,
            "frames_processed":    frames_processed,
            "class_counts":        class_counts,
            "threat_counts":       threat_counts,
            "high_priority":       high_priority,
            "priority":            priority,
            "civilian":            civilian,
            "monitor":             monitor,
            "small_craft":         small_craft,
            "threat_total":        threat_total,
            "threat_ratio":         threat_ratio,
            "avg_confidence":      avg_conf,
            "min_confidence":      min_conf,
            "max_confidence":      max_conf,
            "per_class_conf":      per_class_conf,
            "duration_stats":      duration_stats,
            "timestamp":           datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }
        return self.stats

    # ── 3. Charts ──────────────────────────────────────────────────────────────
    def generate_charts(self) -> dict:
        if self.df is None:
            self.load_csv()
        if self.df.empty:
            return {}

        df = self.df
        charts = {}

        # Chart 1: Threat distribution (pie)
        threat_cols = ["HIGH PRIORITY", "PRIORITY", "MONITOR", "SMALL CRAFT", "CIVILIAN"]
        threat_vals = [self.stats["threat_counts"].get(k, 0) for k in threat_cols]
        threat_labels = [k for k, v in zip(threat_cols, threat_vals) if v > 0]
        threat_data = [v for v in threat_vals if v > 0]
        colors = [RED, ORANGE, ORANGE, "#ccaa00", GREEN][:len(threat_data)]

        if threat_data:
            fig, ax = plt.subplots(figsize=(5, 4))
            wedges, texts, autotexts = ax.pie(
                threat_data, labels=threat_labels, autopct="%1.0f%%",
                colors=colors, startangle=90,
                textprops={"fontsize": 8},
                wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5},
            )
            for t in autotexts:
                t.set_color("#000000")
                t.set_fontweight("bold")
                t.set_fontsize(8)
            ax.set_title("Threat Level Distribution", fontweight="bold", pad=12)
            fig.tight_layout()
            path = str(self._chart_dir / "threat_pie.png")
            fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            charts["threat_pie"] = path

        # Chart 2: Per-class detection counts (horizontal bar)
        class_counts = df["class_name"].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, max(3, len(class_counts) * 0.45)))
        bars = ax.barh(class_counts.index, class_counts.values, color=ACCENT, edgecolor=DARK_BG, height=0.6)
        ax.set_xlabel("Detection Count", fontsize=9)
        ax.set_title("Detections by Vessel Class", fontweight="bold", pad=10)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(int(w)), va="center", fontsize=8, color=ACCENT_SOFT)
        fig.tight_layout()
        path = str(self._chart_dir / "class_bar.png")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        charts["class_bar"] = path

        # Chart 3: Confidence distribution (histogram)
        if "confidence" in df.columns and not df["confidence"].empty:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(df["confidence"], bins=20, color=ACCENT, edgecolor=DARK_BG, alpha=0.85)
            ax.set_xlabel("Confidence Score", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.set_title("Confidence Distribution", fontweight="bold", pad=10)
            ax.axvline(df["confidence"].mean(), color=ORANGE, linestyle="--", linewidth=1.2,
                       label=f"Mean: {df['confidence'].mean():.3f}")
            ax.legend(fontsize=8, facecolor=DARK_BG, edgecolor="#1e3a4a", labelcolor=ACCENT_SOFT)
            fig.tight_layout()
            path = str(self._chart_dir / "conf_hist.png")
            fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            charts["conf_hist"] = path

        # Chart 4: Detections over time (per frame)
        if "frame_id" in df.columns and df["frame_id"].nunique() > 1:
            per_frame = df.groupby("frame_id").size()
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(per_frame.index, per_frame.values, color=ACCENT, linewidth=1.2, alpha=0.9)
            ax.fill_between(per_frame.index, per_frame.values, alpha=0.15, color=ACCENT)
            ax.set_xlabel("Frame ID", fontsize=9)
            ax.set_ylabel("Detections", fontsize=9)
            ax.set_title("Detection Timeline", fontweight="bold", pad=10)
            fig.tight_layout()
            path = str(self._chart_dir / "timeline.png")
            fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            charts["timeline"] = path

        self.chart_paths = charts
        return charts

    # ── 4. Executive Summary (Ollama with template fallback) ───────────────────
    def write_summary(self) -> str:
        s = self.stats
        if s.get("total_detections", 0) == 0:
            return ("No vessel detections were recorded during this session. "
                    "The system maintained operational readiness throughout.")

        # Build the stats context for the LLM
        ctx = (
            f"Session ID: {s['session_id']}\n"
            f"Total detections: {s['total_detections']}\n"
            f"Unique vessels tracked: {s['unique_vessels']}\n"
            f"Frames processed: {s['frames_processed']}\n"
            f"High Priority threats: {s['high_priority']}\n"
            f"Priority threats: {s['priority']}\n"
            f"Civilian vessels: {s['civilian']}\n"
            f"Threat ratio: {s['threat_ratio']}%\n"
            f"Average confidence: {s['avg_confidence']}\n"
            f"Class breakdown: {s['class_counts']}\n"
        )

        # Try Ollama (local, offline)
        try:
            import httpx
            prompt = (
                "You are a maritime domain awareness analyst. Write a concise "
                "3-4 sentence executive summary for a duty officer based on the "
                "following detection session statistics. Be factual, direct, and "
                "flag any HIGH PRIORITY threats. Do not use markdown.\n\n"
                f"{ctx}"
            )
            resp = httpx.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 300},
                },
                timeout=30.0,
            )
            data = resp.json()
            summary = data.get("response", "").strip()
            if summary:
                return summary
        except Exception:
            pass  # fall through to template

        # Template fallback — always works, no dependency
        high = s["high_priority"]
        prio = s["priority"]
        total = s["total_detections"]
        unique = s["unique_vessels"]
        threat_pct = s["threat_ratio"]
        avg_c = s["avg_confidence"]

        if high > 0:
            threat_str = f"{high} HIGH PRIORITY threat(s) detected - immediate action required."
        elif prio > 0:
            threat_str = f"{prio} PRIORITY threat(s) detected - monitor and report to command."
        else:
            threat_str = "No priority-level threats detected. All contacts classified as civilian or small craft."

        return (
            f"During this session, {total} detections were recorded across {unique} unique "
            f"vessels over {s['frames_processed']} processed frames. {threat_str} "
            f"The threat ratio was {threat_pct}% of all detections, with an average model "
            f"confidence of {avg_c:.1%}. The system maintained operational readiness throughout."
        )

    # ── 5. Build PDF ───────────────────────────────────────────────────────────
    @staticmethod
    def _sanitize_pdf_text(text: str) -> str:
        """Replace characters outside Latin-1 (Helvetica's range) with
        ASCII equivalents so FPDF doesn't crash on Unicode."""
        replacements = {
            "\u2014": "-",   # em dash
            "\u2013": "-",   # en dash
            "\u2018": "'",   # left single quote
            "\u2019": "'",   # right single quote
            "\u201c": '"',   # left double quote
            "\u201d": '"',   # right double quote
            "\u2026": "...", # ellipsis
            "\u00b0": " deg",# degree sign
            "\u2122": "(tm)",# trademark
            "\u00a9": "(c)", # copyright
            "\u00ae": "(r)", # registered
        }
        for uni, asc in replacements.items():
            text = text.replace(uni, asc)
        # Strip any remaining non-Latin-1 chars
        return text.encode("latin-1", "replace").decode("latin-1")

    def build_pdf(self) -> bytes:
        """Generate the full PDF report. Returns raw bytes."""
        if not self.stats:
            self.compute_stats()
        if not self.chart_paths:
            self.generate_charts()

        summary = self._sanitize_pdf_text(self.write_summary())
        s = self.stats

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)

        # ── Title page ──────────────────────────────────────────────────────
        pdf.add_page()
        pdf.set_fill_color(7, 15, 24)
        pdf.rect(0, 0, 210, 297, "F")  # full A4 dark background

        pdf.set_xy(15, 30)
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(53, 215, 243)  # accent cyan
        pdf.multi_cell(180, 10, "PROJECT GUARDIAN")

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(155, 238, 255)  # accent_soft
        pdf.set_xy(15, 42)
        pdf.multi_cell(180, 6, "Maritime Domain Awareness - Mission Report")

        pdf.set_draw_color(53, 215, 243)
        pdf.set_line_width(0.5)
        pdf.line(15, 52, 195, 52)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(113, 146, 165)
        pdf.set_xy(15, 56)
        pdf.multi_cell(180, 5,
                        f"Session: {s.get('session_id', 'N/A')}\n"
                        f"Generated: {s.get('timestamp', 'N/A')}\n"
                        f"Source: {self._sanitize_pdf_text(self.session_label)}")

        # ── Executive Summary ───────────────────────────────────────────────
        pdf.set_xy(15, 75)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(53, 215, 243)
        pdf.multi_cell(180, 7, "EXECUTIVE SUMMARY")
        pdf.ln(2)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(237, 247, 251)
        pdf.multi_cell(180, 5.5, summary)
        pdf.ln(3)

        # ── Key Metrics Table ───────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(53, 215, 243)
        pdf.multi_cell(180, 7, "KEY METRICS")
        pdf.ln(1)

        metrics = [
            ("Total Detections",      str(s.get("total_detections", 0))),
            ("Unique Vessels Tracked", str(s.get("unique_vessels", 0))),
            ("Frames Processed",      str(s.get("frames_processed", 0))),
            ("High Priority Threats", str(s.get("high_priority", 0))),
            ("Priority Threats",      str(s.get("priority", 0))),
            ("Civilian Vessels",       str(s.get("civilian", 0))),
            ("Threat Ratio",           f"{s.get('threat_ratio', 0)}%"),
            ("Avg Confidence",         f"{s.get('avg_confidence', 0):.3f}" if isinstance(s.get("avg_confidence"), (int, float)) else str(s.get("avg_confidence", 0))),
            ("Min Confidence",         f"{s.get('min_confidence', 0):.3f}" if isinstance(s.get("min_confidence"), (int, float)) else str(s.get("min_confidence", 0))),
            ("Max Confidence",         f"{s.get('max_confidence', 0):.3f}" if isinstance(s.get("max_confidence"), (int, float)) else str(s.get("max_confidence", 0))),
        ]

        if s.get("duration_stats"):
            ds = s["duration_stats"]
            metrics.append(("Avg Tracking Duration", f"{ds['avg_duration']}s"))
            metrics.append(("Max Tracking Duration", f"{ds['max_duration']}s"))

        pdf.set_font("Helvetica", "", 10)
        for i, (label, value) in enumerate(metrics):
            row_bg = (4, 14, 24) if i % 2 == 0 else (8, 20, 32)
            y = pdf.get_y()
            pdf.set_fill_color(*row_bg)
            pdf.rect(15, y, 180, 7, "F")
            pdf.set_text_color(155, 238, 255)
            pdf.set_xy(18, y + 0.8)
            pdf.cell(120, 5, label)
            pdf.set_text_color(237, 247, 251)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(50, 5, value, align="R")
            pdf.set_font("Helvetica", "", 10)
            pdf.ln(7)

        pdf.ln(3)

        # ── Charts ───────────────────────────────────────────────────────────
        if self.chart_paths:
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(53, 215, 243)
            pdf.multi_cell(180, 7, "ANALYTICS")
            pdf.ln(2)

            for title, key in [
                ("Threat Level Distribution", "threat_pie"),
                ("Detections by Vessel Class", "class_bar"),
                ("Confidence Distribution",    "conf_hist"),
                ("Detection Timeline",          "timeline"),
            ]:
                path = self.chart_paths.get(key)
                if path and Path(path).exists():
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.set_text_color(155, 238, 255)
                    pdf.multi_cell(180, 6, title)
                    pdf.ln(1)
                    # FPDF image with max width 180mm, preserve aspect
                    img_w = 170
                    pdf.image(path, x=20, w=img_w)
                    pdf.ln(5)

        # ── Per-Class Breakdown Table ───────────────────────────────────────
        if s.get("per_class_conf"):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(53, 215, 243)
            pdf.multi_cell(180, 7, "PER-CLASS BREAKDOWN")
            pdf.ln(1)

            # Header
            pdf.set_fill_color(8, 30, 44)
            pdf.rect(15, pdf.get_y(), 180, 8, "F")
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(53, 215, 243)
            pdf.set_xy(18, pdf.get_y() + 1)
            pdf.cell(60, 6, "Vessel Class")
            pdf.cell(25, 6, "Count", align="C")
            pdf.cell(30, 6, "Avg Conf", align="C")
            pdf.cell(30, 6, "Min Conf", align="C")
            pdf.cell(30, 6, "Max Conf", align="C")
            pdf.ln(8)

            pdf.set_font("Helvetica", "", 9)
            for i, (cls, vals) in enumerate(sorted(s["per_class_conf"].items(), key=lambda x: -x[1]["count"])):
                row_bg = (4, 14, 24) if i % 2 == 0 else (8, 20, 32)
                y = pdf.get_y()
                pdf.set_fill_color(*row_bg)
                pdf.rect(15, y, 180, 7, "F")
                pdf.set_text_color(237, 247, 251)
                pdf.set_xy(18, y + 0.8)
                pdf.cell(60, 5, cls.replace("_", " ").title())
                pdf.cell(25, 5, str(vals["count"]), align="C")
                pdf.cell(30, 5, f"{vals['avg_conf']:.3f}", align="C")
                pdf.cell(30, 5, f"{vals['min_conf']:.3f}", align="C")
                pdf.cell(30, 5, f"{vals['max_conf']:.3f}", align="C")
                pdf.ln(7)

        # ── Footer ───────────────────────────────────────────────────────────
        pdf.set_y(-25)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(113, 146, 165)
        pdf.cell(180, 5, "Project Guardian - SEDIC 2026 | Generated by Intelligence Pipeline",
                 align="C", link="https://github.com/nurewnn/SEDIC2026")

        return pdf.output(dest="S").encode("latin-1") if isinstance(pdf.output(dest="S"), str) else bytes(pdf.output(dest="S"))

    def build_pdf_bytes(self) -> bytes:
        """Alias for build_pdf() — returns raw bytes for st.download_button."""
        return self.build_pdf()
