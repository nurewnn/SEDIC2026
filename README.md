<div align="center">

# 🛡️ Project Guardian
### Advanced Maritime Domain Awareness (MDA) System
**SEDIC 2026 — Visual Track · Grand Finale Deployment**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-00FFFF?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.61-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/Dataset-CC%20BY%204.0-green?style=flat-square)

</div>

---

## 📌 Overview

Project Guardian is a real-time Maritime Domain Awareness (MDA) system for detecting and classifying vessels from image and video inputs. Built for SEDIC 2026 Visual Track, the system uses a fine-tuned **YOLOv8s** model trained via transfer learning on a merged open-source maritime dataset.

The final model (`run_clean_dedup_v1 / best.pt`) achieves a pooled military-class recall of **79.8%** on a strictly corrected, leak-free validation set. The deployed application features an offline LLM-driven intelligence pipeline, resilient object tracking, and high-resolution inference to ensure absolute operational readiness.

---

## 🛡️ Grand Finale Defense & Capabilities

Project Guardian has been engineered not just for academic benchmarks, but for real-world tactical deployment. It directly addresses the SEDIC 2026 Multi-Sector Jury expectations:

### 🎖️ Armed Services (Operational Utility)
- **High-Resolution Inference (`imgsz=1280`):** Standard models downscale and lose distant horizon threats. Guardian forces high-resolution analysis to identify critical distant assets (e.g., small local military ships).
- **Automated Intelligence Summaries:** Field commanders require actionable intelligence, not raw video. Our system automatically scrubs video feeds to generate Mission Report PDFs detailing the top 10 peak threat frames, sorted by severity, with automated Gantt chart timelines.

### 🎓 Academia (Scientific & Technical Rigor)
- **Data Integrity Validation:** We independently identified and corrected a 54% data leakage contamination in the original dataset, ensuring our 79.8% recall is a scientifically robust and trustworthy metric.
- **Fail-Safe Architecture:** The real-time tracking system is built with a silent, automatic fallback. If the ByteTrack algorithm encounters a configuration error or noise, the system gracefully degrades to raw frame-by-frame predictions, guaranteeing zero downtime.

### 💼 Defense Industry (Scalability & Integration)
- **Air-Gapped & Cloud-Independent:** The entire pipeline—from YOLOv8 inference to the Ollama LLM summarization—runs 100% locally. It does not require Wi-Fi, ensuring absolute security and offline independence for naval vessels.
- **C2 System Integration:** Project Guardian exports granular, timestamped, and tracked CSV detection logs, enabling frictionless integration with existing Command and Control (C2) or Maritime Traffic APIs.

---

## 📊 Rubric Summary

| Criterion | Max | Target | Our Result |
|---|---|---|---|
| Mandatory Classification | 30 | Civilian, Small Craft, Military — multi-angle | ✅ Met |
| Performance Benchmark | 25 | > 80% recall on military/threat classes | 79.8% — 0.2 pts short |
| Competitive Advantage | 10 | Local (Malaysian) vs. foreign military distinction | ⚠️ Partial |
| Technical Brief | 20 | Dataset, architecture, classification logic | ✅ Complete |
| Video Demonstration | 15 | Max 5-min, YouTube | See checklist |

---

## 🎯 Detection Classes

| Category | Classes | Threat Level |
|---|---|---|
| 🟢 Civilian | `container_ship`, `tanker`, `cargo`, `passenger_ferry` | CIVILIAN |
| 🟡 Small Craft | `yacht`, `speedboat`, `fishing_boat` | SMALL CRAFT |
| 🟠 Monitor | `patrol_boat` | MONITOR *(supplementary)* |
| 🟠 Military (Local) | `local_military_ship` | PRIORITY |
| 🔴 Military (Foreign) | `foreign_military_ship` | HIGH PRIORITY |

> **Note:** `patrol_boat` is treated as a supplementary category (non-Malaysian stock imagery) and is excluded from the Performance Benchmark calculation.

---

## 📈 Performance Results

### Military Class Recall (Primary Benchmark)

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| `foreign_military_ship` | 0.896 | **0.934** | 0.952 |
| `local_military_ship` | 0.728 | **0.663** | 0.716 |
| `patrol_boat` *(supplementary)* | 0.523 | 0.538 | 0.599 |
| **Pooled Military Recall** | — | **79.8%** | — |

### Full Per-Class Results

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| `container_ship` | 0.906 | 0.908 | 0.942 |
| `tanker` | 0.921 | 0.840 | 0.899 |
| `cargo` | 0.753 | 0.759 | 0.812 |
| `passenger_ferry` | 0.851 | 0.860 | 0.901 |
| `yacht` | 0.781 | 0.761 | 0.853 |
| `speedboat` | 0.639 | 0.433 | 0.490 |
| `fishing_boat` | 0.792 | 0.558 | 0.680 |
| `tugboat` | 0.620 | 0.459 | 0.535 |

> ⚠️ An earlier figure of **88.7%** was computed on a validation set later found to be **54% contaminated** with training-set duplicates. The corrected figure of 79.8% is the trustworthy result.

---

## 🔍 Root Cause of the Performance Gap

`foreign_military_ship` recall (93.4%) comfortably exceeds the 80% benchmark on its own. The pooled figure is held below threshold specifically by `local_military_ship` (66.3%), due to two identified causes:

### 1. Data Scarcity — KD Maharaja Lela
A newly delivered RMN frigate with only **2 unique source photographs** in the entire dataset. Live testing confirmed zero detections of this vessel at any confidence threshold — a data-driven blind spot, not a model architecture issue.

### 2. Crowded Multi-Vessel Scene Detection Loss
Images containing multiple RMN vessels in close proximity show partial detection loss. This is a known limitation of single-stage detectors on densely packed small objects, confirmed via dataset instance-count checks and GUI testing.

**Assessment:** The 0.2-point gap is closeable with targeted additional data for `local_military_ship`, and does not reflect a fundamental architecture or methodology weakness.

---

## 🏆 Competitive Advantage

Local vs. foreign military distinction is implemented directly in the primary detector via **two separate trained classes** (`foreign_military_ship`, `local_military_ship`) rather than a secondary classification stage.

- Foreign asset identification: **strong** (93.4% recall)
- Local asset identification: **functional but weaker** (66.3% recall) — root cause documented above

> A stage-2 crop-based nationality classifier was scoped (805 military crops extracted) but deprioritised in favour of Performance Benchmark optimisation.

---

## 🗂️ Project Structure

```
Project/
├── app.py                  # Main Streamlit GUI
├── detector.py             # YOLOv8 inference + object tracking wrapper
├── log_writer.py           # Detection log (CSV) writer with session/tracking data
├── report_generator.py     # Intelligence pipeline: CSV → Pandas → Charts → Ollama → PDF
├── run_qualifier.py        # Standalone qualifier video script
├── utils/
│   ├── __init__.py
│   └── colours.py          # Threat colour + level mapping
├── models/
│   └── best.pt         # Trained model weights (not in repo — see below)
├── data/
│   └── qualifier_clip.mp4  # Qualifier video (place here)
├── outputs/
│   ├── *.csv               # Detection logs (auto-generated)
│   └── charts/             # Auto-generated analytics charts (PNG)
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/syardnl/SEDIC2026.git
cd SEDIC2026
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download model weights

`best.pt` in the `models/` folder:

```
models/best.pt
```

> Contact the team for the Drive link.

### 5. Setup Local LLM (Ollama)

To generate the AI-driven Mission Reports (PDF), you must have Ollama running locally.

1. Download and install [Ollama](https://ollama.com/)
2. Open your terminal and pull the required model:
   ```bash
   ollama run llama3.1:8b
   ```
*(The app will gracefully fall back to a hardcoded template if Ollama is not detected, but the AI summary will be unavailable.)*

### 6. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🖥️ GUI Features

| Feature | Description |
|---|---|
| **Image mode** | Upload JPG/PNG → instant detection with bounding boxes |
| **Video mode** | Upload MP4 → real-time frame-by-frame processing with object tracking |
| **Qualifier Video mode** | Run model on official qualifier clip → generate submission log |
| **Military alert** | 🔴 Pulsing red alert for foreign military / 🟠 Orange alert for local military |
| **High-Res Inference** | Forces `imgsz=1280` inference resolution to detect tiny, distant vessels |
| **Metric cards** | Live count of Total / Military / Civilian / Threats |
| **Object tracking** | ByteTrack persistent vessel IDs with automatic fail-safe fallback to raw predictions |
| **Confidence slider** | Adjust detection threshold (0.01 – 0.95) |
| **Model diagnostics** | Toggle debug panel showing model info and low-threshold probe |
| **CSV download** | One-click download of detection log (with session_id, vessel_id, detection_duration) after each run |
| **Mission Report (PDF)** | Intelligence pipeline: CSV → Pandas stats → Matplotlib charts → Ollama summary → PDF (with up to 10 prioritized peak threat frames and layout wrap protection) |

---

## 📋 Outputs & Automated Reports

### 1. Detection Log (CSV)
All detection runs produce a CSV log with granular frame-by-frame data:

| Column | Description |
|---|---|
| `session_id` | Unique session identifier (UUID hex) |
| `timestamp` | UTC timestamp (ISO 8601) |
| `frame_id` | Frame number (0 for images) |
| `vessel_id` | Object-tracking ID (ByteTrack, stable across frames) |
| `class_name` | Detected vessel class |
| `threat_level` | CIVILIAN / SMALL CRAFT / MONITOR / PRIORITY / HIGH PRIORITY |
| `confidence` | Model confidence (0.000 – 1.000) |
| `x1, y1, x2, y2` | Bounding box coordinates (pixels) |
| `detection_duration` | Seconds this vessel was continuously tracked |

### 2. Incident Report (PDF)
Generated automatically from **Image Mode**. This is a rapid-response brief containing:
- AI-generated summary of the situation
- The annotated high-resolution source image
- Breakdown of detected assets and threat levels

### 3. Mission Report (PDF)
Generated automatically from **Video Mode**. This is a comprehensive post-mission intelligence document containing:
- Overall mission summary and executive AI briefing
- Key metrics and tracking statistics
- **Gantt Tracking Chart:** Visual timeline showing exactly when and for how long each vessel was tracked
- **Top 10 Peak Threat Frames:** High-resolution frame grabs of the most critical moments, sorted by threat severity (Foreign Military prioritized first)

---

## 🎬 Qualifier Video — Submission

1. Place the official qualifier clip at `data/qualifier_clip.mp4`
2. Open the app → select **Qualifier Video** mode
3. Click **▶ RUN AND GENERATE LOG**
4. Download `qualifier_detection_log.csv` from the app
5. Submit the CSV as part of the competition package

Or run standalone (no GUI needed):

```bash
python run_qualifier.py
```

Output saved to `outputs/qualifier_detection_log.csv`.


---

## 🏗️ Model

| Detail | Value |
|---|---|
| Architecture | YOLOv8s (over nano — stronger recall ceiling on minority classes) |
| Pretrained checkpoint | Ultralytics COCO |
| Input size | 640 × 640 |
| Classes | 11 |
| Epochs | 150 (149 completed, 6.09 hours) |
| Batch size | 32 |
| Patience | 40 |
| LR schedule | `cos_lr`, lr0=0.01 (auto-optimised) |
| Close mosaic | Last 15 epochs |
| Compute | Google Colab Pro, Tesla T4 GPU |
| Validation set | 797 images, 2,645 instances (deduplicated) |

Multi-angle handling uses a merged frontal + aerial dataset with standard YOLO augmentation (mosaic, scale, rotation) — no custom loss functions or rotated-bounding-box methods.

---

## 🗃️ Dataset & Data Integrity

### Sources

- SeaShips, Sea Vessels, Tanker, Speedboat, Container Ship, Fishing Boat, Cruise, and Local Military Ship datasets (Roboflow Universe, CC BY 4.0)
- ShipRSImageNet V1.1 (academic use only)
- Manually curated RMN and foreign military imagery — Wikimedia Commons, ShipSpotting.com, seaforces.org, official RMN/MOD channels

### Integrity Issues Found and Resolved

| Issue | Scope | Resolution |
|---|---|---|
| Train/validation leakage | 1,118 of 2,057 validation images (54%) were training-set duplicates | Fixed — deduplicated by content hash, re-split fresh |
| Watermarked / collage stock images | 713 images, concentrated in `foreign_military_ship`, `yacht`, `patrol_boat` | Accepted, documented — retained given time constraints |
| Getty Images-sourced filenames | Multiple images retain original Getty filenames | Flagged — licensing origin to confirm before public distribution |

---

## ⚠️ Known Limitations

| # | Limitation | Status |
|---|---|---|
| 1 | Pooled military recall (79.8%) is 0.2 pts short of the 80% benchmark | Documented |
| 2 | `local_military_ship` recall (66.3%) driven by KD Maharaja Lela data scarcity (2 unique images) | Root cause identified (Live app mitigated via `imgsz=1280` high-res inference) |
| 3 | Crowded multi-vessel scenes show partial detection loss | Root cause identified |
| 4 | 713 watermarked/collage images retained in training data | Accepted trade-off |
| 5 | Getty Images filenames present; licensing not independently confirmed | Flagged |
| 6 | Aerial coverage thinner than frontal across the dataset | Documented |
| 7 | Stage-2 nationality classifier scoped, not completed (805 crops extracted) | Deprioritised |
| 8 | Train/validation leakage (54% of original val set) — **fully corrected** | ✅ Resolved |

---

## 🚀 Inference

```python
from ultralytics import YOLO

model = YOLO('models/guardian.pt')
results = model.predict('image.jpg', conf=0.15)
```

> **Confidence threshold:** 0.15 — confirm this matches the deployed GUI's configured value before final submission.

### Class Output Order

```
container_ship, tanker, cargo, passenger_ferry, yacht, speedboat,
fishing_boat, foreign_military_ship, local_military_ship, tugboat, patrol_boat
```

---

## 📦 Requirements

```
streamlit>=1.61
ultralytics>=8.0
opencv-python
pillow
httpx
numpy
pandas
matplotlib
fpdf2
```

Install all:

```bash
pip install -r requirements.txt
```

---

## 👥 Team

| Role | Responsibility |
|---|---|
| #1 & #2 — Model & Multi-view | YOLOv8 architecture, training pipeline, recall optimisation |
| #3 — Data Engineer | Dataset sourcing, curation, labelling, class taxonomy |
| #4 — GUI & Detection Log | Streamlit GUI, inference pipeline, detection log generation |
| #5 — Data Analyst / Storyteller | Technical brief, confusion matrix, submission checklist, demo video |

---

<div align="center">
Built for <strong>SEDIC 2026</strong> · Visual Track — Advanced Maritime Domain Awareness
</div>
