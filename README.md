<div align="center">

# 🛡️ Project Guardian
### Advanced Maritime Domain Awareness (MDA) System
**SEDIC 2026 — Visual Track · Phase 1 Preliminary Qualifier**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-00FFFF?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.61-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/Dataset-CC%20BY%204.0-green?style=flat-square)

</div>

---

## 📌 Overview

Project Guardian is a real-time Maritime Domain Awareness (MDA) system for detecting and classifying vessels from image and video inputs. Built for SEDIC 2026 Visual Track, the system uses a fine-tuned **YOLOv8s** model trained via transfer learning on a merged open-source maritime dataset.

The final model (`run_clean_dedup_v1 / best.pt`) achieves a pooled military-class recall of **79.8%** on a corrected, leak-free validation set of 797 images and 2,645 instances.

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
├── detector.py             # YOLOv8 inference wrapper
├── log_writer.py           # Detection log (CSV) writer
├── run_qualifier.py        # Standalone qualifier video script
├── utils/
│   ├── __init__.py
│   └── colours.py          # Threat colour + level mapping
├── models/
│   └── best.pt         # Trained model weights (not in repo — see below)
├── data/
│   └── qualifier_clip.mp4  # Qualifier video (place here)
├── outputs/
│   └── *.csv               # Detection logs (auto-generated)
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

### 5. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🖥️ GUI Features

| Feature | Description |
|---|---|
| **Image mode** | Upload JPG/PNG → instant detection with bounding boxes |
| **Video mode** | Upload MP4 → real-time frame-by-frame processing |
| **Qualifier Video mode** | Run model on official qualifier clip → generate submission log |
| **Military alert** | 🔴 Pulsing red alert for foreign military / 🟠 Orange alert for local military |
| **Metric cards** | Live count of Total / Military / Civilian / Threats |
| **Confidence slider** | Adjust detection threshold (0.01 – 0.95) |
| **Model diagnostics** | Toggle debug panel showing model info and low-threshold probe |
| **CSV download** | One-click download of detection log after each run |

---

## 📋 Detection Log Format

All detection runs produce a CSV log with these columns:

| Column | Description |
|---|---|
| `timestamp` | UTC timestamp (ISO 8601) |
| `frame_id` | Frame number (0 for images) |
| `class_name` | Detected vessel class |
| `threat_level` | CIVILIAN / SMALL CRAFT / MONITOR / PRIORITY / HIGH PRIORITY |
| `confidence` | Model confidence (0.000 – 1.000) |
| `x1, y1, x2, y2` | Bounding box coordinates (pixels) |

### Example

```csv
timestamp,frame_id,class_name,threat_level,confidence,x1,y1,x2,y2
2026-08-14T08:32:11.042Z,0,local_military_ship,PRIORITY,0.882,104,87,743,498
2026-08-14T08:32:11.042Z,0,container_ship,CIVILIAN,0.951,210,300,890,640
```

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

### Video Demonstration Checklist (15 points)

- [ ] Max 5 minutes, hosted via YouTube
- [ ] Demonstrate model functionality and results
- [ ] Highlight confirmed strengths (foreign military detection, 93.4% recall)
- [ ] Briefly acknowledge `local_military_ship` limitation and its root cause

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
| 2 | `local_military_ship` recall (66.3%) driven by KD Maharaja Lela data scarcity (2 unique images) | Root cause identified |
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
