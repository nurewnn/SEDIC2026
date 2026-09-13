import time, base64, io
from pathlib import Path
import streamlit as st
import cv2, tempfile
import numpy as np
from PIL import Image
from detector   import GuardianDetector
from log_writer import DetectionLogger

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ICON_DIR   = ASSETS_DIR / "icon"

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Guardian — MDA",
    layout="wide",
    page_icon=str(ASSETS_DIR / "icon.svg"),
    initial_sidebar_state="collapsed",
)

# ── Session state defaults ───────────────────────────────────────────────
if "detect_mode" not in st.session_state:
    st.session_state.detect_mode = "Image"

MODEL_PATH = "models/best.pt"

# ── Image helpers ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

@st.cache_data(show_spinner=False)
def _svg_b64(filename: str) -> str:
    return _b64(ICON_DIR / filename)

def svg_icon(filename: str, css_class: str = "ui-svg-icon", alt: str = "") -> str:
    return (
        f'<img src="data:image/svg+xml;base64,{_svg_b64(filename)}" '
        f'class="{css_class}" alt="{alt}">'
    )

@st.cache_data(show_spinner=False)
def _b64_compressed(path: Path, max_width: int = 1920, quality: int = 85) -> tuple[str, str]:
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        ratio = max_width / img.width
        img   = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode(), "image/jpeg"

BG_B64,   BG_MIME   = _b64_compressed(ASSETS_DIR / "bg.jpg",          max_width=1920, quality=85)
SHIP_B64             = _b64(ASSETS_DIR / "blend_sidebar.png")
SHIP_MIME            = "image/png"
ICON_B64             = _b64(ASSETS_DIR / "icon.svg")
IMG_DET_ICON_B64      = _svg_b64("image_detection.svg")
VID_DET_ICON_B64      = _svg_b64("video_detection.svg")
DOWNLOAD_ICON_B64     = _svg_b64("download.svg")

# ── CSS variables (bg photo) ──────────────────────────────────────────────
st.markdown(f"""
<style>
:root {{
    --bg-photo: url("data:{BG_MIME};base64,{BG_B64}");
}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
:root {{
    --svg-image-detection:url("data:image/svg+xml;base64,{IMG_DET_ICON_B64}");
    --svg-video-detection:url("data:image/svg+xml;base64,{VID_DET_ICON_B64}");
    --svg-download:url("data:image/svg+xml;base64,{DOWNLOAD_ICON_B64}");
}}
</style>
""", unsafe_allow_html=True)

# ── Main CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root{
    --bg-deep:#030812;
    --bg-panel:rgba(6,15,26,.94);
    --accent:#35d7f3;
    --accent-soft:#9beeff;
    --text-primary:#edf7fb;
    --text-secondary:#b8d0dc;
    --text-muted:#7192a5;
    --green:#29e58c;
    --red:#ff555d;
    --orange:#ffae4a;
    --shadow:0 12px 32px rgba(0,0,0,.38);
}

html, body, [data-testid="stAppViewContainer"]{
    color:var(--text-primary);
    font-family:'Courier New', monospace;
}

[data-testid="stAppViewContainer"]{
    background-image:
        linear-gradient(90deg,rgba(1,6,12,.84) 0%,rgba(2,8,15,.66) 20%,rgba(2,8,15,.52) 54%,rgba(1,6,12,.68) 100%),
        linear-gradient(180deg,rgba(2,8,15,.40),rgba(1,5,10,.64)),
        var(--bg-photo);
    background-size:cover;
    background-position:center;
    background-attachment:fixed;
    background-repeat:no-repeat;
}

[data-testid="stHeader"]{background:transparent !important;}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none !important;}

.block-container{
    max-width:100% !important;
    padding:.55rem .70rem .80rem .70rem !important;
}
div[data-testid="stHorizontalBlock"]{gap:.65rem !important;}

/* ═══ SIDEBAR ══════════════════════════════════════════════════════════ */
/* Make the left Streamlit column stretch to the same height as the main
   content column.  The sidebar then fills that column instead of being
   locked to only one viewport (100vh). */
div[data-testid="stHorizontalBlock"]:has(.st-key-custom_sidebar){
    align-items:stretch !important;
}

div[data-testid="stHorizontalBlock"]:has(.st-key-custom_sidebar)
> div[data-testid="stColumn"]:has(.st-key-custom_sidebar){
    align-self:stretch !important;
}

div[data-testid="stColumn"]:has(.st-key-custom_sidebar)
> div[data-testid="stVerticalBlock"]{
    height:100% !important;
}

.st-key-custom_sidebar{
    min-height:calc(100vh - 1rem);
    height:100%;
    overflow:visible;
    position:relative;
    top:0;
    box-sizing:border-box;
    padding:1.05rem .9rem 1rem .9rem;
    border:1px solid rgba(53,215,243,.42);
    border-radius:24px;

    /* SAME image as main background, only the gradient is different */
    background-image:
        linear-gradient(
            180deg,
            rgba(0,5,11,.82) 0%,
            rgba(1,9,17,.76) 42%,
            rgba(0,5,11,.88) 100%
        ),
        var(--bg-photo);

    background-size:cover;
    background-position:left center;
    background-repeat:no-repeat;
    background-attachment:fixed;

    box-shadow:
        0 0 0 1px rgba(53,215,243,.06) inset,
        0 12px 34px rgba(0,0,0,.36),
        0 0 24px rgba(53,215,243,.08);

    backdrop-filter:blur(1.5px);
}
/* No independent sidebar scrollbar — the whole page scrolls together. */

.sidebar-brand{width:100%;display:flex;flex-direction:column;align-items:center;text-align:center;margin-top:8px;margin-bottom:22px;}
.sidebar-brand-icon{width:75px;height:75px;object-fit:contain;margin-bottom:10px;filter:drop-shadow(0 0 12px rgba(53,215,243,.45));}
.sidebar-bottom-art{
    width:calc(100% + 1.8rem);
    margin-left:-.9rem;
    margin-top:24px;
    position:relative;
    display:flex;
    justify-content:center;
    align-items:flex-end;
    overflow:hidden;

    /* IMPORTANT: no extra panel/background */
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
}

.sidebar-bottom-art::before,
.sidebar-bottom-art::after{
    content:none !important;
    display:none !important;
}

.sidebar-art-img{
    width:100%;
    max-width:none;
    height:auto;
    display:block;
    object-fit:contain;

    /* Black/dark pixels visually disappear into the sidebar background */
    mix-blend-mode:screen;
    opacity:.58;

    filter:
        saturate(.72)
        brightness(.72)
        contrast(1.02)
        drop-shadow(0 0 12px rgba(53,215,243,.12));

    background:transparent !important;
    box-shadow:none !important;
}
.sb-title{color:#f3fbff;font-size:.78rem;letter-spacing:2.5px;font-weight:800;line-height:1.6;margin:0;text-align:center;text-shadow:0 1px 10px rgba(0,0,0,.85),0 0 12px rgba(53,215,243,.20);}
.sb-section-label{color:#8fb4c7;font-size:.62rem;letter-spacing:2px;font-weight:700;margin:18px 0 8px 2px;text-transform:uppercase;}
.st-key-custom_sidebar hr{border-color:rgba(53,215,243,.20) !important;margin:15px 0;}
.threshold-label{color:#87a8b9;font-size:.68rem;letter-spacing:1.6px;margin-top:5px;background:transparent;}
.threshold-value{color:var(--accent);font-weight:800;}

.st-key-custom_sidebar .stButton > button{
    width:100%;min-height:58px;
    background:linear-gradient(180deg,rgba(4,16,28,.94),rgba(3,12,22,.96));
    border:1px solid rgba(113,174,198,.27);border-radius:9px;
    color:#bed3dd;font-family:'Courier New',monospace;font-size:.63rem;font-weight:700;
    letter-spacing:1.2px;padding:12px 6px;line-height:1.45;white-space:pre-line;
    box-shadow:0 4px 12px rgba(0,0,0,.25);transition:border-color .15s,background .15s,transform .15s;
}
.st-key-custom_sidebar .stButton > button:hover{
    background:linear-gradient(180deg,rgba(6,28,42,.98),rgba(3,18,30,.98));
    border-color:var(--accent);color:#ecfbff;transform:translateY(-1px);
}
.st-key-mode_active_img .stButton > button,
.st-key-mode_active_vid .stButton > button{
    background:linear-gradient(155deg,#0a3947,#062530) !important;
    border:1px solid var(--accent) !important;color:var(--accent-soft) !important;
    box-shadow:0 0 0 1px rgba(53,215,243,.12) inset,0 0 18px rgba(53,215,243,.16) !important;
}
.st-key-custom_sidebar label{color:#9ab6c4 !important;font-size:.76rem !important;}


.sidebar-status-card{
    margin-top:18px;
    padding-top:14px;
    border-top:1px solid rgba(53,215,243,.12);

    /* no separate background: let sidebar background show through */
    background:transparent !important;
    box-shadow:none !important;
}
.sidebar-status-title{
    color:var(--accent);
    font-size:.60rem;
    font-weight:800;
    letter-spacing:1.5px;
    margin-bottom:9px;
}
.sidebar-status-row{
    display:grid;
    grid-template-columns:18px 1fr auto;
    align-items:center;
    gap:7px;
    padding:5px 0;
    color:#c6d9e2;
    font-size:.64rem;
}
.sidebar-status-row b{color:var(--green);font-weight:800;}
.status-dot{width:7px;height:7px;border-radius:50%;display:inline-block;}
.status-dot.ok{
    background:var(--green);
    box-shadow:0 0 7px rgba(41,229,140,.60);
}


/* ═══ SVG ICON SYSTEM ═══════════════════════════════════════════════════ */
.ui-svg-icon{
    width:16px;
    height:16px;
    object-fit:contain;
    display:inline-block;
    vertical-align:-3px;
    margin-right:7px;
    flex:0 0 auto;
}
.ui-svg-icon.sm{width:13px;height:13px;margin-right:6px;vertical-align:-2px;}
.ui-svg-icon.md{width:18px;height:18px;margin-right:8px;vertical-align:-3px;}
.ui-svg-icon.lg{width:22px;height:22px;margin-right:8px;}
.det-row .ico .ui-svg-icon{width:17px;height:17px;margin:0;vertical-align:middle;}
.stat-row .ico .ui-svg-icon{width:15px;height:15px;margin:0;vertical-align:middle;}
.vessel-table .ui-svg-icon{width:14px;height:14px;margin-right:6px;vertical-align:-2px;}

/* Sidebar mode buttons use SVG files from assets/icon */
.st-key-mode_active_img .stButton > button,
.st-key-mode_idle_img .stButton > button,
.st-key-mode_active_vid .stButton > button,
.st-key-mode_idle_vid .stButton > button{
    padding-top:30px !important;
    position:relative;
}
.st-key-mode_active_img .stButton > button::before,
.st-key-mode_idle_img .stButton > button::before,
.st-key-mode_active_vid .stButton > button::before,
.st-key-mode_idle_vid .stButton > button::before{
    content:"";
    position:absolute;
    top:8px;
    left:50%;
    transform:translateX(-50%);
    width:17px;
    height:17px;
    background-position:center;
    background-repeat:no-repeat;
    background-size:contain;
}
.st-key-mode_active_img .stButton > button::before,
.st-key-mode_idle_img .stButton > button::before{
    background-image:var(--svg-image-detection);
}
.st-key-mode_active_vid .stButton > button::before,
.st-key-mode_idle_vid .stButton > button::before{
    background-image:var(--svg-video-detection);
}

/* Download buttons */
.st-key-download_image_log .stDownloadButton > button,
.st-key-download_video_log .stDownloadButton > button{
    position:relative;
    padding-left:34px !important;
}
.st-key-download_image_log .stDownloadButton > button::before,
.st-key-download_video_log .stDownloadButton > button::before{
    content:"";
    position:absolute;
    left:11px;
    top:50%;
    transform:translateY(-50%);
    width:15px;
    height:15px;
    background-image:var(--svg-download);
    background-position:center;
    background-repeat:no-repeat;
    background-size:contain;
}

/* ═══ SECTION HEADER ═══════════════════════════════════════════════════ */
.section-h{
    display:flex;align-items:center;gap:10px;
    width:max-content;max-width:100%;
    color:var(--accent);
    background:rgba(2,10,18,.72);border:1px solid rgba(53,215,243,.15);border-radius:8px;
    padding:7px 11px;margin:5px 0 10px 0;
    font-size:.92rem;font-weight:800;letter-spacing:2px;text-transform:uppercase;
    text-shadow:0 0 14px rgba(53,215,243,.20);box-shadow:0 6px 18px rgba(0,0,0,.18);
}
h2,h3{color:var(--accent) !important;letter-spacing:1.5px;text-transform:uppercase;}

/* ═══ FILE UPLOADER ═════════════════════════════════════════════════════ */
[data-testid="stFileUploaderDropzone"]{
    background:rgba(4,10,18,.95) !important;
    border:1px solid rgba(113,174,198,.24) !important;
    border-radius:10px !important;
    box-shadow:0 7px 18px rgba(0,0,0,.26) !important;
    min-height:54px !important;
    padding:.22rem .60rem !important;
}
[data-testid="stFileUploaderDropzone"] *{color:#cfdee5 !important;}
[data-testid="stFileUploader"]{margin-bottom:.35rem !important;}

/* ═══ DETECTION LAYOUT ══════════════════════════════════════════════════ */

.det-card-head{
    display:grid;
    grid-template-columns:auto 1fr auto;
    align-items:center;
    gap:12px;
    padding:8px 13px;
    margin:0 0 8px 0;
    background:linear-gradient(90deg,rgba(5,22,34,.96),rgba(3,14,24,.94));
    border:1px solid rgba(53,215,243,.20);
    border-radius:9px;
    font-size:.60rem;
    letter-spacing:1.8px;
    color:#91afbd;
    text-transform:uppercase;
    box-shadow:0 5px 14px rgba(0,0,0,.22);
}
.det-card-head .dot-live{
    width:7px;height:7px;border-radius:50%;
    background:var(--green);
    box-shadow:0 0 8px var(--green);
    display:inline-block;margin-right:6px;
    animation:blink 1.4s infinite;
}
.det-source{
    text-align:center;
    color:#627b88;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
}
.det-conf{color:#627b88;white-space:nowrap;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.35;}}

.st-key-detection_workspace{
    background:rgba(2,9,16,.90);
    border:1px solid rgba(53,215,243,.23);
    border-radius:12px;
    padding:12px !important;
    box-shadow:0 10px 26px rgba(0,0,0,.30);
    margin-bottom:10px;
}

.st-key-det_img_pane{
    min-height:455px;
    height:455px;
    background:rgba(1,7,13,.76);
    border:1px solid rgba(53,215,243,.12);
    border-radius:9px;
    padding:10px !important;
    overflow:hidden;
    display:flex;
    align-items:center;
    justify-content:center;
}
.st-key-det_img_pane [data-testid="stImage"]{
    width:100%;
    height:100%;
    display:flex;
    align-items:center;
    justify-content:center;
}
.st-key-det_img_pane [data-testid="stImage"] > div{
    width:100%;
    height:100%;
    display:flex;
    align-items:center;
    justify-content:center;
}
.st-key-det_img_pane [data-testid="stImage"] img{
    width:100% !important;
    height:100% !important;
    max-width:100% !important;
    max-height:100% !important;
    object-fit:contain !important;
    object-position:center center !important;
    border:none !important;
    border-radius:7px !important;
    box-shadow:none !important;
}

/* Plain-class equivalents of the .st-key-* workspace panes above, used for
   the LIVE video frame which is redrawn via plain HTML inside a placeholder
   (st.empty()) rather than st.container(key=...), since re-using an explicit
   container key many times within a single script run is not supported. */
.det-workspace{
    background:rgba(2,9,16,.90);
    border:1px solid rgba(53,215,243,.23);
    border-radius:12px;
    padding:12px;
    box-shadow:0 10px 26px rgba(0,0,0,.30);
    margin-bottom:10px;
}
.det-workspace-inner{
    display:grid;
    grid-template-columns:1.48fr 1fr;
    gap:16px;
    align-items:stretch;
}
.det-img-pane{
    min-height:455px;
    height:455px;
    background:rgba(1,7,13,.76);
    border:1px solid rgba(53,215,243,.12);
    border-radius:9px;
    padding:10px;
    overflow:hidden;
    display:flex;
    align-items:center;
    justify-content:center;
}
.det-img-pane img{
    width:100%;
    height:100%;
    max-width:100%;
    max-height:100%;
    object-fit:contain;
    object-position:center center;
    border:none;
    border-radius:7px;
    box-shadow:none;
}
.det-detail-pane{
    min-height:455px;
    height:100%;
    padding:12px;
    background:rgba(1,8,14,.96);
    border:1px solid rgba(53,215,243,.13);
    border-radius:9px;
    overflow:hidden;
}
@media(max-width:1200px){
    .det-workspace-inner{grid-template-columns:1fr;}
    .det-img-pane,.det-detail-pane{min-height:auto;height:auto;}
    .det-img-pane img{height:auto;max-height:420px;}
}

.tagrow{
    display:flex;
    flex-wrap:wrap;
    gap:6px;
    padding:8px 2px 0 2px;
}
.tagpill{
    padding:4px 9px;
    border-radius:5px;
    font-size:.59rem;
    font-weight:800;
    letter-spacing:.45px;
    color:#031018;
    box-shadow:0 2px 7px rgba(0,0,0,.28);
}

.st-key-det_detail_pane{
    min-height:455px;
    height:100%;
    padding:12px !important;
    background:rgba(1,8,14,.96);
    border:1px solid rgba(53,215,243,.13);
    border-radius:9px;
    overflow:hidden;
}
.det-detail-title{
    color:var(--accent);
    font-size:.70rem;
    font-weight:800;
    letter-spacing:1.5px;
    margin:0 0 10px 0;
    text-transform:uppercase;
}
.chip-row{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:8px;
    margin-bottom:10px;
}
.chip{
    padding:9px 6px;
    border-radius:7px;
    text-align:center;
    background:linear-gradient(180deg,rgba(5,16,28,.98),rgba(2,10,18,.98));
    border:1px solid rgba(53,215,243,.16);
}
.chip .cv{
    font-size:1.18rem;
    font-weight:800;
    color:var(--accent);
    line-height:1.05;
}
.chip .cl{
    font-size:.49rem;
    color:#7a9aaa;
    letter-spacing:1.05px;
    text-transform:uppercase;
    margin-top:3px;
}

.det-row{
    display:grid;
    grid-template-columns:22px minmax(0,1fr) auto auto;
    align-items:center;
    gap:8px;
    padding:9px 10px;
    border-radius:7px;
    background:rgba(4,14,24,.84);
    border:1px solid rgba(53,215,243,.08);
    margin-bottom:6px;
}
.det-row:hover{border-color:rgba(53,215,243,.26);}
.det-row .ico{font-size:.88rem;text-align:center;}
.det-row .name{
    min-width:0;
    font-size:.64rem;
    font-weight:700;
    letter-spacing:.45px;
    color:#d4eaf3;
    text-transform:uppercase;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
}
.det-row .conf{
    font-size:.64rem;
    font-weight:800;
    color:var(--accent-soft);
}
.badge{
    display:inline-block;
    padding:3px 7px;
    border-radius:4px;
    font-size:.50rem;
    font-weight:800;
    letter-spacing:.4px;
}
.det-threat-summary{
    margin-top:10px;
    padding:11px 10px;
    border-top:1px solid rgba(53,215,243,.10);
    background:rgba(3,12,20,.58);
    border-radius:7px;
}
.det-threat-title{
    color:#8fb4c7;
    font-size:.56rem;
    letter-spacing:1.35px;
    text-transform:uppercase;
    margin-bottom:7px;
}
.det-threat-clear{
    color:var(--green);
    font-size:.66rem;
    font-weight:800;
}
.det-threat-warn{
    color:var(--red);
    font-size:.66rem;
    font-weight:800;
}
.detail-download-wrap{
    margin-top:12px;
    padding-top:10px;
    border-top:1px solid rgba(53,215,243,.10);
}
.st-key-detail_download .stDownloadButton > button{
    width:100%;
    min-height:38px;
    background:linear-gradient(180deg,rgba(5,20,31,.98),rgba(2,12,20,.98)) !important;
    border:1px solid rgba(53,215,243,.30) !important;
    color:var(--accent-soft) !important;
    font-size:.60rem !important;
    font-weight:800 !important;
    letter-spacing:1px !important;
}
.st-key-detail_download .stDownloadButton > button:hover{
    border-color:var(--accent) !important;
    background:rgba(7,31,44,.98) !important;
}

.det-threat-sub{
    color:#7f9ead;
    font-size:.58rem;
    margin-top:4px;
    line-height:1.45;
}
.no-det{
    color:#819eac;
    font-size:.76rem;
    letter-spacing:.7px;
    padding:16px 10px;
    text-align:center;
    border:1px dashed rgba(53,215,243,.15);
    border-radius:8px;
    background:rgba(3,10,18,.60);
}

@media(max-width:1200px){
    .st-key-det_img_pane,.st-key-det_detail_pane{
        min-height:auto;
        height:auto;
    }
    .st-key-det_img_pane [data-testid="stImage"]{
        height:auto;
    }
    .st-key-det_img_pane [data-testid="stImage"] img{
        height:auto !important;
        max-height:420px !important;
    }
}


/* ═══ LARGER DETECTION DETAILS TYPOGRAPHY ═══════════════════════════════ */
.det-detail-title{
    font-size:.82rem !important;
    letter-spacing:1.65px !important;
    margin-bottom:12px !important;
}

.chip .cv{
    font-size:1.38rem !important;
}
.chip .cl{
    font-size:.57rem !important;
    letter-spacing:1.10px !important;
}

.det-row{
    min-height:46px;
    padding:11px 12px !important;
    gap:9px !important;
}
.det-row .name{
    font-size:.72rem !important;
    letter-spacing:.48px !important;
}
.det-row .conf{
    font-size:.72rem !important;
}
.det-row .badge,
.badge{
    font-size:.56rem !important;
    padding:4px 8px !important;
}

.det-row .ico .ui-svg-icon{
    width:19px !important;
    height:19px !important;
}

.det-threat-title{
    font-size:.63rem !important;
    letter-spacing:1.4px !important;
}
.det-threat-clear,
.det-threat-warn{
    font-size:.72rem !important;
}
.det-threat-sub{
    font-size:.64rem !important;
    line-height:1.55 !important;
}

.st-key-detail_download .stDownloadButton > button{
    font-size:.66rem !important;
}

/* ═══ ALERTS ════════════════════════════════════════════════════════════ */
.military-alert{
    background:linear-gradient(135deg,rgba(38,0,3,.97),rgba(63,5,8,.97));
    border:1px solid rgba(255,85,93,.88);border-radius:9px;padding:16px 20px;margin:12px 0;
    animation:pulse-border 1.5s infinite;text-align:center;box-shadow:0 8px 24px rgba(0,0,0,.32);
}
.military-alert .alert-title{color:#ff747b;font-size:1.05rem;font-weight:800;letter-spacing:2.5px;}
.military-alert .alert-body{color:#ffc1c5;font-size:.82rem;margin-top:6px;letter-spacing:.8px;}
@keyframes pulse-border{
    0%,100%{box-shadow:0 0 0 rgba(255,85,93,0),0 8px 24px rgba(0,0,0,.32);}
    50%{box-shadow:0 0 16px rgba(255,85,93,.24),0 8px 24px rgba(0,0,0,.32);}
}
.local-alert{
    background:linear-gradient(135deg,rgba(36,18,0,.97),rgba(57,30,2,.97));
    border:1px solid rgba(255,174,74,.82);border-radius:9px;padding:16px 20px;margin:12px 0;
    text-align:center;box-shadow:0 8px 24px rgba(0,0,0,.30);
}
.local-alert .alert-title{color:var(--orange);font-size:1.05rem;font-weight:800;letter-spacing:2px;}
.local-alert .alert-body{color:#ffe0b9;font-size:.82rem;margin-top:6px;letter-spacing:.8px;}

/* ═══ STAT CARDS ════════════════════════════════════════════════════════ */
.stat-card{
    background:linear-gradient(180deg,rgba(4,13,23,.96),rgba(2,9,17,.98));
    border:1px solid rgba(53,215,243,.17);
    box-shadow:0 7px 18px rgba(0,0,0,.24);
    border-radius:10px;
    padding:13px 15px;
    min-height:138px;
    height:100%;
}
.stat-card h4{
    margin:0 0 12px 0;font-size:.72rem;letter-spacing:1.4px;color:#e2f0f5;text-transform:uppercase;
    display:flex;align-items:center;gap:8px;
}
.stat-card h4 .dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);}
.stat-row{
    display:flex;align-items:center;gap:10px;font-size:.78rem;color:#d7e5eb;
    padding:7px 0;border-bottom:1px solid rgba(96,176,205,.08);
}
.stat-row:last-child{border-bottom:none;}
.stat-row .ico{color:var(--accent);width:18px;text-align:center;}
.stat-row b{color:var(--accent-soft);}

/* ═══ VESSEL TABLE ══════════════════════════════════════════════════════ */
.vessel-table{width:100%;border-collapse:collapse;font-size:.76rem;}
.vessel-table th{
    text-align:left;color:#7f9ead;letter-spacing:1px;text-transform:uppercase;font-size:.62rem;
    padding:0 0 9px 0;border-bottom:1px solid rgba(53,215,243,.18);
}
.vessel-table td{padding:8px 0;border-bottom:1px solid rgba(96,176,205,.07);color:#dbe8ee;}
.vessel-table .risk{color:var(--orange);font-weight:800;}
.vessel-table .risk.high{color:var(--red);}
.vessel-table .risk.low{color:var(--green);}

/* ═══ VIDEO PLAYER CARD ═════════════════════════════════════════════════ */
.player-card{
    background:rgba(3,10,18,.95);border:1px solid rgba(53,215,243,.20);
    border-radius:12px;padding:0 0 14px 0;overflow:hidden;margin-bottom:20px;box-shadow:var(--shadow);
}
.player-card-head{
    display:flex;align-items:center;justify-content:space-between;
    padding:10px 16px;
    background:linear-gradient(90deg,rgba(7,26,39,.96),rgba(4,14,24,.96));
    border-bottom:1px solid rgba(53,215,243,.18);
    font-size:.68rem;letter-spacing:2px;color:#91afbd;
}
.thumbstrip [data-testid="stImage"] img{border:1px solid rgba(53,215,243,.20) !important;border-radius:7px !important;opacity:.88;}
.scrub-times{display:flex;justify-content:space-between;color:#7fa0b0;font-size:.68rem;letter-spacing:1px;padding:3px 16px 0 16px;}
.stProgress > div > div{background:linear-gradient(90deg,#0c5367,var(--accent)) !important;}

/* ═══ MISC ══════════════════════════════════════════════════════════════ */
[data-testid="stImage"] img{border:1px solid rgba(53,215,243,.18);border-radius:8px;box-shadow:0 8px 26px rgba(0,0,0,.30);}
.stButton > button{background:#071c29;color:var(--accent-soft);border:1px solid rgba(53,215,243,.34);border-radius:7px;letter-spacing:1.2px;font-family:'Courier New',monospace;}
.stButton > button:hover{background:#0a3143;border-color:var(--accent);color:#fff;}
hr{border-color:rgba(53,215,243,.15) !important;}

@media(max-width:980px){
    div[data-testid="stColumn"]:has(.st-key-custom_sidebar)
    > div[data-testid="stVerticalBlock"]{height:auto !important;}
    .st-key-custom_sidebar{height:auto;min-height:auto;position:relative;top:0;}
    .section-h{font-size:.95rem;}
}
</style>
""", unsafe_allow_html=True)


# ── Layout columns ────────────────────────────────────────────────────────
sidebar_col, content_col = st.columns([0.205, 0.795], gap="small")

# ── Sidebar ───────────────────────────────────────────────────────────────
with sidebar_col:
    with st.container(key="custom_sidebar"):
        st.markdown(f"""
<div class="sidebar-brand">
    <img src="data:image/svg+xml;base64,{ICON_B64}" class="sidebar-brand-icon">
    <div class="sb-title">ADVANCE MARITIME<br>DOMAIN AWARENESS</div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="sb-section-label">Detection Mode</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            with st.container(key="mode_active_img" if st.session_state.detect_mode == "Image" else "mode_idle_img"):
                if st.button("IMAGE\nDETECTION", key="btn_img_mode", use_container_width=True):
                    st.session_state.detect_mode = "Image"; st.rerun()
        with c2:
            with st.container(key="mode_active_vid" if st.session_state.detect_mode == "Video" else "mode_idle_vid"):
                if st.button("VIDEO STREAM\nDETECTION", key="btn_vid_mode", use_container_width=True):
                    st.session_state.detect_mode = "Video"; st.rerun()

        mode = st.session_state.detect_mode
        st.markdown("---")
        st.markdown('<div class="sb-section-label">Model Confidence Threshold</div>', unsafe_allow_html=True)
        conf_thresh = st.slider("Confidence threshold", 0.01, 0.95, 0.25, 0.01, label_visibility="collapsed")
        st.markdown(f'<div class="threshold-label">Threshold: <span class="threshold-value">{conf_thresh:.0%}</span></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-bottom-art">
            <img src="data:{SHIP_MIME};base64,{SHIP_B64}" class="sidebar-art-img">
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-status-card">
            <div class="sidebar-status-title">SYSTEM STATUS</div>
            <div class="sidebar-status-row">{svg_icon("model.svg","ui-svg-icon sm")}<span>Model</span><b>Loaded</b></div>
            <div class="sidebar-status-row">{svg_icon("detection_ready.svg","ui-svg-icon sm")}<span>Detection</span><b>Ready</b></div>
            <div class="sidebar-status-row">{svg_icon("logging.svg","ui-svg-icon sm")}<span>Logging</span><b>Ready</b></div>
        </div>
        """, unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path, conf):
    return GuardianDetector(path, conf)

try:
    detector = load_model(MODEL_PATH, conf_thresh)
except Exception as e:
    st.error(f"⚠ Model failed to load: {e}"); st.stop()


# ── Helper functions ──────────────────────────────────────────────────────
BADGE_STYLES = {
    "HIGH PRIORITY": "background:#ff2222;color:#fff",
    "PRIORITY":      "background:#ff8800;color:#000",
    "MONITOR":       "background:#cc8800;color:#000",
    "SMALL CRAFT":   "background:#ccaa00;color:#000",
    "CIVILIAN":      "background:#00aa55;color:#fff",
}
VESSEL_ICONS = {
    "foreign_military_ship": "foreign_military_ship.svg",
    "local_military_ship":   "local_military_ship.svg",
    "cargo_ship":            "cargo_ship.svg",
    "oil_tanker":            "oil_tanker.svg",
    "tug_boat":              "tug_boat.svg",
    "small_craft":           "small_craft.svg",
    "passenger_ferry":       "passenger_ferry.svg",
}

def threat_badge(level):
    style = BADGE_STYLES.get(level, "background:#333;color:#aaa")
    return f'<span class="badge" style="{style}">{level}</span>'

def bgr_to_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"

def risk_class(level):
    return "high" if level == "HIGH PRIORITY" else ("low" if level == "CIVILIAN" else "")

def military_alert(detections):
    foreign = [d for d in detections if d["class_name"] == "foreign_military_ship"]
    local   = [d for d in detections if d["class_name"] == "local_military_ship"]

    if foreign:
        st.markdown(
            f'<div class="military-alert">'
            f'<div class="alert-title">'
            f'{svg_icon("foreign_military_ship.svg","ui-svg-icon md")}HIGH PRIORITY ALERT'
            f'</div>'
            f'<div class="alert-body">'
            f'FOREIGN MILITARY VESSEL DETECTED &nbsp;|&nbsp; '
            f'Confidence: {foreign[0]["confidence"]:.0%}<br>'
            f'Threat Level: HIGH PRIORITY &nbsp;|&nbsp; '
            f'Immediate action — notify duty officer'
            f'</div></div>',
            unsafe_allow_html=True
        )

    if local:
        st.markdown(
            f'<div class="local-alert">'
            f'<div class="alert-title">'
            f'{svg_icon("local_military_ship.svg","ui-svg-icon md")}PRIORITY ALERT'
            f'</div>'
            f'<div class="alert-body">'
            f'LOCAL MILITARY VESSEL DETECTED &nbsp;|&nbsp; '
            f'Confidence: {local[0]["confidence"]:.0%}<br>'
            f'Threat Level: PRIORITY &nbsp;|&nbsp; '
            f'Log and monitor — report to command'
            f'</div></div>',
            unsafe_allow_html=True
        )

def render_tag_pills(detections, container=st):
    if not detections:
        container.markdown('<div class="tagrow"><span style="color:#819eac;font-size:.8rem;">— no vessels detected —</span></div>', unsafe_allow_html=True)
        return
    pills = "".join(
        f'<span class="tagpill" style="background:{bgr_to_hex(d["colour"])}">'
        f'{d["class_name"].replace("_"," ").upper()} {d["confidence"]:.0%}</span>'
        for d in detections
    )
    container.markdown(f'<div class="tagrow">{pills}</div>', unsafe_allow_html=True)

def _build_detail_panel_html(detections):
    """Builds the chip-row / detection-row / threat-summary HTML shared by
    both the keyed (image mode / final video summary) and plain-HTML
    (live video frame) detection cards."""
    total   = len(detections)
    classes = len({d["class_name"] for d in detections})
    threats = sum(
        1 for d in detections
        if d["threat_level"] in ("HIGH PRIORITY", "PRIORITY")
    )

    if detections:
        rows_html = ""
        for d in detections:
            name = d["class_name"].replace("_", " ").upper()
            icon_file = VESSEL_ICONS.get(d["class_name"], "vessels.svg")
            icon_html = svg_icon(icon_file, "ui-svg-icon")
            badge = threat_badge(d["threat_level"])
            conf = f"{d['confidence']:.0%}"
            hexc = bgr_to_hex(d["colour"])

            rows_html += (
                f'<div class="det-row">'
                f'<span class="ico">{icon_html}</span>'
                f'<span class="name" '
                f'style="border-left:3px solid {hexc};padding-left:7px">'
                f'{name}</span>'
                f'{badge}'
                f'<span class="conf">{conf}</span>'
                f'</div>'
            )
    else:
        rows_html = '<div class="no-det">— NO VESSELS DETECTED —</div>'

    if threats:
        threat_html = (
            '<div class="det-threat-summary">'
            f'<div class="det-threat-title">'
            f'{svg_icon("threat_summary.svg","ui-svg-icon sm")}Threat Summary'
            f'</div>'
            f'<div class="det-threat-warn">'
            f'{threats} priority threat event(s) detected'
            f'</div>'
            '<div class="det-threat-sub">'
            'Review highlighted detections and follow operational procedure.'
            '</div>'
            '</div>'
        )
    else:
        threat_html = (
            '<div class="det-threat-summary">'
            f'<div class="det-threat-title">'
            f'{svg_icon("threat_summary.svg","ui-svg-icon sm")}Threat Summary'
            f'</div>'
            '<div class="det-threat-clear">No threats detected</div>'
            '<div class="det-threat-sub">'
            'All current detections are below priority threat level.'
            '</div>'
            '</div>'
        )

    detail_icon = svg_icon("detection_details.svg", "ui-svg-icon sm")
    threat_color = "var(--red)" if threats else "var(--green)"

    detail_html = (
        f'<div class="det-detail-title">{detail_icon}Detection Details</div>'
        '<div class="chip-row">'
        '<div class="chip">'
        f'<div class="cv">{total}</div>'
        f'<div class="cl">{svg_icon("vessels.svg","ui-svg-icon sm")}Vessels</div>'
        '</div>'
        '<div class="chip">'
        f'<div class="cv">{classes}</div>'
        f'<div class="cl">{svg_icon("classes.svg","ui-svg-icon sm")}Classes</div>'
        '</div>'
        '<div class="chip">'
        f'<div class="cv" style="color:{threat_color}">{threats}</div>'
        f'<div class="cl">{svg_icon("threats.svg","ui-svg-icon sm")}Threats</div>'
        '</div>'
        '</div>'
        f'{rows_html}'
        f'{threat_html}'
    )
    return detail_html

def _card_head_html(source_name, conf_thresh, header_label="DETECTION VIEW"):
    display_name = (
        source_name
        if len(source_name) <= 44
        else source_name[:21] + "…" + source_name[-18:]
    )
    return (
        f'<div class="det-card-head">'
        f'<span><span class="dot-live"></span>{header_label}</span>'
        f'<span class="det-source">{display_name}</span>'
        f'<span class="det-conf">CONF ≥ {conf_thresh:.0%}</span>'
        f'</div>'
    )

def render_detection_card(annotated_rgb, detections, source_name, conf_thresh, log_data=None,
                           img_pane_key="det_img_pane", detail_pane_key="det_detail_pane",
                           workspace_key="detection_workspace", download_key="detail_download",
                           header_label="DETECTION VIEW"):
    """Reference-style detection workspace: image left, all details right.
    Uses keyed st.container()s for styling hooks — safe to call once per
    script run (e.g. image mode, or the final video summary), but must NOT
    be called repeatedly with the same keys inside a loop in a single run."""
    st.markdown(_card_head_html(source_name, conf_thresh, header_label), unsafe_allow_html=True)

    with st.container(key=workspace_key):
        img_col, det_col = st.columns([1.48, 1], gap="medium")

        with img_col:
            with st.container(key=img_pane_key):
                if annotated_rgb is not None:
                    st.image(annotated_rgb, use_container_width=True)
                else:
                    st.markdown('<div class="no-det">— NO FRAME —</div>', unsafe_allow_html=True)

        with det_col:
            with st.container(key=detail_pane_key):
                st.markdown(_build_detail_panel_html(detections), unsafe_allow_html=True)

                if log_data is not None:
                    st.markdown('<div class="detail-download-wrap"></div>', unsafe_allow_html=True)
                    with st.container(key=download_key):
                        st.download_button(
                            "DOWNLOAD DETECTION LOG",
                            log_data,
                            "detection_log.csv",
                            "text/csv",
                            use_container_width=True,
                            key=f"{download_key}_btn"
                        )

def render_aggregate_section(class_counts, class_levels, live_info, tracking_info,
                               title="STREAM STATUS & AGGREGATE STATS"):
    st.markdown(
        f'<div class="section-h">'
        f'{svg_icon("vessel_breakdown.svg","ui-svg-icon md")}{title}'
        f'</div>',
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns([1, 1, 1.20], gap="medium")

    with c1:
        rows = "".join(
            f'<div class="stat-row">'
            f'<span class="ico">{svg_icon(icon_file,"ui-svg-icon sm")}</span>'
            f'{lab}<span style="margin-left:auto"><b>{val}</b></span>'
            f'</div>'
            for icon_file, lab, val in live_info["rows"]
        )
        st.markdown(
            f'<div class="stat-card">'
            f'<h4>{svg_icon("image_status.svg","ui-svg-icon sm")}{live_info["title"]}</h4>'
            f'{rows}</div>',
            unsafe_allow_html=True
        )

    with c2:
        rows = "".join(
            f'<div class="stat-row">'
            f'<span class="ico">{svg_icon(icon_file,"ui-svg-icon sm")}</span>'
            f'{lab}<span style="margin-left:auto"><b>{val}</b></span>'
            f'</div>'
            for icon_file, lab, val in tracking_info["rows"]
        )
        st.markdown(
            f'<div class="stat-card">'
            f'<h4>{svg_icon("object_tracking.svg","ui-svg-icon sm")}{tracking_info["title"]}</h4>'
            f'{rows}</div>',
            unsafe_allow_html=True
        )

    with c3:
        if class_counts:
            body = "".join(
                f'<tr><td>'
                f'{svg_icon(VESSEL_ICONS.get(cls,"vessels.svg"),"ui-svg-icon sm")}'
                f'{cls.replace("_"," ").title()}</td>'
                f'<td>{cnt}</td>'
                f'<td class="risk {risk_class(class_levels.get(cls,""))}">'
                f'{class_levels.get(cls,"—").title()}</td></tr>'
                for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])
            )
            table = (
                '<table class="vessel-table">'
                '<tr><th>Vessel Type</th><th>Count</th><th>Threat</th></tr>'
                f'{body}</table>'
            )
        else:
            table = '<div class="no-det">— no aggregate data —</div>'

        st.markdown(
            f'<div class="stat-card">'
            f'<h4>{svg_icon("vessel_breakdown.svg","ui-svg-icon sm")}VESSEL TYPE BREAKDOWN</h4>'
            f'{table}</div>',
            unsafe_allow_html=True
        )

def fmt_time(s):
    s = max(0, int(s)); m, s = divmod(s, 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def render_video_frame_html(annotated_rgb, detections, source_name, conf_thresh, header_label="DETECTION VIEW — LIVE"):
    """Plain-HTML equivalent of render_detection_card for the LIVE video loop.
    Returns one HTML string (no st.container keys, no widgets) so it can be
    written into an st.empty() placeholder every frame within a single
    script run without triggering duplicate-element-key errors, and without
    the cost of a full Streamlit script rerun per frame."""
    if annotated_rgb is not None:
        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        img_html = f'<img src="data:image/jpeg;base64,{img_b64}" alt="live frame">'
    else:
        img_html = '<div class="no-det">— NO FRAME —</div>'

    return (
        _card_head_html(source_name, conf_thresh, header_label)
        + '<div class="det-workspace"><div class="det-workspace-inner">'
        + f'<div class="det-img-pane">{img_html}</div>'
        + f'<div class="det-detail-pane">{_build_detail_panel_html(detections)}</div>'
        + '</div></div>'
    )


# ══════════════════════════════════════════════════════════════════════════
# CONTENT AREA
# ══════════════════════════════════════════════════════════════════════════
with content_col:

    # ─── IMAGE MODE ───────────────────────────────────────────────────────
    if mode == "Image":
        st.markdown(f'<div class="section-h">{svg_icon("image_detection.svg","ui-svg-icon md")}IMAGE DETECTION</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp"],
                                    label_visibility="collapsed")

        if uploaded:
            img        = np.array(Image.open(uploaded).convert("RGB"))
            frame_bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            detections = detector.predict(frame_bgr)
            annotated  = detector.annotate(frame_bgr, detections)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            logger = DetectionLogger("outputs/single_image_log.csv")
            logger.log(frame_id=0, detections=detections)
            logger.close()

            with open("outputs/single_image_log.csv", "rb") as f:
                log_data = f.read()

            military_alert(detections)
            render_detection_card(
                annotated_rgb,
                detections,
                uploaded.name,
                conf_thresh,
                log_data=log_data
            )

    # ─── VIDEO MODE ───────────────────────────────────────────────────────
    elif mode == "Video":
        st.markdown(f'<div class="section-h">{svg_icon("video_detection.svg","ui-svg-icon md")}VIDEO STREAM DETECTION</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a video", type=["mp4","avi","mov"],
                                    label_visibility="collapsed")

        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.close()
            video_path   = tfile.name
            source_label = uploaded.name

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"⚠ Cannot open video: {video_path}"); st.stop()

            total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps      = cap.get(cv2.CAP_PROP_FPS) or 25
            width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            skip     = max(1, int(fps // 5))
            duration = total / fps if fps else 0

            # Placeholders updated IN PLACE for every processed frame — this
            # keeps the whole video loop inside a single Streamlit script run
            # (no st.rerun() per frame), which is what actually eliminates
            # the lag: a rerun re-executes the entire page (CSS, sidebar,
            # asset encoding) on top of the detection work every frame.
            alert_ph    = st.empty()
            card_ph     = st.empty()
            progress_ph = st.empty()
            time_ph     = st.empty()
            stats_ph    = st.empty()

            frame_id = 0
            class_counts, class_levels = {}, {}
            threat_events   = 0
            log_rows        = []
            last_detections = []
            last_annotated_rgb = None

            while cap.isOpened():
                if frame_id % skip == 0:
                    ret, frame = cap.read()          # decode: we're using this one
                else:
                    ret = cap.grab()                 # cheap skip: no decode
                    frame = None
                if not ret:
                    break

                if frame_id % skip == 0:
                    detections    = detector.predict(frame)
                    annotated     = detector.annotate(frame, detections)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    last_detections     = detections
                    last_annotated_rgb  = annotated_rgb
                    log_rows.append((frame_id, detections))

                    for d in detections:
                        class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
                        class_levels[d["class_name"]] = d["threat_level"]
                        if d["threat_level"] in ("HIGH PRIORITY", "PRIORITY"):
                            threat_events += 1

                    with alert_ph.container():
                        military_alert(detections)

                    card_ph.markdown(
                        render_video_frame_html(annotated_rgb, detections, source_label, conf_thresh),
                        unsafe_allow_html=True
                    )

                    elapsed = frame_id / fps if fps else 0
                    progress_ph.progress(min(frame_id / max(total, 1), 1.0))
                    time_ph.markdown(
                        f'<div class="scrub-times"><span>{fmt_time(elapsed)}</span><span>{fmt_time(duration)}</span></div>',
                        unsafe_allow_html=True
                    )

                    live_info = {"title": "LIVE STREAM STATUS", "rows": [
                        ("video_detection.svg", "Resolution", f"{width}×{height}@{fps:.0f}fps"),
                        ("classes.svg", "Frames Processed", f"{frame_id}/{total}"),
                        ("logging.svg", "Source", source_label),
                    ]}
                    tracking_info = {"title": "OBJECT TRACKING", "rows": [
                        ("object_tracking.svg", "Tracking", len(detections)),
                        ("classes.svg", "Classes", len(class_counts)),
                        ("threats.svg", "Threat Events", threat_events),
                    ]}
                    with stats_ph.container():
                        render_aggregate_section(class_counts, class_levels, live_info, tracking_info)

                frame_id += 1

            cap.release()

            # ── Final downloadable log covering every processed frame ───────
            log_data = None
            if log_rows:
                logger = DetectionLogger("outputs/video_log.csv")
                for fid, dets in log_rows:
                    logger.log(fid, dets)
                logger.close()
                with open("outputs/video_log.csv", "rb") as f:
                    log_data = f.read()

            # ── Replace the live card with the final (keyed) summary card ───
            alert_ph.empty()
            with alert_ph.container():
                military_alert(last_detections)

            card_ph.empty()
            with card_ph.container():
                render_detection_card(
                    last_annotated_rgb,
                    last_detections,
                    source_label,
                    conf_thresh,
                    log_data=log_data,
                    header_label="DETECTION VIEW"
                )

            st.success(f"✅ Processed {frame_id} frames — session complete.")