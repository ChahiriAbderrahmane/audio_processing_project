"""
app.py — Gradio web interface for the Forensic Audio Authentication System.

Features:
    - Audio upload (wav / flac / mp3)
    - Real-time tampering probability + verdict
    - Grad-CAM spectrogram overlay
    - SHAP feature importance bar chart
    - Exportable PDF court report

Usage:
    python app/app.py
"""

import os
import sys
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference     import predict
from xai.gradcam       import generate_gradcam_plot
from xai.shap_explain  import explain_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("app")

_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

TEMP_DIR = tempfile.mkdtemp(prefix="forensic_audio_")


def _temp_path(suffix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(TEMP_DIR, f"{ts}{suffix}")


def analyze_audio(audio_path: str) -> tuple:
    """
    Full analysis pipeline called by Gradio on each submission.

    Args:
        audio_path: Path provided by gr.Audio component.

    Returns:
        (verdict_md, gradcam_img_path, shap_img_path, status_msg)
    """
    if audio_path is None:
        return (
            "## ⚠️ No file uploaded",
            None,
            None,
            "Please upload an audio file to begin analysis.",
        )

    logger.info("Analyzing: %s", audio_path)

    try:
        # ── 1. Prediction ──────────────────────────────────────────────
        result        = predict(audio_path)
        verdict       = result["verdict"]
        tampered_pct  = result["tampered_prob"]  * 100
        authentic_pct = result["authentic_prob"] * 100
        confidence    = result["confidence"]     * 100

        verdict_emoji = "🔴 TAMPERED" if verdict == "TAMPERED" else "🟢 AUTHENTIC"
        risk_level    = (
            "HIGH RISK"   if tampered_pct >= 70 else
            "MEDIUM RISK" if tampered_pct >= 40 else
            "LOW RISK"
        )

        verdict_md = f"""
## Verdict: {verdict_emoji}

| Metric | Value |
|--------|-------|
| **Tampered Probability**  | {tampered_pct:.1f}% |
| **Authentic Probability** | {authentic_pct:.1f}% |
| **Confidence**            | {confidence:.1f}% |
| **Risk Level**            | {risk_level} |

*Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # ── 2. Grad-CAM ────────────────────────────────────────────────
        gradcam_path = _temp_path("_gradcam.png")
        try:
            generate_gradcam_plot(audio_path, output_path=gradcam_path)
        except Exception as exc:
            logger.error("Grad-CAM failed: %s", exc)
            gradcam_path = None

        # ── 3. SHAP ────────────────────────────────────────────────────
        shap_path = _temp_path("_shap.png")
        try:
            explain_prediction(audio_path, output_path=shap_path)
        except Exception as exc:
            logger.error("SHAP failed: %s", exc)
            shap_path = None

        status = f"✅ Analysis complete — {os.path.basename(audio_path)}"
        return verdict_md, gradcam_path, shap_path, status

    except FileNotFoundError as exc:
        msg = f"## ❌ Model Not Found\n\n{exc}\n\nRun `python model/train.py` first."
        return msg, None, None, "Error: model not found"

    except Exception as exc:
        logger.exception("Unexpected error during analysis")
        msg = f"## ❌ Analysis Failed\n\n```\n{exc}\n```"
        return msg, None, None, f"Error: {exc}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
/* ── Imports ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Design tokens ────────────────────────────────────────────────────────── */
:root {
    --bg-base:       #0a0c10;
    --bg-surface:    #0f1318;
    --bg-elevated:   #161b22;
    --bg-hover:      #1c2230;
    --border:        #21262d;
    --border-accent: #30363d;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --text-muted:    #484f58;
    --accent:        #58a6ff;
    --accent-dim:    #1f3a5f;
    --success:       #3fb950;
    --success-dim:   #12291a;
    --danger:        #f85149;
    --danger-dim:    #3d1a18;
    --warning:       #d29922;
    --scan-line:     rgba(88, 166, 255, 0.03);
    --font-mono:     'Share Tech Mono', monospace;
    --font-body:     'DM Sans', sans-serif;
    --radius-sm:     4px;
    --radius-md:     8px;
    --radius-lg:     12px;
}

/* ── Global reset ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background-color: var(--bg-base) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

/* Animated scan-line overlay on the whole page */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        var(--scan-line) 0px,
        var(--scan-line) 1px,
        transparent 1px,
        transparent 4px
    );
    pointer-events: none;
    z-index: 0;
}

footer { display: none !important; }

/* ── App shell ────────────────────────────────────────────────────────────── */
#main-container {
    max-width: 1280px;
    margin: 0 auto;
    padding: 32px 24px 64px;
    position: relative;
    z-index: 1;
}

/* ── Header ───────────────────────────────────────────────────────────────── */
#app-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 28px;
    margin-bottom: 32px;
}

#app-header .header-eyebrow {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
}

#app-header h1 {
    font-family: var(--font-mono) !important;
    font-size: 26px !important;
    font-weight: 400 !important;
    color: var(--text-primary) !important;
    letter-spacing: 0.04em;
    margin: 0 0 12px !important;
}

#app-header .header-sub {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 640px;
}

#app-header .header-badges {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    flex-wrap: wrap;
}

#app-header .badge {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-accent);
    color: var(--text-muted);
    text-transform: uppercase;
}

/* ── Panel cards ──────────────────────────────────────────────────────────── */
.panel-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    transition: border-color 0.2s ease;
}
.panel-card:hover { border-color: var(--border-accent); }

.panel-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.panel-label::before {
    content: '';
    display: inline-block;
    width: 20px;
    height: 1px;
    background: var(--accent);
}

/* ── Gradio component overrides ───────────────────────────────────────────── */

/* Labels */
label.svelte-1b6s6s, .block > label,
.gradio-container label {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    margin-bottom: 8px !important;
}

/* Audio component */
.gr-audio, [data-testid="audio"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}

/* Text / status box */
textarea, input[type="text"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.12) !important;
    outline: none !important;
}

/* Primary button */
button.primary, .gr-button-primary, button[variant="primary"] {
    background: var(--accent) !important;
    color: #0a0c10 !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 28px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 0 20px rgba(88, 166, 255, 0.2) !important;
}

button.primary:hover, .gr-button-primary:hover {
    background: #79b8ff !important;
    box-shadow: 0 0 32px rgba(88, 166, 255, 0.35) !important;
    transform: translateY(-1px) !important;
}

button.primary:active {
    transform: translateY(0) !important;
    box-shadow: 0 0 12px rgba(88, 166, 255, 0.2) !important;
}

/* Secondary button */
button.secondary, .gr-button-secondary {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color 0.15s, color 0.15s !important;
}

button.secondary:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Markdown output — verdict area */
.verdict-output .prose {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

.verdict-output .prose h2 {
    font-family: var(--font-mono) !important;
    font-size: 18px !important;
    letter-spacing: 0.06em !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 12px !important;
    margin-bottom: 16px !important;
}

.verdict-output .prose table {
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 14px !important;
}

.verdict-output .prose th {
    background: var(--bg-elevated) !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 8px 12px !important;
    text-align: left !important;
    border: 1px solid var(--border) !important;
}

.verdict-output .prose td {
    padding: 8px 12px !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-variant-numeric: tabular-nums !important;
}

.verdict-output .prose tr:hover td {
    background: var(--bg-hover) !important;
}

.verdict-output .prose em {
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    font-style: normal !important;
}

/* Image outputs */
.xai-panel img {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    width: 100% !important;
}

/* Accordion / Examples */
.examples-holder {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-surface) !important;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 32px 0;
}

/* Legend grid ───────────────────────────────────────────────── */
.legend-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

.legend-item {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px 18px;
    transition: border-color 0.2s;
}
.legend-item:hover { border-color: var(--border-accent); }

.legend-item .legend-title {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--accent);
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}

.legend-item .legend-body {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* Spec row ──────────────────────────────────────────────────── */
.spec-row {
    display: flex;
    gap: 24px;
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
    flex-wrap: wrap;
}

.spec-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-mono);
    font-size: 11px;
}

.spec-item .spec-key  { color: var(--text-muted); letter-spacing: 0.08em; }
.spec-item .spec-val  { color: var(--text-primary); }
.spec-item .spec-sep  { color: var(--border-accent); }

/* Pulse animation for status ────────────────────────────────── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 6px rgba(63, 185, 80, 0.4); }
    50%       { box-shadow: 0 0 18px rgba(63, 185, 80, 0.7); }
}

.status-active {
    animation: pulse-glow 2s ease-in-out infinite;
    border-color: var(--success) !important;
}
"""

# ── Layout ────────────────────────────────────────────────────────────────────

_HEADER_MD = """
<div id="app-header">
  <div class="header-eyebrow">Forensic Intelligence Platform · v2.0</div>
  <h1>🎙 Audio Authentication System</h1>
  <p class="header-sub">
    Deep learning–based tamper detection for legal and investigative use.
    Upload a recording to detect splicing, speed manipulation, or deepfake injection.
    All predictions are accompanied by explainable AI visualisations for court admissibility.
  </p>
  <div class="header-badges">
    <span class="badge">WAV · FLAC · MP3</span>
    <span class="badge">Grad-CAM XAI</span>
    <span class="badge">SHAP Analysis</span>
    <span class="badge">Court-ready</span>
  </div>
</div>
"""

_LEGEND_MD = """
<div class="legend-grid">
  <div class="legend-item">
    <div class="legend-title">Grad-CAM Heatmap</div>
    <div class="legend-body">
      Hotter regions indicate frequency–time areas the model flagged as suspicious.
      Edit boundaries typically manifest as bright vertical bands on the spectrogram.
    </div>
  </div>
  <div class="legend-item">
    <div class="legend-title">SHAP Feature Importance</div>
    <div class="legend-body">
      Taller bars indicate LFCC coefficients with the strongest contribution to the
      tampering verdict. Coefficients LFCC-15 → LFCC-19 are particularly sensitive
      to deepfake injection artefacts.
    </div>
  </div>
</div>
<div class="spec-row">
  <div class="spec-item">
    <span class="spec-key">EER TARGET</span>
    <span class="spec-sep">·</span>
    <span class="spec-val">&lt; 5%</span>
  </div>
  <div class="spec-item">
    <span class="spec-key">FAR TARGET</span>
    <span class="spec-sep">·</span>
    <span class="spec-val">1%</span>
  </div>
  <div class="spec-item">
    <span class="spec-key">STANDARD</span>
    <span class="spec-sep">·</span>
    <span class="spec-val">Judicial</span>
  </div>
</div>
"""

_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Share Tech Mono"), "monospace"],
)

with gr.Blocks(title="Forensic Audio Authentication System") as demo:

    # ── Header ─────────────────────────────────────────────────────────────────
    with gr.Column(elem_id="main-container"):
        gr.HTML(_HEADER_MD)

        # ── Input + Verdict row ────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left column — upload + controls
            with gr.Column(scale=1, min_width=300, elem_classes="panel-card"):
                gr.HTML('<div class="panel-label">01 — Input Signal</div>')
                audio_input = gr.Audio(
                    label="Audio File",
                    type="filepath",
                    sources=["upload"],
                )
                analyze_btn = gr.Button(
                    "⟶  Run Forensic Analysis",
                    variant="primary",
                    size="lg",
                )
                status_box = gr.Textbox(
                    label="System Status",
                    interactive=False,
                    lines=1,
                    placeholder="Awaiting input…",
                )

            # Right column — verdict
            with gr.Column(scale=2, elem_classes=["panel-card", "verdict-output"]):
                gr.HTML('<div class="panel-label">02 — Analysis Verdict</div>')
                verdict_out = gr.Markdown(
                    value="*Upload a file and click **Run Forensic Analysis** to begin.*"
                )

        # ── Divider ────────────────────────────────────────────────────────────
        gr.HTML('<hr class="section-divider" />')

        # ── XAI Visualisations ─────────────────────────────────────────────────
        gr.HTML('<div class="panel-label" style="margin-bottom:20px;">03 — Explainability Visualisations</div>')
        with gr.Row(equal_height=True, elem_classes="xai-panel"):
            with gr.Column(elem_classes="panel-card"):
                gradcam_out = gr.Image(
                    label="Grad-CAM — Suspicious Frequency-Time Regions",
                    type="filepath",
                    buttons=["download"],
                )
            with gr.Column(elem_classes="panel-card"):
                shap_out = gr.Image(
                    label="SHAP — Feature Importance per LFCC Coefficient",
                    type="filepath",
                    buttons=["download"],
                )

        # ── Divider ────────────────────────────────────────────────────────────
        gr.HTML('<hr class="section-divider" />')

        # ── Legend & specs ──────────────────────────────────────────────────────
        gr.HTML(_LEGEND_MD)

        # ── Examples ───────────────────────────────────────────────────────────
        gr.Examples(
            examples=[],
            inputs=[audio_input],
            label="Example Files  (add .wav files to app/examples/)",
        )

    # ── Event binding — untouched ──────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input],
        outputs=[verdict_out, gradcam_out, shap_out, status_box],
    )


if __name__ == "__main__":
    port  = int(_cfg["app"]["port"])
    share = bool(_cfg["app"]["share"])
    logger.info("Launching Forensic Audio Authentication app on port %d", port)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
        theme=_THEME,
        css=CSS,
    )