"""
llm_feedback.py
Gemini 2.5 Flash feedback layer — grounded critique of one creative.
Sends: original + Grad-CAM overlay + score + feature deltas vs top-performer stats.
Returns markdown with 3 sections: Strengths · Weaknesses · Alternatives.

Requires:
- pip install google-genai
- .streamlit/secrets.toml with:
    GEMINI_API_KEY = "AIza..."
- data/top_performer_stats.json (from compute_top_performer_stats.py)
"""

import io
import json
from pathlib import Path
from typing import Dict, Any

import streamlit as st
from PIL import Image

STATS_JSON = Path("data/top_performer_stats.json")
MODEL      = "gemini-3.6-flash"
MAX_TOKENS = 700

# features we care about explaining (subset of full spec — keeps prompt tight)
FEATURES_TO_SHOW = [
    "brightness", "contrast", "saturation", "colorfulness",
    "edge_density", "warm_ratio", "text_area_frac",
    "face_count", "face_area_frac",
    "title_length", "title_word_count",
]


@st.cache_data(show_spinner=False)
def _load_stats() -> Dict[str, Any]:
    if not STATS_JSON.exists():
        return {}
    return json.loads(STATS_JSON.read_text())


def _downscale(pil_img: Image.Image, max_side: int = 512) -> Image.Image:
    """Downscale for cheap vision input; return fresh PIL image."""
    img = pil_img.convert("RGB").copy()
    img.thumbnail((max_side, max_side))
    return img


def _feature_delta_table(cv: dict, meta: dict) -> str:
    """Build 'this vs top-performer median' comparison string for grounding."""
    stats = _load_stats().get("features", {})
    if not stats:
        return "(top-performer reference stats unavailable)"

    all_feats = {**cv, **meta}
    lines = ["| feature | this creative | top-performer median | delta |",
             "|---|---|---|---|"]
    for f in FEATURES_TO_SHOW:
        if f not in all_feats or f not in stats:
            continue
        v_this = all_feats[f]
        v_top  = stats[f]["median"]
        if v_top == 0:
            delta = "—"
        else:
            pct = 100 * (v_this - v_top) / abs(v_top)
            arrow = "↑" if pct > 0 else "↓"
            delta = f"{arrow} {abs(pct):.0f}%"
        lines.append(f"| {f} | {v_this:.3f} | {v_top:.3f} | {delta} |")
    return "\n".join(lines)


def _build_prompt(title: str, prob: float, cnn_prob: float,
                  cv: dict, meta: dict) -> str:
    delta_table = _feature_delta_table(cv, meta)
    return f"""You are a paid-ad creative critic. Analyze the two attached images.
The FIRST image is the ad creative. The SECOND image is a Grad-CAM overlay showing
where the CNN reacted (red/warm = high attention).

**Context:**
- Title on ad: "{title}"
- Model score (calibrated probability of beating this channel's own trailing-median views): {prob:.3f}
  (0.5 = coin flip; below 0.5 = predicted underperform; above = predicted overperform)

**How this creative compares to historical top performers (overperformed = 1) on measurable features:**

{delta_table}

**Your task:** Give feedback in exactly these 3 sections, using markdown headers:

### ✅ Strengths (keep)
2-3 bullets — what's working, tied to the feature deltas or visible design where possible.

### ⚠️ Weaknesses (fix)
2-3 bullets — what's dragging the score down. Reference specific feature gaps when relevant.

### 🔄 Alternatives (try)
2-3 concrete A/B test suggestions — specific enough that a designer could act on them tomorrow.

**Rules:**
- Be concrete. "Increase contrast" is bad; "Darker background behind headline text" is good.
- Ground claims in the feature table when you can.
- Do NOT restate the score or repeat the table.
- Total under 250 words. No preamble."""


def get_feedback(pil_img: Image.Image, cam_overlay: Image.Image,
                 title: str, prob: float, cnn_prob: float,
                 meta: dict, cv: dict) -> str:
    """Call Gemini 2.5 Flash with grounded prompt. Returns markdown."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "❌ `google-genai` not installed. Run: `pip install google-genai`"

    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        return ("❌ No API key. Add to `.streamlit/secrets.toml`:\n\n"
                '```toml\nGEMINI_API_KEY = "AIza..."\n```')

    client = genai.Client(api_key=api_key)

    orig    = _downscale(pil_img)
    overlay = _downscale(cam_overlay)
    prompt  = _build_prompt(title, prob, cnn_prob, cv, meta)

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=[orig, overlay, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS,
                temperature=0.4,
            ),
        )
        return resp.text
    except Exception as e:
        return f"❌ API error: {type(e).__name__}: {e}"
