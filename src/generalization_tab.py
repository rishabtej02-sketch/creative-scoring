"""
generalization_tab.py
Renders the "Generalization demo" tab for app.py.

Auto-selects data source:
  - if data/sample_scores.csv exists → uses curated sample (for HF deploy)
  - else → uses full data/paid_ad_scores.csv (local dev)
"""

from pathlib import Path
import pandas as pd
import streamlit as st

# ---- paths ----
FIG_3WAY      = Path("figs/score_dist_3way.png")
FIG_PER_BRAND = Path("figs/score_dist_per_brand.png")
BRAND_TABLE   = Path("data/brand_spread_table.csv")

FULL_SCORES   = Path("data/paid_ad_scores.csv")
SAMPLE_SCORES = Path("data/sample_scores.csv")
FULL_META_DIR = Path("data/meta_ads/images")
FULL_GOOG_DIR = Path("data/google_ads/images")
SAMPLE_IMG    = Path("data/sample_images")  # sample_images/{source}/{brand}/{cid}.ext


def _using_sample() -> bool:
    return SAMPLE_SCORES.exists() and SAMPLE_IMG.exists()


def _scores_path() -> Path:
    return SAMPLE_SCORES if _using_sample() else FULL_SCORES


def _section_header():
    st.header("Generalization demo → real paid ads")
    st.markdown(
        "Model trained on **YouTube thumbnails** (organic, has labels). "
        "Applied to **Meta + Google paid ads** (no perf labels). "
        "This tab shows how scores transfer across domains."
    )
    st.divider()


def _headline_finding():
    st.subheader("Headline: 3× compression, ranking preserved")
    if FIG_3WAY.exists():
        st.image(str(FIG_3WAY), caption="Score distribution: YT vs Meta vs Google")
    else:
        st.warning(f"missing → {FIG_3WAY}")

    st.markdown(
        "- **σ shrinks ~3×** on paid ads (YT 0.164 → paid ~0.055)\n"
        "- **Means aligned** (~0.45 across all 3 sources)\n"
        "- **Relative ranking preserved** within each source\n"
        "- Paid ads = polished graphics → model less differentiating, "
        "but *ordering* of good vs bad creative still holds"
    )


def _per_brand_view():
    st.subheader("Per-brand mean scores")
    if BRAND_TABLE.exists():
        df = pd.read_csv(BRAND_TABLE)
        st.dataframe(df, width="stretch", height=300)

        if {"brand", "source", "mean_score"}.issubset(df.columns):
            pivot = df.pivot(index="brand", columns="source", values="mean_score")
            st.bar_chart(pivot)
        elif FIG_PER_BRAND.exists():
            st.image(str(FIG_PER_BRAND))
    else:
        st.warning(f"missing → {BRAND_TABLE}")


@st.cache_data
def _load_scores(path_str: str):
    return pd.read_csv(path_str)


def _resolve_img(source: str, brand: str, cid: str):
    """Try sample dir first, then fall back to full local paths."""
    exts = (".jpg", ".png", ".jpeg")
    if _using_sample():
        for ext in exts:
            p = SAMPLE_IMG / source / brand / f"{cid}{ext}"
            if p.exists():
                return p
    root = FULL_META_DIR if source == "meta" else FULL_GOOG_DIR
    for ext in exts:
        p = root / brand / f"{cid}{ext}"
        if p.exists():
            return p
    return None


def _sample_gallery():
    st.subheader("Sample: top & bottom scored real ads")

    scores_path = _scores_path()
    if not scores_path.exists():
        st.info(f"no scores file → {scores_path}")
        return

    if _using_sample():
        st.caption("Showing curated sample (5 hero brands, top/bottom scored ads).")

    df = _load_scores(str(scores_path))

    source = st.radio(
        "Source",
        sorted(df["source"].unique()),
        horizontal=True,
        key="gen_src",
    )
    src_df = df[df["source"] == source]

    brands = sorted(src_df["brand"].dropna().unique())
    brand  = st.selectbox("Brand", brands, key=f"gen_brand_{source}")

    sub = src_df[src_df["brand"] == brand].sort_values("score", ascending=False)
    st.caption(
        f"{len(sub)} ads → μ={sub['score'].mean():.3f}, σ={sub['score'].std():.3f}, "
        f"range {sub['score'].min():.3f}–{sub['score'].max():.3f}"
    )

    max_n = max(1, min(5, len(sub) // 2))
    n = st.slider("Show top/bottom N", 1, max_n, min(3, max_n),
                  key=f"gen_n_{source}")
    top, bot = sub.head(n), sub.tail(n)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top scored**")
        _render_ads(top, source, brand)
    with col_b:
        st.markdown("**Bottom scored**")
        _render_ads(bot, source, brand)


def _render_ads(rows: pd.DataFrame, source: str, brand: str):
    for _, r in rows.iterrows():
        cid = r["creative_id"]
        p = _resolve_img(source, brand, cid)
        if p:
            st.image(str(p), caption=f"score={r['score']:.3f}")
        else:
            st.caption(f"{cid} → img not found (score={r['score']:.3f})")


def _framing_note():
    st.divider()
    st.info(
        "**Framing:** Meta + Google = *application set*, not test set. "
        "No perf labels available (Google Ads Transparency has no first-shown "
        "dates for Indian commercial ads). Distribution shift is a *finding*, "
        "not a bug — model was trained on organic thumbnails, paid ads are "
        "professionally polished → naturally compressed score range."
    )


def render():
    """Entry point called from app.py tab context."""
    _section_header()
    _headline_finding()
    st.divider()
    _per_brand_view()
    st.divider()
    _sample_gallery()
    _framing_note()


if __name__ == "__main__":
    st.set_page_config(page_title="Generalization demo", layout="wide")
    render()
