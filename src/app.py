"""Pre-flight creative scoring - Streamlit demo (v2, Phase 3c).

Every number on screen comes from one canonical extractor (`scorer.py`).
This file reimplements NO feature. It only loads, calls, and renders.

Tabs:
  1 Score & allocate  - upload 1-6 drafts, calibrated score, percentile in the
                        brand's own pool, budget split
  2 Deep dive         - spec check, Grad-CAM, similar winners, AI feedback
  3 Model comparison  - same creative, same evidence, several models
  4 Generalization    - existing YouTube vs Meta vs Google distribution demo

Two numbers stay separate on purpose:
  "Will it RUN?"     -> platform_specs.check_image
  "Will it PERFORM?" -> scorer.score_image

Run:
    cd ~/creative-scoring
    streamlit run src/app.py
"""
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import rag_chain
import scorer
from brand_profiles import BrandProfiles, norm_brand
from generalization_tab import render as render_gen_tab
from platform_specs import check_all, check_image, platform_list

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

IMG_DIRS = {
    "meta": DATA / "meta_ads" / "images",
    "google": DATA / "google_ads" / "images",
}
PLATFORM_LABEL = {"youtube": "YouTube", "meta": "Meta", "google": "Google"}
DEFAULT_MODEL_HINT = "flash lite"
TMP = Path(tempfile.gettempdir()) / "creative_scoring_uploads"
TMP.mkdir(parents=True, exist_ok=True)


# ============================================================ cached loaders
@st.cache_resource(show_spinner=False)
def load_bundle():
    """One CLIP load for the whole session. Without the cache every widget
    interaction reloads 600 MB of weights."""
    return scorer.load_all(with_clip=True, verbose=False)


@st.cache_resource(show_spinner=False)
def load_profiles():
    try:
        return BrandProfiles.load()
    except Exception as e:
        st.session_state["_profiles_err"] = str(e)
        return None


@st.cache_resource(show_spinner=False)
def load_retriever():
    try:
        from retrieval import Retriever
        return Retriever.load()
    except Exception as e:
        st.session_state["_retriever_err"] = str(e)
        return None


@st.cache_resource(show_spinner=False)
def load_cnn():
    """Lazy - only built when a Grad-CAM panel is actually opened."""
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small
    ckpt = torch.load(MODELS / "cnn_gradcam.pt", map_location="cpu",
                      weights_only=False)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def model_labels():
    """Dropdown options straight from the registry, which itself is built from
    data/llm_models.json. Nothing about models is hardcoded here."""
    try:
        from llm_registry import available
        specs = available()
    except Exception:
        return [], {}
    labels = [s.label for s in specs]
    meta = {s.label: {"vision": bool(s.vision), "provider": s.provider,
                      "notes": getattr(s, "notes", "")} for s in specs}
    return labels, meta


@st.cache_data(show_spinner=False)
def brand_catalog():
    """Brand keys live in brand_profiles.json; image folders live on disk with
    underscores (amazon_india vs amazonindia). Resolve the two at runtime with
    norm_brand instead of keeping a hand-written map that will rot."""
    bp = load_profiles()
    out = {}
    if bp is None:
        return out
    for platform, pool in bp.pools.items():
        dirmap = {}
        base = IMG_DIRS.get(platform)
        if base and base.exists():
            for d in base.iterdir():
                if d.is_dir():
                    dirmap[norm_brand(d.name)] = str(d)
        entries = []
        for key, e in pool.items():
            if key == "_global" or not isinstance(e, dict):
                continue
            entries.append({
                "key": key,
                "display": e.get("display", key),
                "n": int(e.get("n", 0)),
                "dir": dirmap.get(key, ""),
            })
        entries.sort(key=lambda x: x["display"].lower())
        out[platform] = entries
    return out


# ============================================================ allocation
def allocate(weights, total, min_frac=None, max_frac=None, power=1.0):
    """Proportional split with a floor and a ceiling. The floor keeps a
    promising creative from being starved of the impressions it needs to prove
    itself; the ceiling stops one uncertain bet taking the whole budget."""
    w = np.asarray(weights, dtype=float)
    if len(w) == 0:
        return []
    if power == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = np.power(np.maximum(w, 1e-9), power)
        w = w / w.sum()
    if min_frac is None and max_frac is None:
        return (w * total).tolist()
    lo = min_frac if min_frac is not None else 0.0
    hi = max_frac if max_frac is not None else 1.0
    lo = min(lo, 1.0 / len(w))
    hi = max(hi, 1.0 / len(w))
    for _ in range(50):
        clipped = np.clip(w, lo, hi)
        diff = 1.0 - clipped.sum()
        free = (w > lo) & (w < hi)
        if not free.any() or abs(diff) < 1e-9:
            w = clipped
            break
        w = clipped.copy()
        w[free] += diff * (w[free] / max(w[free].sum(), 1e-9))
    w = np.clip(w, lo, hi)
    w = w / w.sum()
    return (w * total).tolist()


# ============================================================ Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._a)
        self.h2 = target_layer.register_full_backward_hook(self._g)

    def _a(self, m, i, o):
        self.activations = o.detach()

    def _g(self, m, gi, go):
        self.gradients = go[0].detach()

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x):
        import torch
        import torch.nn.functional as F
        self.model.zero_grad()
        logit = self.model(x)
        s = logit.squeeze()
        s.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam - cam.min()
        if float(cam.max()) > 0:
            cam = cam / cam.max()
        return cam.squeeze().cpu().numpy(), float(torch.sigmoid(s).item())


def gradcam_overlay(pil, alpha=0.45):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    model = load_cnn()
    cam_obj = GradCAM(model, model.features[-1])
    try:
        cam, prob = cam_obj(tf(pil).unsqueeze(0))
    finally:
        cam_obj.close()
    W, H = pil.size
    heat = cv2.applyColorMap(
        cv2.resize((cam * 255).astype(np.uint8), (W, H),
                   interpolation=cv2.INTER_LINEAR), cv2.COLORMAP_JET)
    heat_pil = Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
    return Image.blend(pil.convert("RGB"), heat_pil, alpha=alpha), prob


# ============================================================ small helpers
def save_upload(up) -> Path:
    p = TMP / up.name
    with open(p, "wb") as f:
        f.write(up.getbuffer())
    return p


def rank_dict(rank):
    return asdict(rank) if is_dataclass(rank) else dict(rank or {})


def spec_badge(status):
    return {"pass": "OK", "warn": "WARN", "fail": "FAIL"}.get(status, status)


def resolve_hit_path(raw):
    """clip_paid_meta.csv stores Windows backslash relative paths."""
    if not raw:
        return None
    p = Path(str(raw).replace("\\", "/"))
    cand = p if p.is_absolute() else ROOT / p
    return cand if cand.exists() else None


def pretty_model(label, meta):
    m = meta.get(label, {})
    tag = " [borrowed capacity]" if m.get("provider") == "openai" and \
        "OpenRouter" in label else ""
    if "OpenRouter" in label:
        tag = " [borrowed capacity]"
    return f"{label}{tag}"


def pick_default_model(labels):
    for lb in labels:
        if DEFAULT_MODEL_HINT in lb.lower() and "[text-only]" not in lb:
            return lb
    for lb in labels:
        if "(Google)" in lb and "[text-only]" not in lb:
            return lb
    return labels[0] if labels else None


def build_ctx(r, k, winners_only, use_ocr):
    """One assembly point. Retrieval, percentile, spec and features all reach
    the model through rag_chain.build_context, exactly as the CLI does."""
    ocr_text = ""
    if use_ocr:
        try:
            ocr_text = rag_chain._ocr(str(r["path"]), verbose=False) or ""
        except Exception as e:
            ocr_text = ""
            st.caption(f"OCR unavailable: {str(e)[:80]}")
    return rag_chain.build_context(
        score=r["score"], platform=r["platform"], brand=r["brand_key"],
        own_feats=r["cv"], ocr_text=ocr_text, image_path=str(r["path"]),
        k=k, spec_report=r.get("spec_report"),
        clip_vec=r.get("clip_vec"), verbose=False)


# ============================================================ UI shell
st.set_page_config(page_title="Pre-Flight Creative Scoring", layout="wide")
st.title("Pre-Flight Creative Scoring")
st.caption("Score draft creatives before any budget is committed. "
           "Scores are within-brand, within-format - they measure creative "
           "quality, not brand size.")

with st.expander("What this tool does (and does not) do"):
    st.markdown("""
- **Does:** rank *your own* drafts by predicted probability of beating your own
  channel's trailing-median baseline, place each in your brand's own score
  pool, and split budget on that ranking.
- **Does not:** predict virality, or compare you against bigger brands - that
  would just measure brand size.
- **Two separate numbers.** *Will it run?* is the platform spec check.
  *Will it perform?* is the model score. They are never merged.
- **Model:** CLIP + metadata + CV + title TF-IDF -> LightGBM -> isotonic
  calibration. Held-out test AUC = 0.615. Every individual signal lands near
  0.56; stacking reaches 0.615. That is the information ceiling once brand
  size is normalised out, not a pipeline failure.
- **Labels:** only the YouTube pool carries measured overperformance labels.
  Meta and Google pools are unlabeled reference distributions - a percentile
  there means *more winner-like than X%*, never *beats X%*.
- **Heatmap:** shows *where* a weaker CNN (AUC 0.565) reacted. Correlation,
  not instruction.
""")

bundle_ready = (MODELS / "calibrated_model.pkl").exists()
if not bundle_ready:
    st.error("models/calibrated_model.pkl not found. Run the training pipeline "
             "first.")
    st.stop()

catalog = brand_catalog()
labels, lmeta = model_labels()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Creative context")
    platforms = [p for p in ("youtube", "meta", "google") if p in catalog] or \
        ["youtube"]
    platform = st.selectbox("Platform", platforms,
                            format_func=lambda p: PLATFORM_LABEL.get(p, p))
    entries = catalog.get(platform, [])
    brand_opts = [""] + [e["key"] for e in entries]
    disp = {e["key"]: f"{e['display']}  (n={e['n']})" for e in entries}
    brand_key = st.selectbox(
        "Brand pool", brand_opts,
        format_func=lambda k: disp.get(k, "- none / unknown brand -"))
    if brand_key == "":
        st.caption("No brand selected -> ranked against the whole platform "
                   "pool instead.")

    st.header("Budget")
    total_budget = st.number_input("Total budget (Rs)", min_value=1000,
                                   value=100000, step=1000)
    basis = st.radio(
        "Allocate on",
        ["Percentile in brand pool", "Raw calibrated score"],
        help="The percentile is comparable across brands; the raw score is "
             "compressed on paid platforms (Meta 3.02x, Google 2.96x tighter "
             "than YouTube), so raw-score splits look flatter than they are.")
    power = st.slider("Aggressiveness (0 = equal, 1 = proportional, "
                      "2 = winner-heavy)", 0.0, 2.0, 1.0, 0.1)
    min_frac = st.slider("Min share per creative", 0.0, 0.5, 0.10, 0.05)
    max_frac = st.slider("Max share per creative", 0.5, 1.0, 0.60, 0.05)

    st.header("Evidence")
    winners_only = st.toggle("Retrieve winners only", value=True,
                             help="On: neighbours come from creatives that beat "
                                  "their own baseline. Off: the full deduped "
                                  "index (25,514 rows).")
    k_neighbours = st.slider("Neighbours (k)", 3, 10, 5)
    use_ocr = st.toggle("Run OCR on upload", value=False,
                        help="Adds on-image text to the prompt. Slow on first "
                             "run (EasyOCR model download).")

    st.header("AI feedback")
    if labels:
        default_lb = pick_default_model(labels)
        model_label = st.selectbox(
            "Model", labels, index=labels.index(default_lb) if default_lb else 0,
            format_func=lambda lb: pretty_model(lb, lmeta))
        st.caption("Google / Groq run on our own quota. OpenRouter free models "
                   "are borrowed shared capacity - fine for a fallback, never "
                   "the primary in a live demo.")
    else:
        model_label = None
        st.warning("No LLM keys found in the environment. Feedback disabled.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["1 - Score & allocate", "2 - Deep dive", "3 - Model comparison",
     "4 - Generalization"])

# ================================================================ TAB 1
with tab1:
    st.header("Upload draft creatives")
    uploads = st.file_uploader("PNG / JPG. 1-6 files.",
                               type=["png", "jpg", "jpeg"],
                               accept_multiple_files=True)

    titles = []
    if uploads:
        if platform == "youtube":
            st.subheader("Title for each creative")
            st.caption("Titles feed the TF-IDF text arm of the model.")
            cols = st.columns(min(3, len(uploads)))
            for i, up in enumerate(uploads):
                with cols[i % len(cols)]:
                    st.image(up, caption=up.name, width="stretch")
                    titles.append(st.text_input(f"Title - {up.name}",
                                                key=f"title_{i}", value=""))
        else:
            titles = ["" for _ in uploads]
            cols = st.columns(min(3, len(uploads)))
            for i, up in enumerate(uploads):
                with cols[i % len(cols)]:
                    st.image(up, caption=up.name, width="stretch")
            st.caption("Paid creatives are scored with an empty title. Meta and "
                       "Google ad_text fields carry legal disclosure, not "
                       "YouTube-equivalent title signal - feeding them in would "
                       "add noise, not information.")

        if st.button("Score creatives", type="primary"):
            with st.spinner("Loading CLIP + calibrated model (first run only)..."):
                bundle = load_bundle()
            bp = load_profiles()

            results = []
            prog = st.progress(0.0, text="Scoring...")
            for i, up in enumerate(uploads):
                path = save_upload(up)
                title = titles[i] if i < len(titles) else ""
                score, meta, cv, clip_vec = scorer.score_image(
                    str(path), title=title, bundle=bundle)

                pil = Image.open(path).convert("RGB")
                text_pct = float(cv.get("text_area_frac", 0.0)) * 100.0
                try:
                    reports = check_all(str(path), text_area_pct=text_pct)
                except Exception:
                    reports = {}
                own_spec = reports.get(platform)
                if own_spec is None and reports:
                    own_spec = list(reports.values())[0]

                rank = None
                if bp is not None:
                    try:
                        rank = bp.percentile(score, platform,
                                             brand_key or None)
                    except Exception:
                        rank = None

                results.append({
                    "name": up.name, "path": path, "title": title,
                    "pil": pil, "score": score, "meta": meta, "cv": cv,
                    "clip_vec": clip_vec, "platform": platform,
                    "brand_key": brand_key or None,
                    "spec_report": own_spec, "spec_all": reports,
                    "rank": rank, "rank_d": rank_dict(rank) if rank else {},
                })
                prog.progress((i + 1) / len(uploads),
                              text=f"Scored {i+1}/{len(uploads)}")
            prog.empty()

            st.session_state["results"] = results
            st.session_state["total_budget"] = total_budget
            for k in [k for k in st.session_state if k.startswith("fb_")]:
                del st.session_state[k]

    if "results" in st.session_state:
        results = st.session_state["results"]
        total_budget = st.session_state.get("total_budget", total_budget)

        if basis.startswith("Percentile") and all(r["rank"] for r in results):
            weights = [max(r["rank"].pct, 1.0) for r in results]
            basis_note = "percentile inside the selected pool"
        else:
            weights = [r["score"] for r in results]
            basis_note = "raw calibrated score"
        amounts = allocate(weights, total_budget, min_frac=min_frac,
                           max_frac=max_frac, power=power)
        for r, a in zip(results, amounts):
            r["amount"] = a

        st.header("Results")
        rows = []
        for r in results:
            d = r["rank_d"]
            rows.append({
                "creative": r["name"],
                "score": round(r["score"], 4),
                "percentile": (f"{r['rank'].pct:.1f}" if r["rank"] else "-"),
                "band": d.get("band", "-"),
                "pool": d.get("name", d.get("brand", "-")),
                "pool n": d.get("n", "-"),
                "spec": (spec_badge(r["spec_report"].verdict)
                         if r.get("spec_report") else "-"),
                "budget (Rs)": round(r["amount"]),
                "share": f"{r['amount']/total_budget:.1%}",
            })
        summary = pd.DataFrame(rows).sort_values("score", ascending=False)
        st.dataframe(summary, width="stretch", hide_index=True)
        st.caption(f"Budget split on **{basis_note}**, power={power:.1f}, "
                   f"floor={min_frac:.0%}, ceiling={max_frac:.0%}.")

        caveats = {r["rank_d"].get("caveat", "") for r in results
                   if r.get("rank_d")}
        for c in sorted(x for x in caveats if x):
            st.warning(c)

        st.subheader("Score comparison")
        st.bar_chart(pd.DataFrame({"score": [r["score"] for r in results]},
                                  index=[r["name"] for r in results]))
        st.caption("Open **Deep dive** for spec checks, heatmaps, retrieved "
                   "winners and AI feedback on any single creative.")
    else:
        st.info("Upload creatives above to begin.")

# ================================================================ TAB 2
with tab2:
    if "results" not in st.session_state:
        st.info("Score some creatives first (tab 1).")
    else:
        results = st.session_state["results"]
        names = [r["name"] for r in results]
        pick = st.selectbox("Creative", names, key="dd_pick")
        r = next(x for x in results if x["name"] == pick)
        d = r["rank_d"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Calibrated score", f"{r['score']:.4f}")
        c2.metric("Percentile in pool",
                  f"{r['rank'].pct:.1f}" if r["rank"] else "-",
                  help=d.get("caveat", ""))
        c3.metric("Spec verdict",
                  spec_badge(r["spec_report"].verdict)
                  if r.get("spec_report") else "-")
        if r["rank"]:
            try:
                st.caption(r["rank"].sentence(d.get("name")))
            except Exception:
                pass

        st.divider()
        st.subheader("Will it RUN? - platform spec check")
        st.caption("Independent of the score. A creative that cannot run is not "
                   "improved by better colour.")
        for pkey, rep in (r.get("spec_all") or {}).items():
            with st.expander(f"{rep.display} - {spec_badge(rep.verdict)}",
                             expanded=(pkey == r["platform"])):
                st.write(rep.summary())
                bad = [c for c in rep.checks if c.status != "pass"]
                if not bad:
                    st.success("All checks pass.")
                for c in bad:
                    line = f"**[{spec_badge(c.status)}] {c.name}** - {c.detail}"
                    if getattr(c, "fix", ""):
                        line += f"  \nFix: {c.fix}"
                    (st.error if c.status == "fail" else st.warning)(line)

        st.divider()
        st.subheader("Where the CNN looked")
        if st.button("Compute Grad-CAM", key=f"cam_{pick}"):
            with st.spinner("Loading MobileNetV3 + backprop..."):
                try:
                    st.session_state[f"cam_img_{pick}"] = gradcam_overlay(r["pil"])
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")
        if f"cam_img_{pick}" in st.session_state:
            overlay, cnn_prob = st.session_state[f"cam_img_{pick}"]
            a, b = st.columns(2)
            a.image(r["pil"], caption="Original", width="stretch")
            b.image(overlay, caption="Grad-CAM (red = attention)",
                    width="stretch")
            st.caption(f"CNN probability {cnn_prob:.3f} (AUC 0.565 - this model "
                       "exists for the heatmap, not for the score).")

        st.divider()
        st.subheader("Measured features")
        with st.expander("Feature values"):
            st.json({"metadata (defaults at inference)": r["meta"],
                     "CV (from this image)": r["cv"]})

        st.divider()
        st.subheader("Similar high-performing creatives")
        if st.button("Retrieve neighbours", key=f"rag_{pick}"):
            ret = load_retriever()
            if ret is None:
                st.error("No index. Run: python src/retrieval.py --build")
            else:
                st.session_state[f"hits_{pick}"] = ret.search(
                    r["clip_vec"], k=k_neighbours, winners_only=winners_only,
                    brand=r["brand_key"], platform=r["platform"])
        hits = st.session_state.get(f"hits_{pick}")
        if hits:
            plats = {h.platform for h in hits}
            if r["platform"] not in plats:
                st.info(f"Cross-domain retrieval: neighbours come from "
                        f"{sorted(plats)}, the upload is {r['platform']}. Only "
                        "YouTube carries performance labels, so the winners "
                        "index skews there. Treat these as visual analogues, "
                        "not same-platform results.")
            cols = st.columns(min(5, len(hits)))
            for i, h in enumerate(hits):
                with cols[i % len(cols)]:
                    p = resolve_hit_path(h.path)
                    if p:
                        st.image(str(p), width="stretch")
                    st.caption(f"{h.brand} / {h.platform}  \n"
                               f"sim {h.sim:.3f} - score {h.score:.3f}")
            from retrieval import Retriever as _R
            st.code(_R.prompt_block(hits, r["cv"]), language="text")

        st.divider()
        st.subheader("AI feedback")
        if not model_label:
            st.warning("No model available - set GOOGLE_API_KEY / GROQ_API_KEY.")
        else:
            fb_key = f"fb_{pick}"
            allow_fb = st.toggle("Allow fallback to other providers on failure",
                                 value=True, key=f"fbk_{pick}")
            if st.button(f"Get feedback - {model_label}", key=f"btn_{pick}",
                         type="primary"):
                with st.spinner("Assembling grounded context and calling the model..."):
                    ctx = build_ctx(r, k_neighbours, winners_only, use_ocr)
                    order = [model_label] + (
                        [lb for lb in labels if lb != model_label]
                        if allow_fb else [])
                    vision = lmeta.get(model_label, {}).get("vision", False)
                    fb, used, errs = rag_chain.run_feedback_fallback(
                        order, ctx, image_path=str(r["path"]) if vision else None,
                        verbose=False)
                    st.session_state[fb_key] = (fb, used, errs)
            if fb_key in st.session_state:
                fb, used, errs = st.session_state[fb_key]
                if fb is None:
                    st.error("Every model failed.")
                    for lb, e in errs:
                        st.caption(f"{lb}: {str(e)[:200]}")
                else:
                    if used != model_label:
                        st.warning(f"Fell back to **{used}** - "
                                   f"{model_label} was unavailable.")
                    if errs:
                        with st.expander("Fallback trace"):
                            for lb, e in errs:
                                st.caption(f"{lb}: {str(e)[:200]}")
                    st.markdown(rag_chain.render(fb))

# ================================================================ TAB 3
with tab3:
    st.header("Same creative, same evidence, different models")
    st.caption("The supervised model produces the numbers. The generative "
               "layer turns them into advice. Running a vision model beside a "
               "text-only one shows which claims came from the pixels and which "
               "came from the feature table.")
    if "results" not in st.session_state:
        st.info("Score some creatives first (tab 1).")
    elif not labels:
        st.warning("No LLM keys found.")
    else:
        results = st.session_state["results"]
        pick2 = st.selectbox("Creative", [r["name"] for r in results],
                             key="cmp_pick")
        r2 = next(x for x in results if x["name"] == pick2)
        vis = [lb for lb in labels if lmeta.get(lb, {}).get("vision")]
        txt = [lb for lb in labels if not lmeta.get(lb, {}).get("vision")]
        default_pick = [x for x in (vis[:1] + txt[:1]) if x]
        chosen = st.multiselect("Models (2-3)", labels, default=default_pick,
                                format_func=lambda lb: pretty_model(lb, lmeta))
        if st.button("Run comparison", type="primary", key="cmp_run"):
            if not 1 < len(chosen) <= 4:
                st.warning("Pick 2-4 models.")
            else:
                with st.spinner(f"Calling {len(chosen)} models..."):
                    ctx = build_ctx(r2, k_neighbours, winners_only, use_ocr)
                    out = {}
                    for lb in chosen:
                        v = lmeta.get(lb, {}).get("vision", False)
                        one = rag_chain.compare_models(
                            [lb], ctx,
                            image_path=str(r2["path"]) if v else None)
                        out.update(one)
                    st.session_state["cmp_out"] = out
        if "cmp_out" in st.session_state:
            out = st.session_state["cmp_out"]
            cols = st.columns(len(out))
            for col, (lb, res) in zip(cols, out.items()):
                with col:
                    v = lmeta.get(lb, {}).get("vision", False)
                    st.markdown(f"**{lb}**")
                    st.caption("sees the image" if v
                               else "numbers only (ablation arm)")
                    if res["feedback"] is None:
                        st.error(str(res["error"])[:300])
                    else:
                        st.markdown(rag_chain.render(res["feedback"]))
            st.info("Expected pattern: the numbers alone reproduce most of the "
                    "feedback; the vision arm adds object-level claims it can "
                    "only get from pixels. Disagreement on a feature the text "
                    "arm can see is a warning about that arm, not about the "
                    "feature.")

# ================================================================ TAB 4
with tab4:
    render_gen_tab()

st.divider()
st.caption(
    "Scores are calibrated probabilities of beating a channel's own trailing "
    "median, within format. Test AUC 0.615. Meta and Google are an application "
    "set, not a test set - no performance labels are published for them. "
    "Treat every number here as a hint about what to test, not a guarantee."
)
