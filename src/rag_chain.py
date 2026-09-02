#!/usr/bin/env python3
"""
rag_chain.py - grounded creative feedback via LangChain.

THE PROBLEM THIS SOLVES:
The old feedback path sent the image plus one static top_performer_stats.json
to Gemini. Every upload got the same reference numbers, so the advice came
back generic - "increase contrast, reduce text" - true of almost any ad and
therefore worth nothing.

WHAT REPLACES IT - four grounding sources assembled per upload:
  1. RETRIEVED  k nearest high-performing creatives (CLIP + retrieval.py),
                with feature deltas vs the upload. This is the RAG part.
  2. RANKED     the brand-and-platform percentile (brand_profiles.py).
  3. CHECKED    the spec verdict (platform_specs.py).
  4. MEASURED   the upload's own CV features and OCR text.

The model is told, explicitly, to cite which of these it used per claim and
to say "not supported by the retrieved evidence" rather than invent. That
instruction is why the output schema has a `grounded_on` field: it makes
ungroundedness visible instead of letting it hide in fluent prose.

CHAIN: prompt | llm | structured parser   (LCEL, so .batch() runs the model
comparison tab in parallel for free)

--auto  (added Phase 3c)
Without it the CLI passed --score (default 0.50, a placeholder) and no
own_feats, so the MEASURED block was empty and the score in the prompt was
fabricated. --auto routes the image through scorer.py instead: one CLIP load
returns the real calibrated score, the nine canonical CV features and the
embedding, so retrieval, the spec check and the prompt all read from the same
extraction. scorer.py is the only module permitted to compute CV features
(verify_parity.py enforces < 1e-6 against features.csv).

CLI:
    python src/rag_chain.py --auto --dry-run ad.jpg --platform meta --brand zomato
    python src/rag_chain.py --auto --demo ad.jpg --platform meta --brand zomato
    python src/rag_chain.py --auto --demo ad.jpg --platform meta --compare
    python src/rag_chain.py --dry-run ad.jpg --score 0.61   # manual score
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from typing import List, Optional

# Windows consoles default to cp1252. Neighbour titles are real ad copy and
# carry emoji - one cloud glyph in a retrieved YouTube title crashed the
# --dry-run print with UnicodeEncodeError. The prompt itself was fine; only
# stdout could not render it. Never let a display encoding kill a run.
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydantic import BaseModel, Field  # noqa: E402

# Labeled winners exist only for YouTube - Meta and Google carry no
# performance labels, so the retrieval index's "winners" pool is
# YouTube-only. The model must be told, or it will read a cross-domain
# neighbour as a same-platform benchmark.
CROSS_DOMAIN_NOTE = (
    "NOTE ON THE RETRIEVED SET: performance labels exist only for YouTube "
    "thumbnails. Meta and Google ads were collected without any outcome data, "
    "so every 'high-performing' neighbour above is a YouTube creative, "
    "whatever platform this upload targets. Treat them as evidence about "
    "which visual patterns correlate with overperformance in general, not as "
    "a same-platform benchmark. Do not claim a neighbour performed well on "
    "the platform under review."
)


# ---------------------------------------------------------------- schema
class Fix(BaseModel):
    change: str = Field(description="One specific, executable change. "
                                    "Name the element and the direction.")
    why: str = Field(description="Which retrieved evidence motivates it. "
                                 "Cite neighbour numbers or the spec check.")
    effort: str = Field(description="low | medium | high")


class Feedback(BaseModel):
    verdict: str = Field(description="One sentence. Include the percentile "
                                     "and the pool it is measured against.")
    strengths: List[str] = Field(default_factory=list,
                                 description="2-3 items. Each must reference a "
                                             "measured feature or a neighbour "
                                             "comparison, not a vibe.")
    weaknesses: List[str] = Field(default_factory=list,
                                  description="2-3 items, same rule.")
    fixes: List[Fix] = Field(default_factory=list,
                             description="2-4 concrete changes, "
                                         "highest expected impact first.")
    grounded_on: List[str] = Field(
        default_factory=list,
        description="Which sources were actually used: retrieved_neighbours, "
                    "brand_percentile, spec_check, own_features, image. "
                    "Omit any source you did not rely on.")
    confidence: str = Field(description="low | medium | high, plus a short "
                                        "reason tied to evidence quality "
                                        "(pool size, neighbour similarity).")


SYSTEM = """You are a performance-creative analyst. You review one ad creative
and give feedback that a designer can act on today.

You are given retrieved evidence: similar high-performing creatives with their
measured feature values, this creative's own measured features, its percentile
inside a reference pool of comparable ads, and a platform specification check.

Rules, in priority order:
1. Ground every claim in the supplied evidence. Quote the actual numbers -
   "11 words vs a neighbour median of 4" beats "too much text".
2. If the evidence does not support a point, do not make it. Saying
   "not supported by the retrieved evidence" is a correct answer.
3. Never claim measured business outcomes. The percentile expresses similarity
   to historically strong creatives, not observed clicks, sales or ROAS.
   On unlabeled platforms say "more winner-like than X%", never "beats X%".
4. Retrieved neighbours may come from a different platform than the upload.
   Respect any note attached to the retrieved set and do not present a
   cross-platform neighbour as a same-platform result.
5. Spec failures outrank aesthetic notes. A creative that cannot run is not
   improved by better colour.
6. Be concrete and brief. No filler, no restating the brief back.
"""

HUMAN = """CREATIVE UNDER REVIEW
brand: {brand} | platform: {platform}

MODEL SCORE
raw calibrated score: {score:.4f}
{percentile_line}
{pool_line}
{reliability_line}

PLATFORM SPEC CHECK
{spec_block}

THIS CREATIVE'S MEASURED FEATURES
{own_features}
{ocr_line}

RETRIEVED EVIDENCE
{rag_block}

{image_note}
Return the structured feedback."""


# ---------------------------------------------------------------- context
def _fmt_feats(d: dict) -> str:
    if not d:
        return "(none extracted)"
    return " | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in d.items())


def build_context(score, platform, brand=None, own_feats=None, ocr_text="",
                  image_path=None, k=5, spec_report=None, text_area_pct=None,
                  clip_vec=None, verbose=False) -> dict:
    """Assemble all four grounding sources. Every piece degrades gracefully -
    a missing index or missing profile weakens the prompt, it does not break
    the call."""
    own_feats = own_feats or {}

    # text_area_frac is a FRACTION here; platform_specs.py wants PERCENT.
    # Convert at this boundary only, never inside the extractor.
    if text_area_pct is None and "text_area_frac" in own_feats:
        try:
            text_area_pct = float(own_feats["text_area_frac"]) * 100.0
        except Exception:
            text_area_pct = None

    ctx = {
        "brand": brand or "unknown", "platform": platform, "score": score,
        "percentile_line": "percentile: unavailable",
        "pool_line": "", "reliability_line": "",
        "spec_block": "(not run)", "rag_block": "(no index available)",
        "own_features": _fmt_feats(own_feats),
        "ocr_line": f"on-image text (OCR): \"{ocr_text[:300]}\"" if ocr_text
                    else "on-image text (OCR): (none detected)",
        "image_note": "", "_pct": None, "_hits": [],
    }

    # 2. RANKED
    try:
        from brand_profiles import BrandProfiles
        bp = BrandProfiles.load()
        r = bp.percentile(score, platform, brand)
        ctx["_pct"] = r.pct
        ctx["percentile_line"] = (
            f"percentile: p{r.pct:.1f} ({r.label}) - {r.sentence()} "
            f"[pool n={r.n}, basis={r.basis}, "
            f"{'labeled' if r.labeled else 'UNLABELED'}]")
        ctx["pool_line"] = bp.context_block(platform, brand)
        if r.caveat:
            ctx["reliability_line"] = f"caveat: {r.caveat}"
    except Exception as e:
        if verbose:
            print(f"  [warn] brand profiles: {e}")

    # 3. CHECKED
    try:
        if spec_report is None and image_path:
            from platform_specs import check_image
            spec_report = check_image(image_path, platform,
                                      text_area_pct=text_area_pct)
        if spec_report is not None:
            ctx["spec_block"] = spec_report.prompt_block()
    except Exception as e:
        if verbose:
            print(f"  [warn] spec check: {e}")

    # 1. RETRIEVED
    try:
        from retrieval import Retriever
        if clip_vec is not None:
            rt = Retriever.load()
            hits = rt.search(clip_vec, k=k, winners_only=True,
                             brand=brand, platform=platform)
            ctx["_hits"] = hits
            block = rt.prompt_block(hits, own_feats)
            plats = set()
            for h in hits:
                p = getattr(h, "platform", None)
                if p is None and isinstance(h, dict):
                    p = h.get("platform")
                if p:
                    plats.add(str(p))
            if plats and plats != {platform}:
                block = f"{block}\n\n{CROSS_DOMAIN_NOTE}"
                if verbose:
                    print(f"  [note] neighbours from {sorted(plats)}, "
                          f"query platform {platform} -> cross-domain "
                          f"caveat added to prompt")
            ctx["rag_block"] = block
    except Exception as e:
        if verbose:
            print(f"  [warn] retrieval: {e}")
    return ctx


# ---------------------------------------------------------------- chain
def _image_part(image_path, max_px=768):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _as_text(raw) -> str:
    """LangChain content is str for most providers, list-of-blocks for some."""
    if isinstance(raw, list):
        return " ".join(b.get("text", "") for b in raw if isinstance(b, dict))
    return raw if isinstance(raw, str) else str(raw)


def _strip_reasoning(txt: str) -> str:
    """Reasoning models emit their scratchpad before the answer. Some wrap it
    in <think> tags, some just narrate. Drop the tagged blocks and any
    markdown fences; the brace scan below handles the untagged case."""
    txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.S | re.I)
    txt = re.sub(r"<thinking>.*?</thinking>", " ", txt, flags=re.S | re.I)
    txt = txt.replace("```json", " ").replace("```", " ")
    return txt


def _json_candidates(txt: str):
    """Yield every balanced {...} span, longest first.

    Cannot just take the first '{' - a reasoning trace quotes fragments of
    the schema while planning, so the first brace is usually a decoy. Cannot
    take a greedy regex either; that swallows prose between two objects.
    Brace-match with string awareness, then prefer the biggest object."""
    spans = []
    for start in (i for i, c in enumerate(txt) if c == "{"):
        depth, in_str, esc = 0, False, False
        for i in range(start, len(txt)):
            c = txt[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    spans.append(txt[start:i + 1])
                    break
    spans.sort(key=len, reverse=True)
    return spans


def parse_feedback_loose(raw) -> Optional[Feedback]:
    """Last-resort parser for models that will not emit clean JSON.

    Qwen3.6 (the text-only ablation arm) is a reasoning model: Groq rejects
    the function-call path outright, and the plain completion comes back as
    pages of deliberation with the object buried inside. Without this the
    no-pixels arm produces nothing, and the vision-vs-text comparison that
    the report depends on has only one side."""
    txt = _strip_reasoning(_as_text(raw))
    for cand in _json_candidates(txt):
        try:
            d = json.loads(cand)
        except Exception:
            continue
        if not isinstance(d, dict) or "verdict" not in d:
            continue
        fixes = []
        for f in d.get("fixes") or []:
            if isinstance(f, dict):
                fixes.append(Fix(change=str(f.get("change", "")),
                                 why=str(f.get("why", "")),
                                 effort=str(f.get("effort", "unknown"))))
            elif isinstance(f, str):
                fixes.append(Fix(change=f, why="(not stated by model)",
                                 effort="unknown"))
        try:
            return Feedback(
                verdict=str(d.get("verdict", "")),
                strengths=[str(x) for x in (d.get("strengths") or [])],
                weaknesses=[str(x) for x in (d.get("weaknesses") or [])],
                fixes=fixes,
                grounded_on=[str(x) for x in (d.get("grounded_on") or [])],
                confidence=str(d.get("confidence", "unknown")))
        except Exception:
            continue
    return None


def _quiet_reasoning(llm, max_tokens=3000):
    """Reasoning models spend their whole budget deliberating and get cut off
    before emitting the answer. Qwen3.6 returned an unclosed <think> block:
    not a parsing problem, a budget problem - the JSON never existed.

    Providers disagree on how to turn thinking down, so try the strongest
    lever first and fall back. .bind() is a Runnable method, so this works
    without knowing how llm_registry built the client."""
    attempts = [
        dict(max_tokens=max_tokens, reasoning_effort="none"),
        dict(max_tokens=max_tokens, reasoning_format="hidden"),
        dict(max_tokens=max_tokens),
    ]
    for kw in attempts:
        try:
            return llm.bind(**kw), kw
        except Exception:
            continue
    return llm, {}


def _looks_truncated(txt: str) -> bool:
    """An opened-but-unclosed thinking block means the response was cut off
    mid-deliberation. Distinguishes 'ran out of room' from 'wrote bad JSON'."""
    low = txt.lower()
    return ("<think>" in low and "</think>" not in low) or \
           (low.count("{") > low.count("}"))


def run_feedback(model_label, ctx, image_path=None, temperature=0.3,
                 max_tokens=3000):
    """LCEL: messages -> llm -> Feedback. Vision models get the pixels;
    text-only models get identical grounding minus the image, which is the
    ablation arm."""
    from langchain_core.messages import SystemMessage, HumanMessage
    from llm_registry import get_llm, BY_LABEL

    spec = BY_LABEL[model_label]
    c = dict(ctx)
    if spec.vision and image_path:
        c["image_note"] = "The creative image is attached. Use it."
    else:
        c["image_note"] = ("NO IMAGE IS ATTACHED - this model is text-only. "
                           "Reason from the measured features and retrieved "
                           "evidence, and do not describe what you cannot see.")

    text = HUMAN.format(**{k: v for k, v in c.items() if not k.startswith("_")})
    content = [{"type": "text", "text": text}]
    if spec.vision and image_path:
        content.append(_image_part(image_path))

    llm = get_llm(model_label, temperature=temperature)
    msgs = [SystemMessage(content=SYSTEM), HumanMessage(content=content)]

    # Three rungs, cheapest first: native structured output, then a schema
    # prompt parsed strictly, then a brace scan for reasoning models that
    # bury the object in a monologue.
    e_struct = None
    if not getattr(spec, "reasoning", False):
        try:
            return llm.with_structured_output(Feedback).invoke(msgs), None
        except Exception as e:
            e_struct = e

    from langchain_core.output_parsers import PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=Feedback)
    msgs[1].content[0]["text"] += (
        "\n\nReturn ONLY the JSON object, no prose before or after it, no "
        "markdown fences, no reasoning:\n" + parser.get_format_instructions())

    raw = None
    try:
        quiet, kw = _quiet_reasoning(llm, max_tokens)
        raw = quiet.invoke(msgs).content
        return parser.parse(_strip_reasoning(_as_text(raw))), None
    except Exception as e_parse:
        fb = parse_feedback_loose(raw) if raw is not None else None
        if fb is not None:
            return fb, None
        txt = _as_text(raw) if raw is not None else ""
        if txt and _looks_truncated(txt):
            return None, (f"TRUNCATED: model spent its token budget reasoning "
                          f"and never emitted the object ({len(txt)} chars, "
                          f"unclosed block). Raise max_tokens or use a "
                          f"non-reasoning model for this arm.")
        head = txt[:200] if txt else "(no response)"
        return None, (f"structured={type(e_struct).__name__ if e_struct else 'skipped'} | "
                      f"parse={type(e_parse).__name__}: {e_parse} | "
                      f"raw_head={head!r}")


def run_feedback_fallback(labels, ctx, image_path=None, temperature=0.3,
                          verbose=True, max_tokens=3000):
    """Policy lives here, not in llm_registry. The registry builds clients;
    the chain decides what to do when one is throttled.

    Google and Groq bill our own quota. OpenRouter free tier is a shared
    capacity pool - Nemotron returned 502 'Worker local total request limit
    reached (16/16)' three minutes after probing clean - so it is the last
    arm, never the first."""
    errs = []
    for lb in labels:
        fb, err = run_feedback(lb, ctx, image_path, temperature, max_tokens)
        if fb is not None:
            return fb, lb, errs
        errs.append((lb, err))
        if verbose:
            print(f"  [fallback] {lb} failed -> {str(err)[:120]}")
    return None, None, errs


def compare_models(labels, ctx, image_path=None, max_tokens=3000):
    """Same creative, same evidence, several models. Agreement between a
    vision model and the text-only arm tells you whether a finding came from
    the pixels or from the numbers."""
    out = {}
    for lb in labels:
        fb, err = run_feedback(lb, ctx, image_path,
                               max_tokens=max_tokens)
        out[lb] = {"feedback": fb, "error": err}
    return out


def render(fb: Feedback) -> str:
    L = [f"VERDICT: {fb.verdict}", ""]
    if fb.strengths:
        L.append("STRENGTHS")
        L += [f"  + {s}" for s in fb.strengths]
    if fb.weaknesses:
        L.append("\nWEAKNESSES")
        L += [f"  - {w}" for w in fb.weaknesses]
    if fb.fixes:
        L.append("\nFIXES")
        for i, f in enumerate(fb.fixes, 1):
            L.append(f"  {i}. [{f.effort}] {f.change}")
            L.append(f"     why: {f.why}")
    L.append(f"\ngrounded on: {', '.join(fb.grounded_on) or '(none declared)'}")
    L.append(f"confidence : {fb.confidence}")
    return "\n".join(L)


# ---------------------------------------------------------------- cli
def _embed_one(path):
    """CLIP vector only. Used when --auto is off, so the score stays whatever
    the caller passed."""
    try:
        from embed_paid import embed_paths
        V, ok = embed_paths([path], show=False)
        return V[0] if len(V) else None
    except Exception as e:
        print(f"  [warn] could not embed image: {e}")
        return None


def auto_extract(path, title="", verbose=True):
    """One scorer.py pass -> (score, cv_feats, clip_vec).

    This is the only sanctioned way to obtain a score inside this module.
    rag_chain.py does not score; it consumes a score. app.py is the other
    caller and must go through scorer.py too."""
    import scorer
    if verbose:
        print("  loading scorer (model + tfidf + CLIP)...")
    bundle = scorer.load_all(with_clip=True, verbose=verbose)
    score, meta, cv, clip_vec = scorer.score_image(path, title, bundle)
    if verbose:
        print(f"  scored -> {score:.4f}")
    return score, cv, clip_vec


def _ocr(path, verbose=True):
    """EasyOCR is optional and slow to import. Only loaded when asked."""
    try:
        import easyocr
        rd = easyocr.Reader(["en"], gpu=False, verbose=False)
        return " ".join(rd.readtext(str(path), detail=0))
    except Exception as e:
        if verbose:
            print(f"  [warn] OCR unavailable: {e}")
        return ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", metavar="IMAGE")
    ap.add_argument("--dry-run", metavar="IMAGE",
                    help="build and print the prompt, call no model")
    ap.add_argument("--auto", action="store_true",
                    help="score the image with scorer.py instead of using "
                         "--score, and feed its CV features into the prompt")
    ap.add_argument("--title", default="",
                    help="title text; leave empty for paid creatives")
    ap.add_argument("--ocr", action="store_true",
                    help="run EasyOCR and include on-image text")
    ap.add_argument("--platform", default="meta")
    ap.add_argument("--brand", default=None)
    ap.add_argument("--score", type=float, default=0.50,
                    help="ignored when --auto is set")
    ap.add_argument("--model", default=None)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--fallback", action="store_true",
                    help="try each available model in order until one answers")
    ap.add_argument("--max-tokens", type=int, default=3000,
                    help="output ceiling for the prompt-parse path; raise it "
                         "if a reasoning model reports TRUNCATED")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()

    img = a.demo or a.dry_run
    if not img:
        ap.print_help()
        sys.exit(0)
    if not os.path.exists(img):
        sys.exit(f"no such image: {img}")

    print("building context...")
    if a.auto:
        score, own_feats, vec = auto_extract(img, a.title, verbose=True)
    else:
        print("  [warn] --auto not set: --score is a placeholder and the "
              "measured-features block will be empty")
        score, own_feats, vec = a.score, {}, _embed_one(img)

    ocr_text = _ocr(img) if a.ocr else ""

    ctx = build_context(score, a.platform, a.brand, own_feats=own_feats,
                        ocr_text=ocr_text, image_path=img, clip_vec=vec,
                        k=a.k, verbose=True)

    if a.dry_run:
        print("\n" + "=" * 72)
        print(HUMAN.format(**{k: v for k, v in ctx.items()
                              if not k.startswith("_")},))
        print("=" * 72)
        sys.exit(0)

    from llm_registry import available
    av = [s.label for s in available()]
    if not av:
        sys.exit("no model keys set - run: python src/llm_registry.py --list")

    if a.fallback and not a.compare:
        order = [a.model] + [x for x in av if x != a.model] if a.model else av
        fb, used, errs = run_feedback_fallback(order, ctx, img,
                                              max_tokens=a.max_tokens)
        if fb is None:
            print("all models failed:")
            for lb, e in errs:
                print(f"  {lb}: {str(e)[:200]}")
            sys.exit(1)
        print("\n" + "=" * 72)
        print(f"MODEL: {used}" + (f"  (after {len(errs)} failure(s))"
                                  if errs else ""))
        print("=" * 72)
        print(render(fb))
        sys.exit(0)

    labels = av if a.compare else [a.model or av[0]]
    for lb, res in compare_models(labels, ctx, img,
                                 max_tokens=a.max_tokens).items():
        print("\n" + "=" * 72)
        print(f"MODEL: {lb}")
        print("=" * 72)
        if res["error"]:
            print(f"  ERROR: {res['error']}")
        else:
            print(render(res["feedback"]))
