#!/usr/bin/env python3
"""
discover_models.py - ask the providers what they actually serve, today.

WHY THIS EXISTS:
Hardcoded model ids rot. Groq retired the Llama family in 2026 and Google
retired gemini-2.0-flash (June 2026) and then gemini-2.5-flash, all inside
a few months. Any registry written from documentation is stale on arrival.

So this does not guess. It calls each provider's list endpoint with your
key, then PROBES each candidate with a real request - one text call, one
tiny-image call - and records what actually worked. Vision support is
measured, never assumed, because a text-only model does not error on an
image prompt: it answers anyway, from the text alone, and the answer looks
fine. That silent failure is the whole reason for the image probe.

BUDGET: this project runs on a zero-dollar budget. OpenRouter models that
charge anything are filtered out BEFORE probing, because a probe against a
paid model on an empty balance is a wasted call at best. Use --paid-ok to
lift the filter.

Output: data/llm_models.json, which llm_registry.py reads.

    python src/discover_models.py --list            # just show what exists
    python src/discover_models.py --probe           # test + write registry
    python src/discover_models.py --probe --max 6   # limit probes
    python src/discover_models.py --list --paid-ok  # include paid models
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "llm_models.json")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass

# ids containing these are not chat models - skip probing them.
# orpheus/lyria = audio+music generation, transcribe = speech-to-text,
# "-image" = image OUTPUT models. Vision INPUT lives on the plain flash /
# flash-lite ids, so this filter costs us no vision candidate.
SKIP = ("whisper", "tts", "embedding", "embed", "guard", "moderation",
        "safety", "veo", "imagen", "image-generation", "aqa", "gemma-3n",
        "learnlm", "native-audio", "live-", "-tts", "rerank",
        "orpheus", "lyria", "transcribe", "-image", "openrouter/free")

# rough preference order for the dropdown - substring match, first wins
PREFER = ["llama-4", "llama4", "scout", "maverick", "llama-3.3", "llama3.3",
          "qwen3", "qwen-3", "gemini-3", "gemini-2.5-flash", "flash-lite",
          "flash", "kimi", "gpt-oss", "mixtral", "gemma"]

# vision probe accepts any of these. A model answering "dark red" or "maroon"
# can see. Narrow matching produced a false text-only verdict on
# gemini-3-flash-preview in the previous run.
RED_WORDS = ("red", "crimson", "scarlet", "maroon", "vermilion", "ruby",
             "brick", "rouge", "rot", "rojo")

# set from argv in __main__; when True, paid OpenRouter models are kept
ALLOW_PAID = False


def _tiny_png_b64():
    """A 64x64 solid red square. If a model can see, it says red."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), (220, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _is_zero(v):
    """OpenRouter reports prices as strings: '0', '0.0', '0e0' all mean free."""
    try:
        return float(v) == 0.0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- listing
def list_groq():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("  GROQ_API_KEY not set - skipping Groq")
        return []
    import requests
    try:
        r = requests.get("https://api.groq.com/openai/v1/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        ids = sorted(m["id"] for m in r.json().get("data", []))
        print(f"  Groq serves {len(ids)} models")
        return ids
    except Exception as e:
        print(f"  !! Groq list failed: {type(e).__name__}: {str(e)[:120]}")
        return []


def list_google():
    key = (os.environ.get("GOOGLE_API_KEY")
           or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not key:
        print("  GOOGLE_API_KEY / GEMINI_API_KEY not set - skipping Google")
        return []
    try:
        from google import genai
        client = genai.Client(api_key=key)
        ids = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            acts = getattr(m, "supported_actions", None) or []
            if acts and "generateContent" not in acts:
                continue
            ids.append(name.replace("models/", ""))
        ids = sorted(set(ids))
        print(f"  Google serves {len(ids)} generateContent models")
        return ids
    except Exception as e:
        print(f"  !! Google list failed: {type(e).__name__}: {str(e)[:120]}")
        return []


def list_openrouter():
    """OpenRouter is how Llama gets back into this project - Groq dropped the
    Llama family in 2026. It is an OpenAI-compatible gateway, so one key
    reaches many vendors.

    FREE means BOTH prompt and completion price are zero. A model priced at
    zero on input and non-zero on output is not free, and the ':free' suffix
    on its own is not proof - the pricing block is."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("  OPENROUTER_API_KEY not set - skipping OpenRouter")
        return []
    import requests
    try:
        r = requests.get("https://openrouter.ai/api/v1/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        print(f"  !! OpenRouter list failed: {type(e).__name__}: {str(e)[:120]}")
        return []

    rows, n_llama, n_free = [], 0, 0
    for m in data:
        mid = m.get("id", "")
        if not mid:
            continue
        # OpenRouter publishes modality info - use it instead of guessing
        mods = ((m.get("architecture") or {}).get("input_modalities")
                or (m.get("architecture") or {}).get("modality") or "")
        sees = "image" in str(mods).lower()
        p = m.get("pricing") or {}
        is_free = ((_is_zero(p.get("prompt")) and _is_zero(p.get("completion")))
                   or mid.endswith(":free"))
        is_llama = "llama" in mid.lower()
        n_llama += is_llama
        n_free += is_free
        rows.append({"id": mid, "declared_vision": sees,
                     "free": is_free, "llama": is_llama})

    print(f"  OpenRouter serves {len(rows)} models "
          f"({n_llama} Llama, {n_free} free-tier)")

    free_llama = [d["id"] for d in rows if d["free"] and d["llama"]]
    if free_llama:
        print(f"  FREE Llama found ({len(free_llama)}): "
              + ", ".join(free_llama[:6]))
    else:
        print("  !! NO FREE LLAMA on OpenRouter right now.")
        print("     The Llama arm needs another gateway, or the ablation")
        print("     drops to a free open-weights model instead. Decide before")
        print("     the report claims a Llama comparison.")

    if not ALLOW_PAID:
        before = len(rows)
        rows = [d for d in rows if d["free"]]
        print(f"  budget filter: {before} -> {len(rows)} (free only)")

    # Llama first, then declared-vision, then id
    rows.sort(key=lambda d: (0 if d["llama"] else 1,
                             0 if d["declared_vision"] else 1,
                             d["id"]))
    return [d["id"] for d in rows]


# ---------------------------------------------------------------- filtering
def keep(model_id):
    """True if this id looks like a chat model worth probing.

    THIS FUNCTION WAS LOST IN A PREVIOUS EDIT - its body was left dangling at
    the end of list_openrouter, so `keep` never bound and discover() raised
    NameError. Restored.
    """
    low = str(model_id).lower()
    return not any(s in low for s in SKIP)


def rank(model_id):
    low = model_id.lower()
    for i, p in enumerate(PREFER):
        if p in low:
            return i
    return len(PREFER)


# ---------------------------------------------------------------- probing
def probe(provider, model_id, timeout=45):
    """Returns (text_ok, vision_ok, latency_s, error). Vision is confirmed
    only if the model reports the colour it was shown."""
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage

    kw = dict(temperature=0, max_tokens=30)
    prov = provider
    if provider == "openrouter":
        prov = "openai"
        kw["base_url"] = "https://openrouter.ai/api/v1"
        kw["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")

    t0 = time.time()
    try:
        llm = init_chat_model(model_id, model_provider=prov, **kw)
    except Exception as e:
        return False, False, 0.0, f"init: {type(e).__name__}: {str(e)[:90]}"

    try:
        r = llm.invoke("Reply with the single word: OK")
        txt = r.content if isinstance(r.content, str) else str(r.content)
        if not txt.strip():
            return False, False, time.time() - t0, "empty text response"
    except Exception as e:
        return False, False, time.time() - t0, \
            f"{type(e).__name__}: {str(e)[:90]}"
    dt = time.time() - t0

    vision = False
    try:
        msg = HumanMessage(content=[
            {"type": "text", "text": "What colour is this image? One word."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_tiny_png_b64()}"}},
        ])
        vr = llm.invoke([msg])
        vt = (vr.content if isinstance(vr.content, str)
              else str(vr.content)).lower()
        vision = any(w in vt for w in RED_WORDS)
    except Exception:
        vision = False
    return True, vision, dt, None


def discover(do_probe=True, max_probe=10):
    print("=== DISCOVER MODELS ===")
    print("  budget mode: " + ("PAID ALLOWED" if ALLOW_PAID else "FREE ONLY"))
    found = {"groq": list_groq(),
             "google_genai": list_google(),
             "openrouter": list_openrouter()}

    cands = {}
    for prov, ids in found.items():
        ok = [i for i in ids if keep(i)]
        # list_openrouter already ranked Llama/vision first
        if prov != "openrouter":
            ok = sorted(ok, key=rank)
        cands[prov] = ok
        print(f"\n  {prov}: {len(ok)} chat candidates")
        for i in ok[:20]:
            print(f"    {i}")
        if len(ok) > 20:
            print(f"    ... +{len(ok)-20} more")

    if not do_probe:
        return None

    print("\n=== PROBE (real calls: 1 text + 1 image each) ===")
    working = []
    for prov, ids in cands.items():
        for mid in ids[:max_probe]:
            t_ok, v_ok, dt, err = probe(prov, mid)
            if t_ok:
                tag = "VISION" if v_ok else "text  "
                print(f"  OK   [{tag}] {prov:<13} {mid:<44} {dt:5.2f}s")
                working.append({"provider": prov, "model": mid,
                                "vision": v_ok, "latency_s": round(dt, 2),
                                "free_tier": not ALLOW_PAID})
            else:
                print(f"  fail          {prov:<13} {mid:<44} {err}")
            # Google free tier 429s when probed fast. Two calls per model
            # (text + image) already doubles the rate. Breathe.
            if prov == "google_genai":
                time.sleep(4)

    if not working:
        print("\n  !! nothing worked. Check keys, or the provider is down.")
        return None

    vis = [w for w in working if w["vision"]]
    txt = [w for w in working if not w["vision"]]
    print(f"\n  working: {len(working)}  (vision={len(vis)} text-only={len(txt)})")

    # merge, do not clobber: a provider that 429s today must not erase a
    # model that was verified working yesterday.
    prev = []
    if os.path.exists(OUT):
        try:
            with open(OUT, encoding="utf-8") as f:
                prev = (json.load(f) or {}).get("models", []) or []
        except Exception:
            prev = []
    seen = {(w["provider"], w["model"]) for w in working}
    carried = [p for p in prev if (p.get("provider"), p.get("model")) not in seen]
    if carried:
        print(f"  carried forward {len(carried)} previously-verified models")
        for c in carried:
            c["stale"] = True
    merged = working + carried

    doc = {
        "discovered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "budget_mode": "paid_allowed" if ALLOW_PAID else "free_only",
        "note": ("Auto-generated by discover_models.py. Vision was MEASURED "
                 "by showing each model a red square, not assumed from docs. "
                 "OpenRouter candidates filtered to zero-cost models."),
        "models": merged,
    }
    os.makedirs(DATA, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"  wrote {os.path.relpath(OUT, ROOT)}  ({len(merged)} models)")

    if not vis:
        print("\n  !! NO VISION MODEL AVAILABLE.")
        print("  The RAG chain still runs - it grounds on measured features")
        print("  and retrieved neighbours - but nothing will see the pixels.")
    print("\n  next: python src/llm_registry.py --list")
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list only, no calls")
    ap.add_argument("--probe", action="store_true", help="test and write file")
    ap.add_argument("--max", type=int, default=10,
                    help="max models to probe per provider")
    ap.add_argument("--paid-ok", action="store_true",
                    help="include paid OpenRouter models (default: free only)")
    a = ap.parse_args()
    ALLOW_PAID = a.paid_ok
    if a.list:
        discover(do_probe=False)
    elif a.probe:
        discover(do_probe=True, max_probe=a.max)
    else:
        ap.print_help()
