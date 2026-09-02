#!/usr/bin/env python3
"""
llm_registry.py - one dropdown, many models.

WHY A REGISTRY AND NOT A HARDCODED CLIENT:
The rubric wants a supervised-vs-GenAI comparison. Swapping the generative
layer at runtime turns that into a live demo instead of a claim in a table.

VISION IS THE TRAP. Not every model here can see the image. Llama 3.3 70B is
text-only; Llama 4 Scout/Maverick are multimodal. If you send an image to a
text-only model it does not error - it answers anyway, from the feature
numbers alone, and the answer looks plausible. So each entry carries a
`vision` flag and rag_chain.py routes on it.

That is not a limitation to hide. "Same creative, same grounded features,
with pixels vs without pixels" is a clean ablation and it belongs in the
report.

MODEL IDS DRIFT. Groq in particular retires ids. Do not trust this file -
run the checker, which calls every configured model for real:

    python src/llm_registry.py --check

    python src/llm_registry.py --list
    python src/llm_registry.py --ask "one sentence on ad creative testing"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

# .env support is optional; env vars work either way
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass


@dataclass
class Spec:
    label: str            # what the dropdown shows
    provider: str         # init_chat_model provider string
    model: str            # provider-side model id
    key_env: str          # env var holding the api key
    vision: bool          # can it actually see the image?
    free: bool
    notes: str = ""
    tags: list = field(default_factory=list)
    base_url: str = ""    # set for OpenAI-compatible gateways (OpenRouter)


# Providers reachable through the OpenAI-compatible protocol. OpenRouter is
# here because Groq dropped Llama entirely in 2026; this is how Llama gets
# back into the comparison.
GATEWAYS = {
    "openrouter": ("openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


REGISTRY = [
    Spec("Llama 4 Scout (Groq)", "groq",
         "meta-llama/llama-4-scout-17b-16e-instruct", "GROQ_API_KEY",
         vision=True, free=True,
         notes="Open-weight multimodal. Primary Llama option.",
         tags=["llama", "open", "vision"]),

    Spec("Llama 4 Maverick (Groq)", "groq",
         "meta-llama/llama-4-maverick-17b-128e-instruct", "GROQ_API_KEY",
         vision=True, free=True,
         notes="Larger Llama 4. Slower, usually richer.",
         tags=["llama", "open", "vision"]),

    Spec("Llama 3.3 70B (Groq)", "groq",
         "llama-3.3-70b-versatile", "GROQ_API_KEY",
         vision=False, free=True,
         notes="TEXT ONLY. Use as the no-pixels ablation arm.",
         tags=["llama", "open", "text-only", "ablation"]),

    Spec("Gemini 2.0 Flash", "google_genai",
         "gemini-2.0-flash", "GOOGLE_API_KEY",
         vision=True, free=True,
         notes="Already wired in your app. Fast, generous free tier.",
         tags=["closed", "vision"]),

    Spec("Gemini 2.5 Flash", "google_genai",
         "gemini-2.5-flash", "GOOGLE_API_KEY",
         vision=True, free=True,
         notes="Newer Gemini. Falls back to 2.0 if unavailable.",
         tags=["closed", "vision"]),

    Spec("Claude Haiku 4.5", "anthropic",
         "claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY",
         vision=True, free=False,
         notes="Paid. Strongest structured-output adherence.",
         tags=["closed", "vision", "paid"]),
]

BY_LABEL = {s.label: s for s in REGISTRY}

# ---------------------------------------------------------------- discovered
DISCOVERED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "llm_models.json")


def _label_for(model_id, provider, vision):
    """Readable dropdown name from a raw model id."""
    base = model_id.split("/")[-1]
    free = base.endswith(":free")
    base = base.replace(":free", "")
    pretty = base.replace("-", " ").replace("_", " ").title()
    pretty = pretty.replace("Instruct", "").strip()
    host = {"groq": "Groq", "google_genai": "Google",
            "openrouter": "OpenRouter"}.get(provider, provider)
    tail = "" if vision else " [text-only]"
    return f"{pretty}{' free' if free else ''} ({host}){tail}"


def load_discovered(path=DISCOVERED, replace=True):
    """Adopt the output of discover_models.py.

    Hardcoded ids rot fast - both Groq and Google retired ids under this
    project inside one quarter. When a discovery file exists it wins,
    because it was produced by calling the providers, not by reading docs.
    """
    global REGISTRY, BY_LABEL
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return False

    specs = []
    for m in doc.get("models", []):
        prov = m["provider"]
        base = ""
        if prov in GATEWAYS:
            prov_real, base, key_env = GATEWAYS[prov]
        else:
            prov_real = prov
            key_env = ("GROQ_API_KEY" if prov == "groq" else "GOOGLE_API_KEY")
        specs.append(Spec(
            label=_label_for(m["model"], m["provider"], m.get("vision", False)),
            provider=prov_real, model=m["model"], key_env=key_env,
            vision=bool(m.get("vision", False)), free=True,
            notes=(f"verified {doc.get('discovered_at','')[:10]}, "
                   f"{m.get('latency_s','?')}s probe"),
            tags=["discovered"] + (["vision"] if m.get("vision") else
                                   ["text-only", "ablation"]),
            base_url=base))
    if not specs:
        return False
    REGISTRY = specs if replace else specs + REGISTRY
    BY_LABEL = {s.label: s for s in REGISTRY}
    return True


load_discovered()



def have_key(spec: Spec) -> bool:
    return bool(os.environ.get(spec.key_env, "").strip())


def available(vision_only=False):
    """Only models whose API key is actually present."""
    return [s for s in REGISTRY
            if have_key(s) and (s.vision or not vision_only)]


def get_llm(label: str, temperature=0.3, max_tokens=1400):
    spec = BY_LABEL.get(label)
    if spec is None:
        raise KeyError(f"unknown model '{label}'. have: {list(BY_LABEL)}")
    if not have_key(spec):
        raise RuntimeError(
            f"{spec.label} needs {spec.key_env} in your environment or .env")
    from langchain.chat_models import init_chat_model
    kw = dict(temperature=temperature, max_tokens=max_tokens)
    if spec.base_url:
        kw["base_url"] = spec.base_url
        kw["api_key"] = os.environ[spec.key_env]
    return init_chat_model(spec.model, model_provider=spec.provider, **kw)


def list_models(verbose=True):
    if verbose:
        print(f"{'model':<44}{'provider':<15}{'vision':<8}{'key':<9}notes")
        print("-" * 110)
        for s in REGISTRY:
            print(f"{s.label:<44}{s.provider:<15}"
                  f"{'yes' if s.vision else 'NO':<8}"
                  f"{'set' if have_key(s) else 'MISSING':<9}{s.notes}")
        missing = sorted({s.key_env for s in REGISTRY if not have_key(s)})
        if missing:
            print(f"\nmissing keys: {', '.join(missing)}")
            print("put them in .env at the repo root, one per line, "
                  "e.g.  GROQ_API_KEY=gsk_...")
    return REGISTRY


def check(prompt="Reply with exactly: OK"):
    """Actually call each model. Model ids get retired without notice, so a
    registry that has never been exercised is a registry that lies."""
    import time
    print("=== LIVE CHECK ===")
    ok, bad, skip = [], [], []
    for s in REGISTRY:
        if not have_key(s):
            skip.append(s.label)
            print(f"  SKIP  {s.label:<28} ({s.key_env} not set)")
            continue
        t0 = time.time()
        try:
            r = get_llm(s.label, temperature=0, max_tokens=20).invoke(prompt)
            txt = (r.content if isinstance(r.content, str)
                   else str(r.content))[:40].replace("\n", " ")
            print(f"  OK    {s.label:<28} {time.time()-t0:5.1f}s  \"{txt}\"")
            ok.append(s.label)
        except Exception as e:
            msg = str(e).replace("\n", " ")[:110]
            print(f"  FAIL  {s.label:<28} {type(e).__name__}: {msg}")
            bad.append(s.label)
    print(f"\n  working={len(ok)}  failed={len(bad)}  no key={len(skip)}")
    if bad:
        print("  -> failed ids are usually retired. Check the provider's "
              "current model list and edit REGISTRY.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ask")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    if a.check:
        check()
    elif a.ask:
        label = a.model or (available()[0].label if available() else None)
        if not label:
            sys.exit("no model available - set GROQ_API_KEY or GOOGLE_API_KEY")
        print(f"[{label}]")
        print(get_llm(label).invoke(a.ask).content)
    else:
        list_models()
