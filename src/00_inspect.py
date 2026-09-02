#!/usr/bin/env python3
"""
00_inspect.py - print every schema I need before writing the RAG layer.
Run from repo root:  python src/00_inspect.py
Paste the whole output back.
"""
import os, sys, glob, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")


def line(c="-", n=70):
    print(c * n)


def show_csv(path):
    import pandas as pd
    try:
        df = pd.read_csv(path, nrows=2000)
    except Exception as e:
        print(f"  !! read failed: {e}")
        return
    try:
        total = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1
    except Exception:
        total = "?"
    print(f"  rows={total}  cols={len(df.columns)}")
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        samp = s.dropna().head(2).tolist()
        samp = [str(x)[:40] for x in samp]
        print(f"    {c:<28} {str(s.dtype):<10} nuniq={nun:<7} eg={samp}")


def show_npy(path):
    import numpy as np
    for pk in (False, True):
        try:
            a = np.load(path, allow_pickle=pk)
            print(f"  shape={getattr(a,'shape',None)} dtype={a.dtype} allow_pickle={pk}")
            try:
                print(f"  head={[str(x)[:30] for x in a[:2]]}")
            except Exception:
                pass
            return
        except Exception as e:
            last = e
    print(f"  !! load failed: {last}")


def main():
    print(f"ROOT = {ROOT}")
    print(f"python = {sys.version.split()[0]}")

    line("=")
    print("CSV FILES in data/")
    line("=")
    for p in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
        print(f"\n[{os.path.basename(p)}]  {os.path.getsize(p)/1e6:.1f} MB")
        show_csv(p)

    line("=")
    print("NPY / NPZ in data/ and models/")
    line("=")
    for d in (DATA, MODELS):
        for p in sorted(glob.glob(os.path.join(d, "*.np[yz]"))):
            print(f"\n[{os.path.relpath(p, ROOT)}]  {os.path.getsize(p)/1e6:.1f} MB")
            show_npy(p)

    line("=")
    print("JSON in data/ and models/")
    line("=")
    for d in (DATA, MODELS):
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            print(f"\n[{os.path.relpath(p, ROOT)}]  {os.path.getsize(p)/1e6:.2f} MB")
            try:
                j = json.load(open(p, encoding="utf-8"))
                if isinstance(j, dict):
                    print(f"  dict keys[:20] = {list(j.keys())[:20]}")
                    for k in list(j.keys())[:2]:
                        print(f"    {k} -> {str(j[k])[:200]}")
                elif isinstance(j, list):
                    print(f"  list len={len(j)}  head={str(j[:2])[:300]}")
            except Exception as e:
                print(f"  !! {e}")

    line("=")
    print("PKL in models/")
    line("=")
    for p in sorted(glob.glob(os.path.join(MODELS, "*.pkl"))):
        print(f"  {os.path.basename(p):<32} {os.path.getsize(p)/1e6:.1f} MB")

    line("=")
    print("IMAGE DIRS (counts)")
    line("=")
    for p in sorted(glob.glob(os.path.join(DATA, "*"))):
        if os.path.isdir(p):
            n = sum(len(f) for _, _, f in os.walk(p))
            print(f"  {os.path.relpath(p, ROOT):<40} files={n}")

    line("=")
    print("PACKAGES")
    line("=")
    for m in ("numpy", "pandas", "torch", "transformers", "lightgbm",
              "sklearn", "streamlit", "faiss", "langchain", "langchain_core",
              "langchain_groq", "langchain_google_genai", "PIL"):
        try:
            mod = __import__(m)
            print(f"  {m:<26} {getattr(mod,'__version__','?')}")
        except Exception:
            print(f"  {m:<26} MISSING")


if __name__ == "__main__":
    main()
