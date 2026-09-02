"""
compute_top_performer_stats.py
One-off: compute mean/median of CV + metadata features for overperformed=1 rows.
Output → data/top_performer_stats.json (used by LLM feedback prompt for grounding).
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

FEATURES_CSV = Path("data/features.csv")
FEATURE_SPEC = Path("models/feature_spec.json")
OUT_JSON     = Path("data/top_performer_stats.json")

# ---- load ----
assert FEATURES_CSV.exists(), f"missing {FEATURES_CSV}"
df = pd.read_csv(FEATURES_CSV)
print(f"loaded features.csv → {len(df)} rows, {len(df.columns)} cols")

assert "overperformed" in df.columns, "no 'overperformed' col in features.csv"

# ---- get feature col names ----
# prefer feature_spec.json if present; else auto-detect numeric non-label cols
cv_cols, meta_cols = [], []
if FEATURE_SPEC.exists():
    spec = json.loads(FEATURE_SPEC.read_text())
    cv_cols   = spec.get("CV_COLS",       spec.get("cv_cols", []))
    meta_cols = spec.get("METADATA_COLS", spec.get("metadata_cols", []))
    print(f"feature_spec.json → {len(cv_cols)} CV cols, {len(meta_cols)} meta cols")
else:
    print("no feature_spec.json → auto-detecting numeric cols")
    exclude = {"overperformed", "labelable", "video_id", "score", "creative_id", "brand"}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    meta_cols = [c for c in numeric if c not in exclude]

all_cols = [c for c in (cv_cols + meta_cols) if c in df.columns]
missing  = [c for c in (cv_cols + meta_cols) if c not in df.columns]
if missing:
    print(f"WARN: {len(missing)} spec cols missing in features.csv → skipped")
    print(f"  e.g. {missing[:5]}")

print(f"using {len(all_cols)} feature cols total")

# ---- filter top performers ----
top = df[df["overperformed"] == 1]
print(f"overperformed=1 → {len(top)} rows ({100*len(top)/len(df):.1f}%)")
assert len(top) > 50, "too few top performers → check label col"

# ---- compute stats ----
stats = {
    "n_top_performers": int(len(top)),
    "n_total":          int(len(df)),
    "base_rate":        round(len(top) / len(df), 4),
    "features": {}
}

for col in all_cols:
    s = top[col].dropna()
    if len(s) == 0:
        continue
    stats["features"][col] = {
        "mean":   round(float(s.mean()),   4),
        "median": round(float(s.median()), 4),
        "p25":    round(float(s.quantile(0.25)), 4),
        "p75":    round(float(s.quantile(0.75)), 4),
        "std":    round(float(s.std()),    4),
    }

# ---- save ----
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(stats, indent=2))
print(f"\n✓ saved → {OUT_JSON}")
print(f"  {len(stats['features'])} feature stats written")

# ---- preview ----
print("\nsample (first 3 features):")
for k, v in list(stats["features"].items())[:3]:
    print(f"  {k}: mean={v['mean']}, median={v['median']}")
