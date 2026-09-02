"""
plot_score_dists.py
3-way score distribution: YouTube (training) vs Meta (paid) vs Google (paid)
Outputs:
  figs/score_dist_3way.png       - overlaid histograms + per-source stats box
  figs/score_dist_per_brand.png  - per-brand mean score, sorted, by source
  data/brand_spread_table.csv    - per-brand mean/std/n for report
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

# ---- load ----
yt   = pd.read_csv(ROOT / "data/youtube_scores.csv")
paid = pd.read_csv(ROOT / "data/paid_ad_scores.csv")

yt = yt.dropna(subset=["score"])
paid = paid.dropna(subset=["score"])

meta   = paid[paid["source"] == "meta"]["score"].values
google = paid[paid["source"] == "google"]["score"].values
ytv    = yt["score"].values

print(f"YouTube: n={len(ytv):>6} μ={ytv.mean():.3f} σ={ytv.std():.3f} range=[{ytv.min():.3f},{ytv.max():.3f}]")
print(f"Meta   : n={len(meta):>6} μ={meta.mean():.3f} σ={meta.std():.3f} range=[{meta.min():.3f},{meta.max():.3f}]")
print(f"Google : n={len(google):>6} μ={google.mean():.3f} σ={google.std():.3f} range=[{google.min():.3f},{google.max():.3f}]")

# ---- plot 1: overlaid histograms ----
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(0, 1, 51)

ax.hist(ytv,    bins=bins, alpha=0.55, label=f"YouTube (train domain, n={len(ytv):,})",   color="#2E86AB", density=True)
ax.hist(meta,   bins=bins, alpha=0.55, label=f"Meta ads (paid, n={len(meta):,})",         color="#E63946", density=True)
ax.hist(google, bins=bins, alpha=0.55, label=f"Google ads (paid, n={len(google):,})",     color="#F4A261", density=True)

ax.axvline(ytv.mean(),    color="#2E86AB", ls="--", lw=1.2)
ax.axvline(meta.mean(),   color="#E63946", ls="--", lw=1.2)
ax.axvline(google.mean(), color="#F4A261", ls="--", lw=1.2)

stats_txt = (
    f"YouTube  μ={ytv.mean():.3f}  σ={ytv.std():.3f}\n"
    f"Meta     μ={meta.mean():.3f}  σ={meta.std():.3f}\n"
    f"Google   μ={google.mean():.3f}  σ={google.std():.3f}"
)
ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes, fontsize=10,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

ax.set_xlabel("Predicted overperformance score")
ax.set_ylabel("Density")
ax.set_title("Domain transfer: training vs paid-ad application set\n"
             "Paid-ad scores compress to 0.4–0.5 band; YouTube spans full 0–1")
ax.legend(loc="upper right")
ax.set_xlim(0, 1)
plt.tight_layout()
out1 = FIGS / "score_dist_3way.png"
plt.savefig(out1, dpi=150)
plt.close()
print(f"saved → {out1}")

# ---- per-brand spread table ----
brand_stats = (paid.dropna(subset=["score"])
                   .groupby(["source", "brand"])["score"]
                   .agg(["mean", "std", "count"])
                   .reset_index()
                   .sort_values(["source", "mean"], ascending=[True, False]))
brand_out = ROOT / "data/brand_spread_table.csv"
brand_stats.to_csv(brand_out, index=False)
print(f"saved → {brand_out}  ({len(brand_stats)} brand-source rows)")

# ---- plot 2: per-brand mean, sorted ----
fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=True)
for ax, src, color in zip(axes, ["meta", "google"], ["#E63946", "#F4A261"]):
    sub = brand_stats[brand_stats["source"] == src].sort_values("mean")
    ax.barh(sub["brand"], sub["mean"], xerr=sub["std"], color=color, alpha=0.8, ecolor="gray")
    ax.axvline(sub["mean"].mean(), color="black", ls="--", lw=1, label=f"overall μ={sub['mean'].mean():.3f}")
    spread = sub["mean"].max() - sub["mean"].min()
    ax.set_title(f"{src.title()} ads · per-brand mean score\nspread={spread:.3f} ({sub['mean'].max():.3f} − {sub['mean'].min():.3f})")
    ax.set_xlabel("mean predicted score")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
out2 = FIGS / "score_dist_per_brand.png"
plt.savefig(out2, dpi=150)
plt.close()
print(f"saved → {out2}")

print("\ndone.")
