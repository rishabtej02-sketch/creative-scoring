#!/usr/bin/env python3
"""
platform_specs.py - "will it RUN?" gate. Pure rules, no model.

Two separate questions, deliberately kept apart:
  1. will it run?      -> this file. deterministic spec check.
  2. will it perform?  -> the scorer + brand percentile.

A creative can pass every spec and still score badly, and vice versa.
Never merge these two numbers into one.

Rules live in data/platform_specs.json - edit there, not here.

CLI:
    python src/platform_specs.py path/to/creative.jpg --platform meta
    python src/platform_specs.py path/to/creative.jpg --all

API:
    from platform_specs import check_image, load_specs, platform_list
    report = check_image("ad.jpg", "meta", text_area_pct=31.0)
    report.verdict      # "pass" | "warn" | "fail"
    report.best_fit     # "Feed portrait 4:5"
    report.checks       # [Check(...), ...]
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, "data", "platform_specs.json")

ICON = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}
RANK = {"pass": 0, "warn": 1, "fail": 2}


@dataclass
class Check:
    name: str
    status: str          # pass | warn | fail
    detail: str
    fix: str = ""


@dataclass
class Report:
    platform: str
    display: str
    width: int
    height: int
    checks: list = field(default_factory=list)
    best_fit: str = ""
    best_fit_delta: float = 0.0

    @property
    def verdict(self) -> str:
        if not self.checks:
            return "pass"
        return max(self.checks, key=lambda c: RANK[c.status]).status

    @property
    def n_fail(self):
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def n_warn(self):
        return sum(1 for c in self.checks if c.status == "warn")

    def summary(self) -> str:
        v = self.verdict
        if v == "pass":
            return f"Runs on {self.display} as {self.best_fit}."
        if v == "warn":
            return f"Runs on {self.display} ({self.best_fit}) with {self.n_warn} warning(s)."
        return f"Will not run as-is on {self.display}: {self.n_fail} blocking issue(s)."

    def prompt_block(self) -> str:
        """Compact form for LLM grounding."""
        lines = [f"platform={self.platform} verdict={self.verdict} "
                 f"size={self.width}x{self.height} best_fit={self.best_fit}"]
        for c in self.checks:
            if c.status != "pass":
                lines.append(f"  [{c.status}] {c.name}: {c.detail}")
        return "\n".join(lines)


_CACHE = None


def load_specs(path=SPEC_PATH) -> dict:
    global _CACHE
    if _CACHE is None:
        with open(path, encoding="utf-8") as f:
            _CACHE = json.load(f)
    return _CACHE


def platform_list():
    p = load_specs()["platforms"]
    return [(k, v["display"]) for k, v in p.items()]


def _best_placement(w, h, placements):
    """Closest placement by log-ratio distance - scale-free, so 300x250 and
    1200x1000 both match a 1.2 ratio equally well."""
    import math
    r = w / h if h else 0
    best, bd = None, 1e9
    for p in placements:
        d = abs(math.log(r / p["ratio"])) if r > 0 else 1e9
        if d < bd:
            best, bd = p, d
    return best, bd


def check_dims(w, h, spec) -> list:
    out = []
    pl, delta = _best_placement(w, h, spec["placements"])
    tol = spec.get("ratio_tolerance", 0.05)

    ratio_ok = True
    if delta <= tol:
        out.append(Check("Aspect ratio", "pass",
                         f"{w}x{h} matches {pl['name']}"))
    elif delta <= tol * 4:
        out.append(Check("Aspect ratio", "warn",
                         f"{w}x{h} is close to {pl['name']} but off by "
                         f"{delta*100:.1f}%; platform will crop or letterbox",
                         f"Resize to {pl['ideal_w']}x{pl['ideal_h']}"))
    else:
        ratio_ok = False
        out.append(Check("Aspect ratio", "fail",
                         f"{w}x{h} matches no supported placement "
                         f"(nearest: {pl['name']})",
                         f"Resize to {pl['ideal_w']}x{pl['ideal_h']}"))

    if not ratio_ok:
        # resolution is moot until the ratio is fixed; judge the source pixels
        enough = min(w, h) >= min(pl["min_w"], pl["min_h"])
        out.append(Check(
            "Resolution", "pass" if enough else "warn",
            (f"{w}x{h} has enough source pixels to crop into "
             f"{pl['ideal_w']}x{pl['ideal_h']}" if enough else
             f"{w}x{h} is too small to crop into {pl['name']} without upscaling"),
            "" if enough else f"Re-export at {pl['ideal_w']}x{pl['ideal_h']}"))
        return out, pl, delta

    if w < pl["min_w"] or h < pl["min_h"]:
        out.append(Check("Resolution", "fail",
                         f"{w}x{h} below minimum {pl['min_w']}x{pl['min_h']} "
                         f"for {pl['name']}",
                         f"Re-export at {pl['ideal_w']}x{pl['ideal_h']}"))
    elif w < pl["ideal_w"] * 0.8 or h < pl["ideal_h"] * 0.8:
        out.append(Check("Resolution", "warn",
                         f"{w}x{h} is under the recommended "
                         f"{pl['ideal_w']}x{pl['ideal_h']}; will look soft on retina",
                         f"Re-export at {pl['ideal_w']}x{pl['ideal_h']}"))
    else:
        out.append(Check("Resolution", "pass", f"{w}x{h} meets {pl['name']}"))
    return out, pl, delta


def check_image(path_or_img, platform: str, text_area_pct=None,
                filesize_bytes=None) -> Report:
    from PIL import Image
    specs = load_specs()["platforms"]
    platform = platform.lower().strip()
    if platform not in specs:
        raise ValueError(f"unknown platform '{platform}'. have: {list(specs)}")
    spec = specs[platform]

    if isinstance(path_or_img, (str, os.PathLike)):
        img = Image.open(path_or_img)
        if filesize_bytes is None:
            filesize_bytes = os.path.getsize(path_or_img)
        fmt = (img.format or "").upper()
    else:
        img = path_or_img
        fmt = (getattr(img, "format", "") or "").upper()

    w, h = img.size
    rep = Report(platform=platform, display=spec["display"], width=w, height=h)

    dim_checks, pl, delta = check_dims(w, h, spec)
    rep.checks += dim_checks
    rep.best_fit, rep.best_fit_delta = pl["name"], delta

    # file size
    if filesize_bytes is not None:
        cap = spec["max_bytes"]
        if platform == "google" and pl["ideal_w"] <= 970 and "Responsive" not in pl["name"]:
            cap = spec.get("max_bytes_fixed_banner", cap)
        mb = filesize_bytes / 1e6
        if filesize_bytes > cap:
            rep.checks.append(Check("File size", "fail",
                                    f"{mb:.2f} MB exceeds {cap/1e6:.2f} MB cap "
                                    f"for {pl['name']}",
                                    "Re-compress as JPG quality 80-85"))
        elif filesize_bytes > cap * 0.85:
            rep.checks.append(Check("File size", "warn",
                                    f"{mb:.2f} MB is near the {cap/1e6:.2f} MB cap"))
        else:
            rep.checks.append(Check("File size", "pass",
                                    f"{mb:.2f} MB under {cap/1e6:.2f} MB"))

    # format
    if fmt:
        ok = [f.upper() for f in spec["formats"]]
        norm = "JPG" if fmt in ("JPEG", "JPG") else fmt
        if norm in ok:
            rep.checks.append(Check("File format", "pass", f"{norm} accepted"))
        else:
            rep.checks.append(Check("File format", "fail",
                                    f"{norm} not accepted (allowed: {', '.join(ok)})",
                                    "Convert to JPG or PNG"))

    # text area - reuses the MSER estimate already in the pipeline
    if text_area_pct is not None:
        ta = spec["text_area"]
        pct = float(text_area_pct)
        if pct >= ta["fail_pct"]:
            rep.checks.append(Check("Text coverage", "fail",
                                    f"~{pct:.0f}% of frame is text "
                                    f"(hard limit {ta['fail_pct']}%)",
                                    "Cut headline to 4-6 words, move detail to caption"))
        elif pct >= ta["warn_pct"]:
            rep.checks.append(Check("Text coverage", "warn",
                                    f"~{pct:.0f}% of frame is text "
                                    f"(recommended under {ta['warn_pct']}%). {ta['note']}",
                                    "Trim to a single headline"))
        else:
            rep.checks.append(Check("Text coverage", "pass",
                                    f"~{pct:.0f}% text, under the "
                                    f"{ta['warn_pct']}% guideline"))
    return rep


def check_all(path, text_area_pct=None) -> dict:
    return {p: check_image(path, p, text_area_pct)
            for p, _ in platform_list()}


def _print(rep: Report):
    print(f"\n=== {rep.display} === {rep.verdict.upper()}")
    print(f"  {rep.summary()}")
    for c in rep.checks:
        print(f"  [{ICON[c.status]:>4}] {c.name:<16} {c.detail}")
        if c.fix and c.status != "pass":
            print(f"         -> {c.fix}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--platform", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--text-area", type=float, default=None,
                    help="text area pct from the MSER estimator")
    a = ap.parse_args()

    if a.all or not a.platform:
        for r in check_all(a.image, a.text_area).values():
            _print(r)
    else:
        _print(check_image(a.image, a.platform, a.text_area))
