"""Self-contained interactive HTML report.

A single-file, dependency-free report (inline CSS + hand-drawn SVG charts -- no
matplotlib, no external assets, no JavaScript). Renders from the profile data dict
produced by :mod:`orchestrate`. Every section degrades gracefully: anything without data
is simply omitted.
"""
from __future__ import annotations

import html
import math

# Families in display order, with the column prefixes that belong to each.
_FAMILIES = [
    ("Temporal", ("temp_",)),
    ("Prosodic", ("pros_",)),
    ("Articulatory", ("art_",)),
    ("Phonatory", ("phon_",)),
    ("Spectral", ("spec_",)),
    ("Complexity", ("comp_",)),
    ("Voice quality", ("qual_",)),
    ("Rhythm", ("rhythm_",)),
    ("Aerodynamic / dynamic", ("aero_", "dyn_", "trans_")),
]

# Surfaced first, in the summary strip.
_HIGHLIGHT_BIOMARKERS = [
    "temp_speech_pause_ratio", "temp_mean_pause_dur", "temp_pause_rate",
    "temp_syllable_rate", "pros_f0_variability", "art_fcr",
    "phon_jitter_local", "phon_shimmer_local_db", "phon_hnr", "phon_cpp",
    "spec_centroid",
]

_EMOTION_COLORS = ["#3a6ea5", "#d64545", "#1f9d68", "#e8a13a", "#8e5bb5",
                   "#2aa9a0", "#c2557a", "#7a8794"]

_CSS = """
:root{
  --bg:#f4f6f9; --card:#ffffff; --ink:#1f2933; --muted:#69707d; --line:#e3e8ef;
  --green:#1f9d68; --amber:#e8a13a; --red:#d64545; --blue:#3a6ea5;
  --shadow:0 1px 3px rgba(16,24,40,.06),0 1px 2px rgba(16,24,40,.10);
}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:var(--ink);background:var(--bg);margin:0;padding:32px 16px;line-height:1.5;}
.wrap{max-width:960px;margin:0 auto;}
h1{font-size:1.5rem;margin:0 0 4px}
h2{font-size:1.05rem;margin:0 0 14px;letter-spacing:.01em}
h3{font-size:.9rem;margin:16px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.sub{color:var(--muted);font-size:.9rem;margin:0 0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin:0 0 20px;box-shadow:var(--shadow)}
.disclaimer{background:#fff8e6;border:1px solid #f2d38a;border-left:4px solid var(--amber);
  border-radius:10px;padding:12px 16px;font-size:.88rem;color:#7a5b12;margin:0 0 22px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.stat{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.stat .v{font-size:1.15rem;font-weight:600;margin-top:2px}
.verdict{display:inline-flex;align-items:center;gap:8px;font-weight:700;font-size:1.05rem;
  padding:8px 14px;border-radius:999px}
.verdict.green{background:#e6f6ee;color:var(--green)}
.verdict.amber{background:#fdf2e0;color:#b9791f}
.verdict.red{background:#fbe9e9;color:var(--red)}
.row{display:flex;align-items:center;gap:10px;margin:7px 0}
.row .lab{flex:0 0 200px;font-size:.83rem}
.row .lab small{display:block;color:var(--muted);font-size:.71rem}
.row .val{flex:0 0 96px;text-align:right;font-variant-numeric:tabular-nums;font-size:.8rem}
.row .pct{flex:0 0 44px;text-align:right;font-variant-numeric:tabular-nums;font-size:.8rem;font-weight:600}
.bar{flex:1;position:relative;height:15px;background:#eef1f5;border-radius:8px;overflow:hidden;min-width:90px}
.flagchip{display:inline-block;background:#fbe9e9;color:var(--red);border-radius:6px;
  padding:1px 7px;font-size:.72rem;font-weight:600;margin:2px}
.okchip{display:inline-block;background:#e6f6ee;color:var(--green);border-radius:6px;
  padding:1px 7px;font-size:.72rem;font-weight:600;margin:2px}
.muted{color:var(--muted);font-size:.85rem}
.legend{font-size:.75rem;color:var(--muted);margin-top:10px}
.legend span{display:inline-block;margin-right:14px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;vertical-align:middle;margin-right:4px}
details{margin-top:10px;border-top:1px solid var(--line);padding-top:8px}
summary{cursor:pointer;color:var(--blue);font-size:.85rem;font-weight:600}
footer{color:var(--muted);font-size:.78rem;text-align:center;margin-top:8px}
table.prov,table.emo{width:100%;border-collapse:collapse;font-size:.82rem}
table.prov td{padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top}
table.prov td:first-child{color:var(--muted);width:34%}
table.emo th,table.emo td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;
  font-variant-numeric:tabular-nums}
table.emo th:first-child,table.emo td:first-child{text-align:left}
table.emo th{color:var(--muted);font-weight:600;font-size:.74rem;text-transform:uppercase}
.z{font-size:.72rem;color:var(--muted)}
.cite{font-size:.71rem;color:var(--muted);font-style:italic}
ul.lim{margin:6px 0 0 18px;padding:0;font-size:.82rem;color:var(--muted)}
ul.lim li{margin:3px 0}
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _pretty(feature: str) -> str:
    return feature.replace("_", " ")


def _fmt(v) -> str:
    """Human-readable number: plain digits for normal magnitudes, no sci-notation."""
    if v is None:
        return "–"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
        return "–"
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"        # 1800.0 -> "1,800"
    if a >= 10:
        return f"{v:.1f}"         # 14.53  -> "14.5"
    if a >= 0.01:
        return f"{v:.3g}"         # 3.9, 0.021
    return f"{v:.1e}"             # 0.0004 -> "4.0e-04"


# --------------------------------------------------------------------------- SVG
def _svg_gauge(prob: float) -> str:
    # The value and "P(AD)" label sit BELOW the pivot, where the needle can never reach
    # (it only sweeps the upper half). Previously the label overlapped the pivot dot.
    w, h, r, cx, cy = 260, 196, 100, 130, 122
    def pt(frac):
        ang = math.pi * (1 - frac)
        return cx + r * math.cos(ang), cy - r * math.sin(ang)
    def arc(f0, f1, color):
        x0, y0 = pt(f0); x1, y1 = pt(f1)
        return (f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}" '
                f'fill="none" stroke="{color}" stroke-width="18"/>')
    nx, ny = pt(max(0.0, min(1.0, prob)))
    zones = (arc(0.0, 0.33, "var(--green)") + arc(0.33, 0.66, "var(--amber)")
             + arc(0.66, 1.0, "var(--red)"))
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:290px" role="img" '
        f'aria-label="Fused probability {prob:.2f}">{zones}'
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="var(--ink)" stroke-width="3"/>'
        f'<circle cx="{cx}" cy="{cy}" r="6" fill="var(--ink)"/>'
        f'<text x="{cx}" y="{cy+44}" text-anchor="middle" font-size="30" font-weight="700" '
        f'fill="var(--ink)">{prob:.2f}</text>'
        f'<text x="{cx}" y="{cy+66}" text-anchor="middle" font-size="13" font-weight="600" '
        f'letter-spacing="0.04em" fill="var(--muted)">P(AD)</text>'
        f'</svg>'
    )


def _bar_marker(pct: float, color="var(--blue)") -> str:
    pct = max(0.0, min(100.0, pct))
    return (
        f'<div class="bar">'
        f'<div style="position:absolute;left:25%;width:50%;top:0;bottom:0;background:#d7e3f2"></div>'
        f'<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#9db4d0"></div>'
        f'<div style="position:absolute;left:calc({pct}% - 5px);top:1px;width:10px;height:13px;'
        f'border-radius:3px;background:{color};box-shadow:0 0 0 2px #fff"></div>'
        f'</div>'
    )


def _stream_bars(per_stream: dict) -> str:
    if not per_stream:
        return ""
    rows = []
    for name, share in sorted(per_stream.items(), key=lambda kv: -kv[1]):
        pct = max(0.0, min(1.0, share)) * 100
        rows.append(
            f'<div class="row"><div class="lab">{_esc(name)}</div>'
            f'<div class="bar"><div style="position:absolute;left:0;top:0;bottom:0;'
            f'width:{pct:.1f}%;background:var(--blue);border-radius:8px"></div></div>'
            f'<div class="pct">{share*100:.0f}%</div></div>'
        )
    return "".join(rows)


def _svg_trajectory(traj: dict) -> str:
    """Multi-line emotion probability trajectory over time."""
    times = traj.get("times") or []
    series = traj.get("series") or {}
    if len(times) < 2 or not series:
        return '<p class="muted">Trajectory too short to plot.</p>'

    W, H = 900, 260
    pad_l, pad_r, pad_t, pad_b = 46, 12, 12, 30
    iw, ih = W - pad_l - pad_r, H - pad_t - pad_b
    t0, t1 = min(times), max(times)
    tspan = (t1 - t0) or 1.0
    ymax = max(1e-6, max(max(v) for v in series.values()))
    ymax = min(1.0, math.ceil(ymax * 10) / 10)

    def X(t): return pad_l + (t - t0) / tspan * iw
    def Y(v): return pad_t + ih - (v / ymax) * ih

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Emotion trajectory">']
    # gridlines + y labels
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + ih - frac * ih
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" '
                     f'stroke="#eef1f5" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
                     f'fill="var(--muted)">{frac*ymax:.2f}</text>')
    # x labels
    for frac in (0, 0.5, 1.0):
        t = t0 + frac * tspan
        parts.append(f'<text x="{X(t):.1f}" y="{H-8}" text-anchor="middle" font-size="10" '
                     f'fill="var(--muted)">{t:.0f}s</text>')
    # one polyline per emotion
    legend = []
    for i, (label, values) in enumerate(series.items()):
        color = _EMOTION_COLORS[i % len(_EMOTION_COLORS)]
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}"
                       for t, v in zip(times, values[:len(times)]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="1.6" stroke-linejoin="round" opacity="0.9"/>')
        legend.append(f'<span><span class="dot" style="background:{color}"></span>'
                      f'{_esc(label)}</span>')
    parts.append('</svg>')
    parts.append(f'<div class="legend">{"".join(legend)}</div>')
    return "".join(parts)


def _svg_umap(umap: dict) -> str:
    cn = umap.get("cn") or []
    ad = umap.get("ad") or []
    me = umap.get("coordinates")
    pts_all = [p for p in (list(cn) + list(ad)) if p and len(p) >= 2]
    if not pts_all:
        return ""
    W, H, pad = 900, 400, 20
    xs = [p[0] for p in pts_all]; ys = [p[1] for p in pts_all]
    if me:
        xs.append(me[0]); ys.append(me[1])
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    xspan = (x1 - x0) or 1.0; yspan = (y1 - y0) or 1.0

    def X(x): return pad + (x - x0) / xspan * (W - 2 * pad)
    def Y(y): return H - pad - (y - y0) / yspan * (H - 2 * pad)

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="UMAP position">']
    for p in cn:
        if p and len(p) >= 2:
            parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="2.6" '
                         f'fill="var(--green)" opacity="0.28"/>')
    for p in ad:
        if p and len(p) >= 2:
            parts.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="2.6" '
                         f'fill="var(--red)" opacity="0.28"/>')
    if me:
        cx, cy = X(me[0]), Y(me[1])
        parts.append(f'<line x1="{cx-9:.1f}" y1="{cy-9:.1f}" x2="{cx+9:.1f}" y2="{cy+9:.1f}" '
                     f'stroke="#fff" stroke-width="5"/>'
                     f'<line x1="{cx-9:.1f}" y1="{cy+9:.1f}" x2="{cx+9:.1f}" y2="{cy-9:.1f}" '
                     f'stroke="#fff" stroke-width="5"/>'
                     f'<line x1="{cx-9:.1f}" y1="{cy-9:.1f}" x2="{cx+9:.1f}" y2="{cy+9:.1f}" '
                     f'stroke="var(--ink)" stroke-width="2.6"/>'
                     f'<line x1="{cx-9:.1f}" y1="{cy+9:.1f}" x2="{cx+9:.1f}" y2="{cy-9:.1f}" '
                     f'stroke="var(--ink)" stroke-width="2.6"/>')
    parts.append('</svg>')
    parts.append('<div class="legend">'
                 '<span><span class="dot" style="background:var(--green)"></span>controls</span>'
                 '<span><span class="dot" style="background:var(--red)"></span>AD</span>'
                 '<span><strong>✕</strong> this speaker</span></div>')
    return "".join(parts)


# ----------------------------------------------------------------- section html
def _verdict(prob):
    # Bands are aligned with the model's 0.5 decision threshold, so the headline can never
    # contradict the prediction (previously p=0.659 read "Borderline" but predicted AD).
    if prob is None:
        return "amber", "Biomarker profile (no fused prediction)"
    if prob >= 0.66:
        return "red", "Patterns consistent with AD"
    if prob >= 0.5:
        return "amber", "Borderline — leaning AD"
    if prob >= 0.33:
        return "amber", "Borderline — leaning healthy control"
    return "green", "Consistent with healthy controls"


_DESC_CACHE: dict = {}


def _describe(feature):
    """(description, unit, direction_in_ad, references) from voxmarkers provenance."""
    if feature in _DESC_CACHE:
        return _DESC_CACHE[feature]
    info = ("", "", "", [])
    try:
        import voxmarkers
        d = voxmarkers.describe(feature)
        info = (d.get("description", ""), d.get("unit", ""),
                d.get("direction_in_ad", ""), d.get("references", []) or [])
    except Exception:
        pass
    _DESC_CACHE[feature] = info
    return info


def _deviation_note(pct, direction):
    """Whether this speaker's deviation runs in the AD-consistent direction."""
    if pct is None or direction not in ("increases", "decreases"):
        return ""
    high, low = pct >= 75, pct <= 25
    if not (high or low):
        return "typical range"
    consistent = (high and direction == "increases") or (low and direction == "decreases")
    side = "high" if high else "low"
    return f"{side} — AD-consistent" if consistent else f"{side} — opposite to AD"


def _biomarker_row(feat, values, percentiles, flags, band):
    band = band or {}
    val = values.get(feat)
    pct = percentiles.get(feat)
    desc, unit, direction, refs = _describe(feat)
    color = "var(--red)" if feat in flags else "var(--blue)"
    note = _deviation_note(pct, direction)
    unit_s = f" {unit}" if unit and unit not in ("a.u.", "ratio") else ""
    med = band.get("median")
    sub = note or (direction and f"AD: {direction}") or ""
    val_txt = f"{_fmt(val) if val is not None else '–'}{_esc(unit_s) if val is not None else ''}"
    if med is not None:
        val_txt += f' <span class="muted">/ {_fmt(med)}</span>'
    pct_txt = f"{pct:.0f}%" if pct is not None else "–"
    bar = _bar_marker(pct, color=color) if pct is not None else '<div class="bar"></div>'
    return (f'<div class="row"><div class="lab">{_esc(_pretty(feat))}'
            f'{f"<small>{_esc(sub)}</small>" if sub else ""}</div>'
            f'{bar}<div class="val">{val_txt}</div>'
            f'<div class="pct">{pct_txt}</div></div>')


def build_html(data: dict, include_spectrogram: bool = False) -> str:
    meta = data.get("metadata", {})
    fusion = data.get("fusion")
    biom = data.get("biomarkers", {})
    emo = data.get("emotion")
    umap = data.get("umap")
    prov = data.get("provenance", {})

    prob = fusion["probability"] if fusion else None
    vclass, vtext = _verdict(prob)

    parts = [f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
             f'<meta name="viewport" content="width=device-width, initial-scale=1.0">'
             f'<title>Acoustic Phenotyping Report</title><style>{_CSS}</style></head>'
             f'<body><div class="wrap">']
    parts.append('<h1>Acoustic Phenotyping Report</h1>')
    parts.append(f'<p class="sub">{_esc(meta.get("audio_file",""))} · task: '
                 f'{_esc(meta.get("task",""))}</p>')
    parts.append('<div class="disclaimer"><strong>Research use only.</strong> This is not a '
                 'medical device and must not be used for diagnosis. Results describe '
                 'acoustic patterns relative to a reference cohort.</div>')

    # verdict + quality
    parts.append('<div class="card">')
    parts.append(f'<span class="verdict {vclass}">{_esc(vtext)}</span>')
    parts.append('<div class="grid" style="margin-top:16px">')
    for k, label in (("duration_seconds", "Duration (s)"), ("snr_db", "SNR (dB)"),
                     ("quality_flag", "Quality")):
        if k in meta:
            parts.append(f'<div class="stat"><div class="k">{label}</div>'
                         f'<div class="v">{_esc(meta[k])}</div></div>')
    parts.append(f'<div class="stat"><div class="k">Streams</div>'
                 f'<div class="v">{len(data.get("streams_present",[]))}</div></div>')
    parts.append('</div></div>')

    # fusion
    if fusion:
        parts.append('<div class="card"><h2>Fusion result</h2>')
        parts.append('<div style="display:flex;flex-wrap:wrap;gap:24px;align-items:center">')
        if prob is not None:
            parts.append(f'<div style="flex:0 0 auto">{_svg_gauge(prob)}</div>')
        parts.append('<div style="flex:1;min-width:240px">'
                     '<div class="muted" style="margin-bottom:6px">Per-stream contribution</div>'
                     f'{_stream_bars(fusion.get("per_stream", {}))}</div>')
        parts.append('</div>')
        if fusion.get("attribution_method"):
            parts.append(f'<p class="legend">Attribution: {_esc(fusion["attribution_method"])} · '
                         f'Prediction: <strong>{_esc(fusion.get("prediction"))}</strong></p>')
        parts.append('</div>')

    # ---------------------------------------------------------------- biomarkers
    parts.append('<div class="card"><h2>Acoustic biomarkers</h2>')
    values = biom.get("values", {})
    percentiles = biom.get("percentiles", {})
    bands = biom.get("bands", {}) or {}
    flags = set(biom.get("flags", []))

    if biom.get("space_warning"):
        parts.append('<div class="disclaimer" style="margin:0 0 14px">⚠️ Percentiles and flags '
                     'may be unreliable: the patient values and the reference cohort appear to '
                     'be in different feature spaces. Align the spaces before interpreting.</div>')

    if values:
        # Only features with a reference cohort get a comparison bar. Drawing an empty bar
        # for the rest would imply a comparison that doesn't exist.
        compared = [f for f in values if percentiles.get(f) is not None]
        uncompared = [f for f in values if percentiles.get(f) is None]
        highlights = [f for f in _HIGHLIGHT_BIOMARKERS if f in compared]

        parts.append(f'<p class="muted">{len(values)} features extracted: {len(compared)} '
                     f'compared with the reference cohort, {len(uncompared)} measured without '
                     f'a reference.</p>')

        if compared:
            parts.append('<h3>Compared with the reference cohort</h3>'
                         '<p class="legend" style="margin-top:0">Shown as <em>value / cohort '
                         'median</em>, with the speaker\'s percentile.</p>')
            for feat in highlights:
                parts.append(_biomarker_row(feat, values, percentiles, flags, bands.get(feat, {})))

            # the remaining compared features, grouped by family, collapsed
            shown = set(highlights)
            for fam_name, prefixes in _FAMILIES:
                fam_feats = [f for f in compared if f.startswith(prefixes) and f not in shown]
                if not fam_feats:
                    continue
                parts.append(f'<details><summary>{_esc(fam_name)} — {len(fam_feats)} more '
                             f'features</summary>')
                for feat in sorted(fam_feats):
                    parts.append(_biomarker_row(feat, values, percentiles, flags,
                                                bands.get(feat, {})))
                parts.append('</details>')

            parts.append('<div class="legend"><span><span class="dot" style="background:#d7e3f2">'
                         '</span>cohort interquartile range</span>'
                         '<span><span class="dot" style="background:var(--blue)"></span>'
                         'this speaker</span></div>')
            if flags:
                parts.append('<p style="margin-top:12px"><strong>Outside typical range:</strong> '
                             + "".join(f'<span class="flagchip">{_esc(_pretty(f))}</span>'
                                       for f in sorted(flags)) + '</p>')
            else:
                parts.append('<p style="margin-top:12px"><span class="okchip">no compared '
                             'feature is outside the typical range</span></p>')

        if uncompared:
            parts.append('<h3>Measured without a reference</h3>'
                         '<p class="legend" style="margin-top:0">No reference-cohort statistics '
                         'exist for these features (the Pitt reference covers '
                         f'{len(compared) or "none"} of {len(values)}; its phonation features were '
                         'never validly extracted), so they are reported as raw values only and '
                         'cannot be judged typical or atypical.</p>')
            fam_of = {}
            for fam_name, prefixes in _FAMILIES:
                for f in uncompared:
                    if f.startswith(prefixes):
                        fam_of.setdefault(f, fam_name)
            order = {name: i for i, (name, _) in enumerate(_FAMILIES)}
            rows = sorted(uncompared, key=lambda f: (order.get(fam_of.get(f), 99), f))
            parts.append('<table class="emo"><tr><th>Feature</th><th>Family</th><th>Value</th>'
                         '<th>Expected in AD</th></tr>')
            for feat in rows:
                _, unit, direction, _ = _describe(feat)
                v = values.get(feat)
                unit_s = f" {unit}" if unit and unit not in ("a.u.", "ratio") else ""
                parts.append(f'<tr><td>{_esc(_pretty(feat))}</td>'
                             f'<td>{_esc(fam_of.get(feat, "—"))}</td>'
                             f'<td>{_fmt(v)}{_esc(unit_s)}</td>'
                             f'<td>{_esc(direction or "—")}</td></tr>')
            parts.append('</table>')

        # citations for the highlighted features
        cites = []
        for feat in highlights:
            _, _, _, refs = _describe(feat)
            if refs:
                cites.append(f'<li><strong>{_esc(_pretty(feat))}</strong> — '
                             f'{_esc("; ".join(refs))}</li>')
        if cites:
            parts.append('<details><summary>References for the key features</summary>'
                         f'<ul class="lim">{"".join(cites)}</ul>'
                         '<p class="cite">Seed references — reconcile against the scoping '
                         'review before publication.</p></details>')
    else:
        parts.append('<p class="muted">No biomarker features available.</p>')
    parts.append('</div>')

    # ------------------------------------------------------------------- emotion
    if emo and (emo.get("table") or emo.get("aggregates")):
        parts.append('<div class="card"><h2>Emotion dynamics</h2>')
        traj = emo.get("trajectory")
        if traj:
            parts.append('<h3>Trajectory over time</h3>')
            parts.append(_svg_trajectory(traj))
        else:
            parts.append('<p class="muted">No per-chunk trajectory available '
                         '(requires the emotion stream).</p>')

        table = emo.get("table") or {}
        if table:
            stats = emo.get("stats", ["mean", "std", "range", "cv"])
            parts.append('<h3>Per-emotion statistics</h3>')
            parts.append('<table class="emo"><tr><th>Emotion</th>'
                         + "".join(f'<th>{_esc(s)}</th>' for s in stats) + '</tr>')
            for emo_name, row in table.items():
                tds = []
                for s in stats:
                    v = row.get(s)
                    z = row.get(f"{s}_z")
                    cell = _fmt(v) if v is not None else "–"
                    if z is not None:
                        cell += f' <span class="z">z={z:+.1f}</span>'
                    tds.append(f'<td>{cell}</td>')
                parts.append(f'<tr><td>{_esc(emo_name)}</td>{"".join(tds)}</tr>')
            parts.append('</table>')
            parts.append('<p class="legend">std / range / cv capture <em>variability</em> of '
                         'affect over the recording; reduced variability (flattened affect) is '
                         'the dementia-relevant pattern. z = SDs from the control mean.</p>')
        parts.append('</div>')

    # ---------------------------------------------------------------------- umap
    if umap and umap.get("coordinates"):
        parts.append('<div class="card"><h2>Position among the reference cohort</h2>')
        parts.append(_svg_umap(umap))
        parts.append('<p class="legend">Placement of this speaker\'s deep vision embedding in '
                     'the Pitt UMAP layout, from its 15 nearest reference recordings (UMAP\'s own '
                     'placement rule, without the final refinement step). Proximity to a cloud '
                     'is suggestive, not diagnostic — the clouds overlap substantially.</p></div>')

    # optional spectrogram
    if include_spectrogram:
        img = _spectrogram_img(data)
        if img:
            parts.append(f'<div class="card"><h2>Log-mel spectrogram</h2>{img}</div>')

    # ---------------------------------------------------------------- provenance
    parts.append('<div class="card"><h2>Provenance &amp; reproducibility</h2><table class="prov">')

    def prow(k, v):
        parts.append(f'<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>')

    if prov.get("generated_utc"):
        prow("Generated", prov["generated_utc"])
    prow("Task", prov.get("task", meta.get("task", "")))
    if prov.get("fusion_model"):
        fm = prov["fusion_model"]
        prow("Fusion model", f'{fm.get("task")} / {fm.get("variant")}')
    if prov.get("stream_models"):
        prow("Stream models", ", ".join(f"{k}: {v}" for k, v in prov["stream_models"].items()))
    if prov.get("schemas"):
        prow("Feature schemas", ", ".join(f"{k}={v}" for k, v in prov["schemas"].items()))
    if prov.get("packages"):
        prow("Package versions", ", ".join(f"{k}={v}" for k, v in prov["packages"].items()))
    if prov.get("reference"):
        r = prov["reference"]
        prow("Reference cohort",
             f'{r.get("source") or "not available"} (space={r.get("space")}, '
             f'{r.get("n_features")} features'
             + (f', n={r["n_controls"]}' if r.get("n_controls") else '') + ')')
    else:
        prow("Reference cohort", data.get("reference_source") or "not available")
    prow("Streams present", ", ".join(data.get("streams_present", [])))
    prow("Training data", prov.get("training_dataset", "Pitt Corpus (DementiaBank)"))
    if prov.get("python"):
        prow("Environment", f'Python {prov["python"]} · {prov.get("platform","")}')
    parts.append('</table>')

    lims = prov.get("limitations") or []
    if lims:
        parts.append('<h3>Limitations</h3><ul class="lim">'
                     + "".join(f'<li>{_esc(x)}</li>' for x in lims) + '</ul>')
    if data.get("notes"):
        parts.append('<details><summary>Pipeline notes</summary><ul class="lim">'
                     + "".join(f'<li>{_esc(n)}</li>' for n in data["notes"]) + '</ul></details>')
    parts.append('</div>')

    parts.append('<footer>Generated by acophenotype · research-use only, not a diagnostic '
                 'device.</footer></div></body></html>')
    return "".join(parts)


def _spectrogram_img(data: dict):
    """Optional base64 log-mel spectrogram (requires matplotlib + in-memory audio)."""
    audio = data.get("_audio")
    if audio is None:
        return None
    try:
        import base64
        import io
        import numpy as np
        import librosa
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y, sr = audio
        mel = librosa.feature.melspectrogram(y=np.asarray(y), sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        fig, ax = plt.subplots(figsize=(9, 3.2))
        librosa.display.specshow(log_mel, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return (f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" '
                f'style="max-width:100%;border-radius:8px">')
    except Exception:
        return None
