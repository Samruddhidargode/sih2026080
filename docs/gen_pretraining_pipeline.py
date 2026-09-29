"""Pre-training pipeline diagram (offline, once per season).
Each box: title + 2-3 short phrases explaining what happens INSIDE that step,
so the diagram is readable without narration. Run: python gen_pretraining_pipeline.py
"""
from xml.sax.saxutils import escape as e

W, H = 1600, 980
C = {
    "src": ("#DCEBFA", "#2F6DB5"), "proc": ("#EFEFEF", "#666666"), "model": ("#EBDDF5", "#7B3FA0"),
    "loss": ("#FDE3CC", "#D2691E"), "label": ("#FFF6CC", "#B8960C"), "out": ("#DDF3E0", "#2E8B57"),
    "group": ("#FAFAFA", "#999999"),
}

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
    '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>',
    f'<rect width="{W}" height="{H}" fill="white"/>',
    '<text x="30" y="42" font-size="28" font-weight="700" fill="#222">Pre-training pipeline (offline, once per season)</text>',
    "<text x=\"30\" y=\"66\" font-size=\"15\" fill=\"#555\">Stage A: train the regime classifier on labels. Stage B: train one U-Net per regime on that regime's days. Then freeze both.</text>",
]


def box(x, y, w, h, kind, title, lines=(), fs=12.5, tfs=15.5, dashed=False):
    fill, stroke = C[kind]
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2.2"{dash}/>')
    ty = y + tfs + 8
    parts.append(f'<text x="{x+w/2}" y="{ty}" text-anchor="middle" font-size="{tfs}" font-weight="700" fill="#222">{e(title)}</text>')
    for ln in lines:
        ty += fs + 6
        parts.append(f'<text x="{x+w/2}" y="{ty}" text-anchor="middle" font-size="{fs}" fill="#333">{e(ln)}</text>')


def group(x, y, w, h, title, subtitle=None):
    fill, stroke = C["group"]
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5" stroke-dasharray="7,5"/>')
    parts.append(f'<text x="{x+12}" y="{y+20}" font-size="14.5" font-weight="700" fill="#555">{e(title)}</text>')
    if subtitle:
        parts.append(f'<text x="{x+12}" y="{y+37}" font-size="12" fill="#777">{e(subtitle)}</text>')


def arrow(pts, label=None, lx=None, ly=None, dashed=False, anchor="middle", fs=13):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    dash = ' stroke-dasharray="7,5"' if dashed else ""
    parts.append(f'<path d="{d}" fill="none" stroke="#333" stroke-width="2.2"{dash} marker-end="url(#arr)"/>')
    if label:
        if lx is None:
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 - 7
        parts.append(f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" font-size="{fs}" font-style="italic" fill="#444" paint-order="stroke" stroke="white" stroke-width="5">{e(label)}</text>')


def legend(x, y, items):
    for i, (k, n) in enumerate(items):
        fill, stroke = C[k]
        xx = x + i * 190
        parts.append(f'<rect x="{xx}" y="{y}" width="20" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text x="{xx+27}" y="{y+12}" font-size="14" fill="#444">{e(n)}</text>')


# ---- data sources
box(30, 100, 300, 118, "src", "GFS 0.25° forecast",
    ["rain, u850, v850, MSLP", "00Z run · Day-1/2/3 · JJAS 2021-25", "byte-range GRIB download (herbie)"])
box(30, 245, 300, 98, "src", "IMD 0.25° daily rain",
    ["ground truth · 129 × 135 grid", "24h ending 03 UTC · -999 -> NaN"])
box(30, 368, 300, 98, "src", "DEM (SRTM 90 m)",
    ["GeoTIFF tiles over India", "~90,000 px block-averaged / cell"])

# ---- alignment engine
box(370, 100, 320, 118, "proc", "BOX 1 · NWP alignment",
    ["24-h totals on the IMD day (fxx 3->27)", "bilinear regrid -> 129 × 135", "exact grid match: GFS & IMD share 0.25°"])
box(370, 368, 320, 98, "proc", "BOX 2 · Terrain engine",
    ["block-average 90 m -> 25 km cell", "-> elevation (m) + land-sea mask"])
box(370, 505, 320, 135, "label", "Ground truth + regime labels",
    ["y = IMD rain (mm), NaN cells masked", "Active/Break: core-zone rain anomaly", ">= +-1sigma for >= 3 consecutive days",
     "Depression: closed low in MSLP; else Coastal/Oro"], fs=11.5)

# ---- tensor
box(750, 190, 260, 108, "proc", "Concat + z-score",
    ["torch.cat(dim=0) of 6 channels", "mean/std from TRAIN years only", "log1p on rainfall first (heavy-tailed)"], fs=11.5)
box(750, 368, 260, 80, "proc", "Unified tensor", ["6 × 129 × 135, one per (day, lead)"])

# ---- models
box(1070, 100, 500, 118, "model", "STAGE A · Regime classifier (CNN)",
    ["input: u850, v850, MSLP only - never rainfall,", "so the same network works on tomorrow's forecast",
     "[Conv3x3->BN->ReLU->MaxPool] x4 -> softmax(4)", "class-weighted cross-entropy (rare regimes up-weighted)"], fs=12)

group(1070, 245, 500, 275, "STAGE B · Four regime-specific U-Nets", "each expert sees only its own regime's training days")
box(1090, 297, 225, 95, "model", "Active U-Net",
    ["standard 3x3 conv", "base architecture for", "widespread, uniform rain"], fs=10.5)
box(1335, 297, 225, 95, "model", "Break U-Net",
    ["3x3 conv + L2 weight decay", "discourages false drizzle on", "mostly dry / zero-rain days"], fs=10.5)
box(1090, 407, 225, 100, "model", "Depression U-Net",
    ["7x7 kernels in encoder", "wider receptive field to capture", "large synoptic-scale rain bands"], fs=10.5)
box(1335, 407, 225, 100, "model", "Coastal/Orographic U-Net",
    ["dense skip connections", "preserve fine ridge / coastline", "detail lost in down-sampling"], fs=10.5)

box(1070, 555, 245, 115, "model", "Output head (every expert)",
    ["4 channels, same design on all 4:", "ch0 = rain mm (softplus, >= 0)", "ch1-3 = P(>64.5/115.5/204.4 mm)", "(sigmoid)"], fs=10.5)
box(1330, 555, 240, 115, "loss", "Asymmetric loss",
    ["MSE(rain) + extra penalty only when", "a cell > 64.5 mm is UNDER-predicted",
     "+ BCE on the 3 probability channels", "over-prediction is not penalised extra"], fs=10.5)

# ---- bottom band
box(30, 720, 660, 90, "proc", "Season split (never random days - adjacent monsoon days are correlated)",
    ["train 2021-23 -> validate 2024 (early stopping) -> test 2025, untouched until final scoring"], fs=12)
box(710, 720, 600, 90, "out", "Saved artifacts -> used unchanged by the live system",
    ["classifier.pt + 4 × unet_<regime>.pt + norm_stats.json", "weights are frozen: no further learning happens live"], fs=12)
box(1330, 720, 240, 90, "proc", "Optional warm start",
    ["init all 4 experts from one", "shared U-Net first - helps", "data-scarce regimes (Depression)"], fs=10.5, dashed=True)

# ---- arrows
arrow([(330, 159), (370, 159)])
arrow([(330, 417), (370, 417)])
arrow([(690, 164), (720, 164), (720, 244), (750, 244)], label="4 dynamic channels", lx=705, ly=140, anchor="start")
arrow([(690, 417), (720, 417), (720, 270), (750, 270)], label="2 static channels", lx=705, ly=350, anchor="start")
arrow([(880, 298), (880, 368)])
arrow([(330, 315), (350, 315), (350, 572), (370, 572)], label="observed rain", lx=353, ly=470, anchor="start")
arrow([(1010, 244), (1040, 244), (1040, 159), (1070, 159)], label="u850, v850, MSLP", lx=1000, ly=290, anchor="end")
arrow([(1010, 408), (1040, 408), (1040, 460), (1070, 460)], label="all 6 channels", lx=1045, ly=480, anchor="start")
arrow([(690, 572), (720, 572), (720, 690), (1580, 690), (1580, 159), (1570, 159)], label="regime label -> Stage A target", lx=900, ly=684)
arrow([(530, 640), (530, 700), (1450, 700), (1450, 670)], label="y (mm) -> loss target", lx=1180, ly=714)
arrow([(1192, 520), (1192, 555)], label="y-hat", lx=1205, ly=540, anchor="start")
arrow([(1170, 615), (1330, 615)])
arrow([(1450, 555), (1450, 520)], dashed=True, label="backprop", lx=1460, ly=537, anchor="start")

legend(30, 850, [("src", "data"), ("proc", "processing"), ("label", "labels / truth"),
                 ("model", "neural network"), ("loss", "loss"), ("out", "artifact")])
parts.append("</svg>")
open("pretraining_pipeline.svg", "w", encoding="utf-8").write("\n".join(parts))
print("written pretraining_pipeline.svg")
