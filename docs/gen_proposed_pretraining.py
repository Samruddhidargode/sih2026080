"""Proposed operational system - pre-training workflow (the target design, not the prototype).
Six regimes (5 monsoon + Western Disturbance), operational NCMRWF/IMD/ISRO feeds, meteorologist
sign-off gate, versioned retraining. Run: python gen_proposed_pretraining.py
"""
from xml.sax.saxutils import escape as e

W, H = 1700, 1420
C = {
    "src": ("#DCEBFA", "#2F6DB5"), "proc": ("#EFEFEF", "#666666"), "model": ("#EBDDF5", "#7B3FA0"),
    "loss": ("#FDE3CC", "#D2691E"), "label": ("#FFF6CC", "#B8960C"), "out": ("#DDF3E0", "#2E8B57"),
    "gov": ("#E0F7F5", "#16A085"), "group": ("#FAFAFA", "#999999"),
}

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
    '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>',
    f'<rect width="{W}" height="{H}" fill="white"/>',
    '<text x="30" y="42" font-size="28" font-weight="700" fill="#222">Proposed system — pre-training workflow (operational target design)</text>',
    "<text x=\"30\" y=\"66\" font-size=\"15\" fill=\"#555\">Covers all six PS regimes across two seasons. Runs on NCMRWF's own operational data, not public proxies.</text>",
]


def box(x, y, w, h, kind, title, lines=(), fs=12.5, tfs=15, dashed=False):
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


def arrow(pts, label=None, lx=None, ly=None, dashed=False, anchor="middle", fs=12.5):
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
        xx = x + i * 205
        parts.append(f'<rect x="{xx}" y="{y}" width="20" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text x="{xx+27}" y="{y+12}" font-size="14" fill="#444">{e(n)}</text>')


# ---- Row A: operational data sources
box(30, 90, 400, 105, "src", "NCMRWF operational NWP", ["NCUM-G / NCUM-R live feed", "rain, wind, MSLP, 500 hPa height"], fs=11.5)
box(450, 90, 400, 105, "src", "IMD ground truth", ["AWS network + gridded rainfall", "near-real-time + daily QC'd product"], fs=11.5)
box(870, 90, 390, 105, "src", "ISRO Cartosat DEM", ["official elevation + coastline"], fs=11.5)
box(1280, 90, 390, 105, "src", "Survey of India boundaries", ["official district polygons"], fs=11.5)

# ---- Row B: two seasonal ingestion tracks
box(30, 225, 800, 100, "proc", "Monsoon ingestion (JJAS)",
    ["align to operational grid (12/4 km)", "channels: rain, u/v wind, MSLP, elevation, land-coast mask"], fs=11.5)
box(860, 225, 810, 100, "proc", "Winter ingestion (Oct–May)",
    ["same alignment, extra channel: 500 hPa geopotential height", "+ jet-stream latitude (Western Disturbance signal)"], fs=11.5)

# ---- Row C: label generation (meteorologist-defined)
group(30, 355, 800, 220, "Monsoon regime labels", "from OBSERVED rainfall + pressure, met.-validated thresholds")
box(50, 400, 370, 80, "label", "Active / Break", ["core-zone rainfall anomaly", "vs 30yr+ climatology"], fs=10.5)
box(440, 400, 370, 80, "label", "Depression", ["closed low in MSLP field"], fs=10.5)
box(50, 490, 370, 75, "label", "Orographic", ["rain locked to terrain lift", "(Ghats, NE hills) regardless of regime"], fs=10.5)
box(440, 490, 370, 75, "label", "Coastal", ["rain driven by onshore flow", "at the coastline"], fs=10.5)

group(860, 355, 810, 220, "Winter regime label", "different physics — needs its own definition")
box(880, 410, 770, 130, "label", "Western Disturbance",
    ["extratropical trough at 500 hPa +", "jet-stream position over north India", "distinct from all five monsoon regimes"], fs=11.5)

box(30, 600, 1640, 55, "gov", "All regime definitions and thresholds require sign-off from IMD / NCMRWF meteorologists before use", fs=12.5, tfs=13)

# ---- Row D: Stage A classifiers (season-gated)
box(30, 685, 300, 90, "proc", "Season gate", ["calendar month decides", "monsoon mode vs winter mode"], fs=11)
box(360, 685, 620, 90, "model", "Monsoon classifier (5-way CNN)",
    ["input: wind + MSLP only (never rainfall)", "-> Active / Break / Depression / Orographic / Coastal"], fs=11.5)
box(1010, 685, 660, 90, "model", "Western Disturbance detector",
    ["input: wind + MSLP + 500 hPa height", "-> WD present / not present, with confidence"], fs=11.5)

# ---- Row E: Stage B - six regime experts
group(30, 810, 1640, 260, "Stage B · Six regime-specific correction U-Nets", "each warm-started from one shared base model, fine-tuned on only its own regime's days")
regimes = ["Active", "Break", "Depression", "Orographic", "Coastal", "Western\nDisturbance"]
rw = (1640 - 40 - 5*20) / 6
for k, r in enumerate(regimes):
    rx = 50 + k * (rw + 20)
    box(rx, 855, rw, 190, "model", r.replace("\n", " "), ["operational-resolution", "U-Net expert"], fs=10.5, tfs=13)

# ---- Row F: output head + loss
box(30, 1105, 800, 110, "model", "Output head (identical on all 6 experts)",
    ["ch0: corrected rainfall (mm) — softplus", "ch1-3: P(> heavy / very heavy / extremely heavy) — sigmoid, monotone"], fs=11.5)
box(860, 1105, 810, 110, "loss", "Asymmetric, threshold-weighted loss",
    ["extra penalty ONLY for under-predicting extreme rain", "same design across monsoon and winter experts"], fs=11.5)

# ---- Bottom: versioning + retraining + sign-off
box(30, 1245, 540, 100, "out", "Versioned model bundle",
    ["classifier + 6 U-Nets + norm stats", "tagged with NWP model version + training window"], fs=11)
box(590, 1245, 540, 100, "gov", "Meteorologist sign-off gate",
    ["regime definitions, thresholds and alert", "cutoffs reviewed before the bundle ships"], fs=11)
box(1150, 1245, 520, 100, "proc", "Retraining trigger",
    ["scheduled annually before each monsoon,", "or whenever NCMRWF upgrades its NWP model"], fs=11)

legend(30, 1375, [("src", "data"), ("proc", "processing"), ("label", "labels"), ("model", "neural network"), ("loss", "loss"), ("gov", "governance / sign-off")])

# ---- arrows
arrow([(430, 195), (430, 225)])
arrow([(1065, 195), (1065, 225)])
arrow([(430, 325), (430, 355)])
arrow([(1265, 325), (1265, 355)])
arrow([(430, 575), (430, 600)])
arrow([(1265, 575), (1265, 600)])
arrow([(430, 655), (430, 685), (330, 685)], label="monsoon days")
arrow([(1265, 655), (1265, 685), (1320, 685)], label="winter days")
arrow([(330, 730), (360, 730)])
arrow([(330, 775), (330, 800), (1010, 800), (1010, 775)], dashed=True)
arrow([(680, 775), (680, 855)], label="regime label -> target")
arrow([(1340, 775), (1340, 855)], label="WD label -> target")
arrow([(430, 1070), (430, 1105)], label="y-hat")
arrow([(830, 1200), (860, 1200)])
arrow([(300, 1215), (300, 1245)])
arrow([(860, 1180), (860, 1215), (860, 1245)], dashed=True)
arrow([(300, 1345), (860, 1345), (860, 1345)], dashed=True)
arrow([(1130, 1295), (1150, 1295)])

parts.append("</svg>")
open("proposed_pretraining_pipeline.svg", "w", encoding="utf-8").write("\n".join(parts))
print("written proposed_pretraining_pipeline.svg")
