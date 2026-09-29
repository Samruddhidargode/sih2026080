"""Generate the two PPT-ready SVG flowcharts for SIH26080 (final architecture).

    python docs/make_flowcharts.py   ->  docs/flow_1_training_pipeline.svg
                                          docs/flow_2_system_workflow.svg
Every box is one call; edit text here and re-run.
"""
from pathlib import Path
from xml.sax.saxutils import escape as _e

OUT = Path(__file__).resolve().parent
W, H = 1600, 900
FONT = "Segoe UI, Arial, sans-serif"

C = {  # fill, stroke
    "src":   ("#DCEBFA", "#2F6DB5"),
    "proc":  ("#EFEFEF", "#666666"),
    "model": ("#EBDDF5", "#7B3FA0"),
    "loss":  ("#FDE3CC", "#D2691E"),
    "label": ("#FFF6CC", "#B8960C"),
    "out":   ("#DDF3E0", "#2E8B57"),
    "group": ("#FAFAFA", "#999999"),
}


class SVG:
    def __init__(self, title, subtitle=""):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
            '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>',
            f'<rect width="{W}" height="{H}" fill="white"/>',
            f'<text x="30" y="44" font-size="28" font-weight="700" fill="#222">{_e(title)}</text>',
            f'<text x="30" y="68" font-size="15" fill="#555">{_e(subtitle)}</text>',
        ]

    def box(self, x, y, w, h, kind, title, lines=(), fs=14, tfs=None, dashed=False):
        fill, stroke = C[kind]
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
        tfs = tfs or fs + 2
        ty = y + tfs + 8
        self.parts.append(f'<text x="{x + w/2}" y="{ty}" text-anchor="middle" font-size="{tfs}" font-weight="700" fill="#222">{_e(title)}</text>')
        for ln in lines:
            ty += fs + 5
            self.parts.append(f'<text x="{x + w/2}" y="{ty}" text-anchor="middle" font-size="{fs}" fill="#333">{_e(ln)}</text>')

    def group(self, x, y, w, h, title):
        fill, stroke = C["group"]
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5" stroke-dasharray="7,5"/>')
        self.parts.append(f'<text x="{x + 12}" y="{y + 20}" font-size="15" font-weight="700" fill="#555">{_e(title)}</text>')

    def path(self, pts, label=None, dashed=False, lx=None, ly=None, anchor="middle"):
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        dash = ' stroke-dasharray="7,5"' if dashed else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="#333" stroke-width="2.2"{dash} marker-end="url(#arr)"/>')
        if label:
            if lx is None:
                (x1, y1), (x2, y2) = pts[-2], pts[-1]
                lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 - 7
            self.parts.append(f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" font-size="13" font-style="italic" fill="#444" '
                              f'paint-order="stroke" stroke="white" stroke-width="5">{_e(label)}</text>')

    def text(self, x, y, s, fs=14, weight=400, color="#444", anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{fs}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{_e(s)}</text>')

    def legend(self, x, y, items):
        for i, (kind, name) in enumerate(items):
            fill, stroke = C[kind]
            xx = x + i * 165
            self.parts.append(f'<rect x="{xx}" y="{y}" width="20" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            self.text(xx + 27, y + 12, name, fs=13)

    def save(self, name):
        self.parts.append("</svg>")
        (OUT / name).write_text("\n".join(self.parts), encoding="utf-8")
        print("wrote", OUT / name)


# ============================================================ 1. PRE-TRAINING PIPELINE
s = SVG("Pre-training pipeline (offline)",
        "Stage A: train the regime classifier on labels.  Stage B: train one U-Net per regime on that regime's days.  Then freeze both.")

# data sources
s.box(30, 100, 280, 92, "src", "GFS 0.25° forecast", ["rain · u850 · v850 · MSLP", "00Z · Day-1/2/3 · JJAS 2021–25"], fs=15)
s.box(30, 240, 280, 80, "src", "IMD 0.25° daily rain", ["ground truth · 129 × 135"], fs=15)
s.box(30, 365, 280, 80, "src", "DEM (SRTM 90 m)", ["GeoTIFF tiles"], fs=15)

# alignment
s.box(350, 100, 320, 92, "proc", "BOX 1 · NWP alignment", ["24-h totals on IMD day (fxx 3→27)", "bilinear regrid → 129 × 135"], fs=15)
s.box(350, 365, 320, 80, "proc", "BOX 2 · Terrain engine", ["block-average 90 m → 25 km", "elevation · land-sea mask"], fs=15)
s.box(350, 500, 320, 128, "label", "Ground truth + regime labels",
      ["y = IMD rain (mm)", "Rajeevan index → Active / Break", "MSLP low → Depression · else Coastal/Oro"], fs=15)

# tensor
s.box(710, 185, 260, 92, "proc", "Concat + z-score", ["μ, σ from train years only", "saved for live use"], fs=15)
s.box(710, 345, 260, 80, "proc", "Unified tensor", ["6 × 129 × 135"], fs=15)

# models
s.box(1010, 100, 560, 105, "model", "STAGE A · Regime classifier (CNN)",
      ["input u850, v850, MSLP · [Conv→BN→ReLU→MaxPool] × 4 → softmax(4)", "class-weighted cross-entropy"], fs=15)

s.group(1010, 235, 560, 265, "STAGE B · Four regime-specific U-Nets (hard-routed)")
s.box(1030, 270, 255, 92, "model", "Active U-Net", ["3×3 conv", "widespread rain"], fs=15)
s.box(1305, 270, 245, 92, "model", "Break U-Net", ["3×3 conv + L2", "sparse / zero-rain"], fs=15)
s.box(1030, 385, 255, 92, "model", "Depression U-Net", ["7×7 kernels", "synoptic-scale bands"], fs=15)
s.box(1305, 385, 245, 92, "model", "Coastal / Orographic U-Net", ["dense skip connections", "ridges · coasts"], fs=15)

s.box(1010, 540, 265, 105, "model", "Output head (each expert)",
      ["ch0 rain mm (softplus)", "ch1–3 P(>64.5/115.5/204.4), sigmoid"], fs=15)
s.box(1300, 540, 270, 105, "loss", "Asymmetric loss",
      ["MSE + λ · penalty on missed >64.5 mm", "+ BCE on probabilities"], fs=15)

# bottom
s.box(30, 715, 640, 66, "proc", "Season split", ["train 2021–23 · validate 2024 · test 2025"], fs=15)
s.box(710, 715, 560, 66, "out", "Saved artifacts", ["classifier.pt · 4 × unet_<regime>.pt · norm_stats.json"], fs=15)
s.box(1300, 715, 270, 66, "proc", "Optional warm start", ["init 4 experts from one shared U-Net"], fs=15, dashed=True)

# arrows
s.path([(310, 146), (350, 146)])
s.path([(310, 405), (350, 405)])
s.path([(670, 146), (690, 146), (690, 231), (710, 231)], label="4 dynamic", lx=695, ly=138, anchor="start")
s.path([(670, 405), (690, 405), (690, 231)], label="2 static", lx=695, ly=428, anchor="start")
s.path([(840, 277), (840, 345)])
s.path([(310, 280), (330, 280), (330, 564), (350, 564)])
s.path([(970, 385), (990, 385), (990, 152), (1010, 152)], label="u850 v850 MSLP", lx=985, ly=270, anchor="end")
s.path([(970, 385), (990, 385), (990, 430), (1010, 430)], label="all 6 ch", lx=995, ly=452, anchor="start")
s.path([(670, 564), (690, 564), (690, 675), (1585, 675), (1585, 152), (1570, 152)], label="regime label", lx=900, ly=670)
s.path([(510, 628), (510, 695), (1435, 695), (1435, 645)], label="y target", lx=1200, ly=709)
s.path([(1290, 500), (1290, 540)], label="ŷ", lx=1300, ly=525, anchor="start")
s.path([(1275, 592), (1300, 592)])
s.path([(1435, 540), (1435, 502)], dashed=True, label="backprop", lx=1445, ly=525, anchor="start")

s.legend(30, 810, [("src", "data"), ("proc", "processing"), ("label", "labels / truth"),
                   ("model", "neural network"), ("loss", "loss"), ("out", "artifact")])
s.save("flow_1_training_pipeline.svg")


# ============================================================ 2. SYSTEM WORKFLOW
s = SVG("System workflow (operational, daily)",
        "Same preprocessing, frozen weights.  Forecast product published immediately; verification runs when IMD observations arrive next day.")

s.box(30, 100, 250, 92, "src", "Today's 00Z GFS", ["rain · u850 · v850 · MSLP", "Day-1 / 2 / 3"], fs=15)
s.box(30, 240, 250, 70, "proc", "Daily scheduler", ["≈ 06:30 IST · < 2 min"], fs=15)
s.box(310, 100, 300, 92, "proc", "BOX 1 · preprocessing", ["24-h totals · regrid", "z-score with saved μ, σ"], fs=15)
s.box(310, 240, 270, 70, "proc", "Cached terrain", ["elevation · land mask"], fs=15)
s.box(640, 100, 250, 92, "proc", "Unified tensor", ["6 × 129 × 135"], fs=15)

s.box(920, 100, 300, 105, "model", "Regime classifier (frozen)",
      ["softmax → e.g. Depression 0.81", "argmax → regime of the day"], fs=15)
s.box(1250, 100, 320, 105, "model", "Selected U-Net (frozen)",
      ["one expert per regime", "out: rain + 3 probabilities"], fs=15)

s.box(1250, 260, 320, 92, "out", "Grid products", ["corrected rain (mm)", "P(>64.5 / 115.5 / 204.4)"], fs=15)
s.box(610, 260, 290, 92, "proc", "District aggregation", ["shapefile · mean & max", "alert colour from P"], fs=15)
s.box(610, 420, 960, 105, "out", "Dashboard (Streamlit + Folium)",
      ["date & lead · Raw / Corrected / Observed toggle · district map & table", "regime badge · metrics scoreboard"], fs=15)

s.box(30, 640, 250, 80, "src", "IMD observations", ["next day · 0.25°"], fs=15)
s.box(640, 600, 560, 128, "loss", "Verification engine (post-event)",
      ["Raw vs Corrected vs IMD @ 64.5 / 115.5 / 204.4 mm", "RMSE · ETS · CSI · POD · FAR · FSS", "by regime and lead"], fs=15)
s.box(1250, 600, 320, 128, "out", "Verification products", ["comparison table", "report (HTML / PDF)"], fs=15)

s.path([(280, 146), (310, 146)])
s.path([(610, 146), (640, 146)])
s.path([(580, 275), (625, 275), (625, 170), (640, 170)])
s.path([(890, 146), (920, 146)])
s.path([(890, 170), (905, 170), (905, 225), (1235, 225), (1235, 170), (1250, 170)], label="all 6 ch", lx=1070, ly=221)
s.path([(1220, 146), (1250, 146)], label="regime", lx=1235, ly=136)
s.path([(1410, 205), (1410, 260)])
s.path([(1250, 306), (900, 306)], label="grid → district", lx=1130, ly=299)
s.path([(755, 352), (755, 420)])
s.path([(1410, 352), (1410, 420)])
s.path([(1070, 205), (1070, 240), (960, 240), (960, 420)], dashed=True, label="regime badge", lx=968, ly=395, anchor="start")
s.path([(1570, 330), (1585, 330), (1585, 570), (1100, 570), (1100, 600)], label="corrected", lx=1240, ly=564)
s.path([(1300, 600), (1300, 525)], dashed=True, label="scoreboard", lx=1308, ly=555, anchor="start")
s.path([(460, 192), (460, 215), (295, 215), (295, 650), (640, 650)], label="raw NWP (baseline)", lx=300, ly=480, anchor="start")
s.path([(280, 690), (640, 690)], label="truth", lx=460, ly=683)
s.path([(1200, 664), (1250, 664)])

s.legend(30, 810, [("src", "input"), ("proc", "processing"), ("model", "frozen model"),
                   ("out", "deliverable"), ("loss", "verification")])
s.save("flow_2_system_workflow.svg")
