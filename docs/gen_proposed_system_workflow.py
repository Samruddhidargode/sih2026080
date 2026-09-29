"""Proposed operational system - daily system workflow (the target design, not the prototype).
Adds: fail-safe fallback to raw NWP, delivery into IMD's existing tooling (not a standalone app),
shadow-mode verification, forecaster feedback loop back into retraining.
Run: python gen_proposed_system_workflow.py
"""
from xml.sax.saxutils import escape as e

W, H = 1750, 1150
C = {
    "src": ("#DCEBFA", "#2F6DB5"), "proc": ("#EFEFEF", "#666666"), "model": ("#EBDDF5", "#7B3FA0"),
    "out": ("#DDF3E0", "#2E8B57"), "loss": ("#FDE3CC", "#D2691E"), "safety": ("#FDE8E8", "#C0392B"),
    "gov": ("#E0F7F5", "#16A085"),
}

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
    '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>',
    f'<rect width="{W}" height="{H}" fill="white"/>',
    '<text x="30" y="42" font-size="28" font-weight="700" fill="#222">Proposed system — daily operational workflow (target design)</text>',
    '<text x="30" y="66" font-size="15" fill="#555">Fails safe to the raw forecast rather than publish a bad correction. Delivers into IMD\'s existing tools, not a new app.</text>',
]


def box(x, y, w, h, kind, title, lines=(), fs=12, tfs=14.5, dashed=False):
    fill, stroke = C[kind]
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2.2"{dash}/>')
    ty = y + tfs + 8
    parts.append(f'<text x="{x+w/2}" y="{ty}" text-anchor="middle" font-size="{tfs}" font-weight="700" fill="#222">{e(title)}</text>')
    for ln in lines:
        ty += fs + 6
        parts.append(f'<text x="{x+w/2}" y="{ty}" text-anchor="middle" font-size="{fs}" fill="#333">{e(ln)}</text>')


def arrow(pts, label=None, lx=None, ly=None, dashed=False, anchor="middle", fs=12.5, color="#333"):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    dash = ' stroke-dasharray="7,5"' if dashed else ""
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{dash} marker-end="url(#arr)"/>')
    if label:
        if lx is None:
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 - 7
        parts.append(f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" font-size="{fs}" font-style="italic" fill="#444" paint-order="stroke" stroke="white" stroke-width="5">{e(label)}</text>')


def legend(x, y, items):
    for i, (k, n) in enumerate(items):
        fill, stroke = C[k]
        xx = x + i * 195
        parts.append(f'<rect x="{xx}" y="{y}" width="20" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text x="{xx+27}" y="{y+12}" font-size="14" fill="#444">{e(n)}</text>')


# ---- Column 1: ingest + classify
box(30, 100, 330, 100, "src", "NCMRWF NWP run lands", ["auto-trigger, 00Z / 12Z", "12 / 4 km operational feed"])
box(30, 240, 330, 90, "proc", "Preprocessing", ["same code path as training", "-> operational-resolution tensor"])
box(30, 370, 330, 100, "model", "Season gate + regime classifier", ["frozen — monsoon or winter mode", "decides the regime, with confidence"])

# ---- Column 2: correct + check
box(410, 100, 330, 100, "model", "Router → 1 of 6 experts", ["frozen U-Net matching the", "classified regime runs"])
box(410, 240, 330, 100, "out", "Correction output", ["rainfall (mm) +", "3 exceedance probabilities"])
box(410, 380, 330, 110, "safety", "Fail-safe sanity check", ["low classifier confidence, or an", "anomalous output — checked before anything ships"])
box(410, 530, 330, 90, "safety", "Fallback: publish raw NWP", ["+ alert the operations team", "correction suppressed for this run only"], dashed=True)
box(410, 660, 330, 90, "proc", "District aggregation", ["Survey of India boundaries", "mean (advisory) + max (flood-risk)"])

# ---- Column 3: publish
box(790, 100, 330, 100, "out", "Version-tagged output", ["stamped with model version +", "NWP version + training window"])
box(790, 240, 330, 110, "out", "Delivered into IMD's existing tools", ["GRIB2 / NetCDF feed into the", "current bulletin & GIS workflow — not a new app"])
box(790, 660, 330, 90, "gov", "Forecaster review", ["can flag a correction as wrong;", "flag is logged, never silently dropped"])

# ---- Column 4: verification + governance loop
box(1210, 100, 330, 90, "src", "IMD observations arrive", ["AWS network (near-real-time) +", "gridded product (next day)"])
box(1210, 240, 330, 130, "loss", "Verification (continuous)", ["raw vs corrected vs observed", "RMSE, ETS, CSI, POD, FAR, FSS", "stratified by regime and by lead"])
box(1210, 420, 330, 100, "gov", "Shadow-mode comparison", ["during trial phase: reviewed by", "IMD's own verification cell"])
box(1210, 660, 330, 100, "gov", "Retraining trigger", ["annually before monsoon, on NWP", "version upgrade, or on feedback trends"])

legend(30, 850, [("src", "input"), ("proc", "processing"), ("model", "frozen model"), ("out", "deliverable"),
                 ("safety", "fail-safe"), ("loss", "verification"), ("gov", "governance / human-in-loop")])

# ---- arrows: main spine
arrow([(195, 200), (195, 240)])
arrow([(195, 330), (195, 370)])
arrow([(360, 420), (410, 150)], label="wind, MSLP, 500 hPa")
arrow([(575, 200), (575, 240)])
arrow([(575, 340), (575, 380)])
arrow([(575, 490), (575, 530)], label="fail", lx=610, ly=512, color="#C0392B")
arrow([(490, 490), (490, 660)], label="pass", lx=460, ly=580, color="#2E8B57")
arrow([(575, 750), (790, 705)], dashed=True, label="raw forecast, unmodified", lx=690, ly=735, anchor="start")
arrow([(740, 700), (765, 700), (765, 295), (790, 295)], label="district table", lx=752, ly=490, anchor="start")

# ---- publish column
arrow([(955, 200), (955, 240)])
arrow([(955, 350), (955, 660)], label="")

# ---- verification branch
arrow([(1120, 295), (1210, 295)], label="corrected forecast", lx=1165, ly=283, anchor="middle")
arrow([(1375, 190), (1375, 240)])
arrow([(1375, 370), (1375, 420)])
arrow([(1375, 520), (1375, 660)])
arrow([(1210, 705), (955, 705)], label="feedback logged")
arrow([(1210, 470), (1120, 705)], dashed=True, label="informs review", lx=1080, ly=600, anchor="start")
arrow([(1375, 760), (1375, 820), (195, 820), (195, 470)], dashed=True, label="feeds the pre-training pipeline's next retraining run", lx=780, ly=812, anchor="middle")

parts.append("</svg>")
open("proposed_system_workflow.svg", "w", encoding="utf-8").write("\n".join(parts))
print("written proposed_system_workflow.svg")
