"""System workflow diagram (operational, every forecast day).
Each box: title + 2-3 short phrases explaining what happens INSIDE that step.
Run: python gen_system_workflow.py
"""
from xml.sax.saxutils import escape as e

W, H = 1600, 980
C = {
    "src": ("#DCEBFA", "#2F6DB5"), "proc": ("#EFEFEF", "#666666"), "model": ("#EBDDF5", "#7B3FA0"),
    "out": ("#DDF3E0", "#2E8B57"), "loss": ("#FDE3CC", "#D2691E"),
}

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
    '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>',
    f'<rect width="{W}" height="{H}" fill="white"/>',
    '<text x="30" y="42" font-size="28" font-weight="700" fill="#222">System workflow (operational, every forecast day)</text>',
    '<text x="30" y="66" font-size="15" fill="#555">Same preprocessing as training, frozen weights. Forecast published immediately; verification runs next day when IMD observations arrive.</text>',
]


def box(x, y, w, h, kind, title, lines=(), fs=12.5, tfs=15.5):
    fill, stroke = C[kind]
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>')
    ty = y + tfs + 8
    parts.append(f'<text x="{x+w/2}" y="{ty}" text-anchor="middle" font-size="{tfs}" font-weight="700" fill="#222">{e(title)}</text>')
    for ln in lines:
        ty += fs + 6
        parts.append(f'<text x="{x+w/2}" y="{ty}" text-anchor="middle" font-size="{fs}" fill="#333">{e(ln)}</text>')


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


box(30, 100, 260, 120, "src", "Today's 00Z GFS",
    ["rain, u850, v850, MSLP", "Day-1 / 2 / 3", "same fields as training, fetched live"], fs=11.5)
box(30, 260, 260, 85, "proc", "Daily scheduler",
    ["runs ~06:30 IST, right after", "the 00Z GFS run lands (~05:30)"], fs=11.5)

box(325, 100, 320, 120, "proc", "BOX 1 · preprocessing",
    ["24-h totals, regrid to 129 × 135", "z-score using SAVED training mean/std",
     "identical code path used at training time"], fs=11.5)
box(325, 260, 280, 85, "proc", "Cached terrain (static)",
    ["elevation + land mask", "computed once, reused every day"], fs=11.5)

box(680, 100, 260, 120, "proc", "Unified tensor",
    ["6 × 129 × 135, one snapshot", "identical layout to training input"], fs=11.5)

box(975, 100, 300, 128, "model", "Regime classifier (frozen)",
    ["softmax -> e.g. Depression 0.81", "argmax -> regime of the day",
     "weights frozen: no learning happens live"], fs=11)
box(1310, 100, 260, 128, "model", "Selected U-Net (frozen)",
    ["loads only the 1 of 4 expert", "networks that matches the regime", "out: rain (mm) + 3 probabilities"], fs=11)

box(975, 280, 595, 108, "out", "Grid products",
    ["corrected 24-h rain (mm) · P(>64.5 / 115.5 / 204.4 mm)", "129 × 135 maps, same grid as the raw input"], fs=12)

box(645, 415, 300, 100, "proc", "District aggregation",
    ["area-weighted overlay: grid -> 676 districts", "mean = advisory value, max = flood-risk flag",
     "alert colour set from the probability"], fs=10.5)
box(975, 415, 595, 100, "out", "Dashboard (Streamlit + Folium)",
    ["date & lead picker · Raw / Corrected / Observed toggle", "district map & table · regime badge · metrics scoreboard",
     "reads precomputed files only - no model runs live"], fs=11)

box(30, 620, 280, 95, "src", "IMD observations",
    ["ground truth for the same day", "arrives ~24 h after the forecast"], fs=11.5)
box(645, 590, 460, 140, "loss", "Verification engine (next day)",
    ["Raw vs Corrected vs IMD, at 64.5/115.5/204.4 mm", "RMSE, ETS, CSI, POD, FAR (xskillscore)",
     "+ FSS (neighbourhood skill, custom)", "grouped by regime and by lead time"], fs=11)
box(1160, 590, 410, 140, "out", "Verification products",
    ["comparison table: raw vs corrected", "auto-generated report (HTML/PDF)",
     "feeds the dashboard's scoreboard"], fs=11.5)

# arrows
arrow([(290, 160), (325, 160)])
arrow([(605, 160), (680, 160)])
arrow([(605, 302), (630, 302), (630, 200), (680, 200)])
arrow([(940, 160), (975, 160)], label="wind + pressure", lx=957, ly=92, anchor="middle")
arrow([(940, 175), (960, 175), (960, 215), (975, 215)], label="all 6 channels", lx=965, ly=245, anchor="start")
arrow([(1275, 160), (1310, 160)], label="regime")
arrow([(1440, 228), (1440, 280), (1272, 280)])
arrow([(1272, 388), (945, 465)], label="grid -> district", lx=1100, ly=440, anchor="middle")
arrow([(795, 515), (795, 555), (1120, 555), (1120, 415)], dashed=True, label="regime badge", lx=1000, ly=548, anchor="start")
arrow([(1272, 388), (1272, 415)])
arrow([(310, 667), (645, 667)], label="ground truth")
arrow([(170, 355), (170, 585), (645, 660)], label="raw NWP (baseline for comparison)", lx=300, ly=500, anchor="start")
arrow([(1272, 515), (1130, 590)], dashed=True, label="corrected forecast", lx=1150, ly=555, anchor="middle")
arrow([(1105, 590), (1160, 660)])
arrow([(1365, 590), (1365, 555), (1272, 505)], dashed=True, label="scoreboard update", lx=1330, ly=545, anchor="start")

legend(30, 890, [("src", "input"), ("proc", "processing"), ("model", "frozen model"),
                 ("out", "deliverable"), ("loss", "verification")])
parts.append("</svg>")
open("system_workflow.svg", "w", encoding="utf-8").write("\n".join(parts))
print("written system_workflow.svg")
