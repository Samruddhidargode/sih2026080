"""Slide-ready flowcharts for the SIH26080 Technical Approach slide.

    python docs/make_slide_flowcharts.py  ->  docs/slide_system_workflow.svg (+ .png)

Layout mirrors the SIH-2025 winner deck: numbered pastel zones, one idea per box,
verification in orange, deliverables in green.  Portrait-ish so two charts sit
side by side on one 16:9 slide.  Arrowheads and label backgrounds are drawn as
plain shapes (no <marker>, no paint-order) so every renderer / PowerPoint agrees.
"""
import math
from pathlib import Path
from xml.sax.saxutils import escape as _e

OUT = Path(__file__).resolve().parent
FONT = "Arial, Segoe UI, sans-serif"

# fill, stroke  (box kinds)
C = {
    "src":   ("#D6EAF8", "#2E86C1"),   # inputs
    "proc":  ("#ECECEC", "#707070"),   # processing
    "model": ("#E8DAEF", "#7D3C98"),   # frozen neural net
    "prod":  ("#FCE4EC", "#C2185B"),   # intermediate product
    "ver":   ("#FDEBD0", "#CA6F1E"),   # verification / loss
    "out":   ("#C8F7C5", "#1E8449"),   # deliverable ("Display")
}
# zone fills (very light) + label colour
Z = {
    "blue":   ("#EAF4FB", "#1B4F72"),
    "purple": ("#F4ECF7", "#4A235A"),
    "pink":   ("#FDEEF3", "#78281F"),
    "green":  ("#EEF9EC", "#145A32"),
}


class SVG:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">',
            f'<rect width="{w}" height="{h}" fill="white"/>',
            f'<text x="{w/2}" y="34" text-anchor="middle" font-size="26" font-weight="700" fill="#111">{_e(title)}</text>',
        ]

    def zone(self, x, y, w, h, colour, label):
        fill, lc = Z[colour]
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}"/>')
        self.parts.append(f'<text x="{x + 14}" y="{y + 24}" font-size="17" font-weight="700" fill="{lc}">{_e(label)}</text>')

    def box(self, x, y, w, h, kind, title, sub=None, fs=14, tfs=16):
        fill, stroke = C[kind]
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        cy = y + h / 2
        if sub:
            self.parts.append(f'<text x="{x + w/2}" y="{cy - 3}" text-anchor="middle" font-size="{tfs}" font-weight="700" fill="#111">{_e(title)}</text>')
            self.parts.append(f'<text x="{x + w/2}" y="{cy + fs + 3}" text-anchor="middle" font-size="{fs}" fill="#333">{_e(sub)}</text>')
        else:
            self.parts.append(f'<text x="{x + w/2}" y="{cy + tfs/3}" text-anchor="middle" font-size="{tfs}" font-weight="700" fill="#111">{_e(title)}</text>')

    def pill(self, x, y, w, h, text, hot=False):
        fill, stroke = ("#BB8FCE", "#4A235A") if hot else ("#F5EEF8", "#A569BD")
        weight = 700 if hot else 500
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        self.parts.append(f'<text x="{x + w/2}" y="{y + h/2 + 5}" text-anchor="middle" font-size="13" font-weight="{weight}" fill="#111">{_e(text)}</text>')

    def arrow(self, pts, label=None, dashed=False, lx=None, ly=None):
        """Polyline + explicit triangular head at the last point."""
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        ang = math.atan2(y2 - y1, x2 - x1)
        hl, hw = 11, 6                        # head length / half-width
        bx, by = x2 - hl * math.cos(ang), y2 - hl * math.sin(ang)   # base of head
        line_pts = pts[:-1] + [(bx, by)]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in line_pts)
        dash = ' stroke-dasharray="7,5"' if dashed else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="2.2"{dash}/>')
        nx, ny = -math.sin(ang) * hw, math.cos(ang) * hw
        self.parts.append(f'<polygon points="{x2:.1f},{y2:.1f} {bx+nx:.1f},{by+ny:.1f} {bx-nx:.1f},{by-ny:.1f}" fill="#222"/>')
        if label:
            if lx is None:
                lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 - 8
            self.label(lx, ly, label)

    def label(self, x, y, s, fs=13):
        tw = len(s) * fs * 0.52
        self.parts.append(f'<rect x="{x - tw/2 - 4}" y="{y - fs}" width="{tw + 8}" height="{fs + 5}" rx="4" fill="white" fill-opacity="0.92"/>')
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{fs}" font-style="italic" fill="#333">{_e(s)}</text>')

    def save(self, name):
        self.parts.append("</svg>")
        p = OUT / name
        p.write_text("\n".join(self.parts), encoding="utf-8")
        print("wrote", p)
        try:
            import pymupdf
            pix = pymupdf.open(p)[0].get_pixmap(dpi=200)
            pix.save(p.with_suffix(".png"))
            print("wrote", p.with_suffix(".png"))
        except Exception as ex:  # png is a convenience; svg is the source
            print("png skipped:", ex)


# ====================================================================== SYSTEM WORKFLOW
W, H = 900, 1010
s = SVG(W, H, "System Workflow  (operational, every day ≈ 06:30 IST)")
s.zone(20, 55, 860, 190, "blue", "Step 1 · Ingestion & Pre-processing")
s.zone(20, 260, 860, 270, "purple", "Step 2 · Regime Diagnosis & Routing")
s.zone(20, 545, 860, 120, "pink", "Step 3 · Grid → District products")
s.zone(20, 680, 860, 315, "green", "Step 4 · Delivery & Verification")

# ---------------- Step 1 ------------------------------------------------ y 55..245
s.box(40, 95, 240, 62, "src", "Today's 00Z GFS forecast", "rain · u850 · v850 · MSLP")
s.box(330, 95, 250, 62, "proc", "Pre-processing", "24-h totals · regrid to IMD 0.25°")
s.box(630, 95, 230, 62, "proc", "Unified tensor", "6 × 129 × 135 · z-scored")
s.box(330, 175, 250, 52, "src", "Cached terrain", "SRTM elevation · land-sea mask")
s.arrow([(280, 126), (330, 126)])
s.arrow([(580, 126), (630, 126)])
s.arrow([(580, 201), (605, 201), (605, 140), (630, 140)])

# ---------------- Step 2 ------------------------------------------------ y 260..530
s.box(40, 305, 240, 70, "model", "Regime Classifier  (frozen)", "wind & pressure → regime + confidence")
# expert bank
s.parts.append('<rect x="325" y="298" width="270" height="118" rx="12" fill="none" stroke="#7D3C98" stroke-width="1.5" stroke-dasharray="6,4"/>')
s.parts.append('<text x="460" y="316" text-anchor="middle" font-size="13" font-weight="700" fill="#4A235A">Expert bank · one U-Net per regime</text>')
for i, name in enumerate(["Active", "Break", "Depression", "Coastal / Oro"]):
    r, c = divmod(i, 2)
    s.pill(340 + c * 130, 328 + r * 42, 120, 32, name, hot=(name == "Depression"))
s.box(640, 305, 220, 70, "model", "Selected U-Net  (frozen)", "e.g. Depression expert")
s.box(40, 440, 240, 62, "model", "Shared U-Net  (fallback)", "used when confidence is low")

s.arrow([(690, 157), (690, 305)], label="all 6 channels", lx=745, ly=272)
s.arrow([(700, 157), (700, 170), (160, 170), (160, 305)], label="wind & pressure only", lx=255, ly=255)
s.arrow([(280, 340), (325, 340)])
s.label(460, 445, "picks the expert trained for today's weather pattern")
s.arrow([(595, 340), (640, 340)])
s.arrow([(160, 375), (160, 440)], dashed=True)
s.arrow([(280, 471), (620, 471), (620, 375)], dashed=True, label="low confidence → fallback", lx=430, ly=490)
s.arrow([(750, 375), (750, 585)], label="corrected output", lx=750, ly=560)

# ---------------- Step 3 ------------------------------------------------ y 545..665
s.box(480, 585, 380, 62, "prod", "Corrected grid", "rain (mm) · P(> 64.5 / 115.5 / 204.4 mm)")
s.box(40, 585, 380, 62, "prod", "District aggregation", "area-weighted mean & max · alert colour from P")
s.arrow([(480, 616), (420, 616)])

# ---------------- Step 4 ------------------------------------------------ y 680..995
s.box(40, 725, 380, 70, "out", "Dashboard  (Streamlit + Folium)", "Raw / Corrected / Observed · regime badge")
s.box(40, 905, 380, 62, "out", "Display: District Alert Map", "+ probability table per district")
s.box(480, 725, 175, 52, "src", "IMD observations", "0.25° · next day")
s.box(685, 725, 175, 52, "src", "Raw GFS  (baseline)", "for comparison")
s.box(480, 810, 380, 62, "ver", "Verification engine  (next day)", "RMSE · ETS · CSI · POD · FAR · FSS")
s.box(480, 905, 380, 62, "out", "Display: Skill Scoreboard", "by regime · by lead day")

s.arrow([(230, 647), (230, 725)])
s.arrow([(230, 795), (230, 905)])
s.arrow([(670, 647), (670, 810)])                     # corrected -> engine, through the gap
s.arrow([(567, 777), (567, 810)])
s.arrow([(772, 777), (772, 810)])
s.arrow([(670, 872), (670, 905)])
s.arrow([(480, 936), (420, 936)], dashed=True, label="scores shown on dashboard", lx=450, ly=985)

s.save("slide_system_workflow.svg")
