from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path("results")
OUT_DIR = Path("results/final_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

items = [
    ("Baseline MLP", ROOT / "baseline" / "val_ap_curve.png"),
    ("Plain GRU", ROOT / "gru" / "val_ap_curve.png"),
    ("GRU + sampler", ROOT / "gru_sampler" / "val_ap_curve.png"),
    ("TCN + sampler", ROOT / "tcn_sampler" / "val_ap_curve.png"),
]

panel_width = 900
panel_height = 650
label_height = 50
gap = 30
margin = 30

panels = []
for label, path in items:
    img = Image.open(path).convert("RGB")
    img = ImageOps.contain(img, (panel_width, panel_height))

    canvas = Image.new("RGB", (panel_width, panel_height + label_height), "white")
    x = (panel_width - img.width) // 2
    y = label_height + (panel_height - img.height) // 2
    canvas.paste(img, (x, y))

    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), label, fill="black")

    panels.append(canvas)

final_width = margin * 2 + panel_width * 2 + gap
final_height = margin * 2 + (panel_height + label_height) * 2 + gap

final_img = Image.new("RGB", (final_width, final_height), "white")

positions = [
    (margin, margin),
    (margin + panel_width + gap, margin),
    (margin, margin + panel_height + label_height + gap),
    (margin + panel_width + gap, margin + panel_height + label_height + gap),
]

for panel, (x, y) in zip(panels, positions):
    final_img.paste(panel, (x, y))

out_path = OUT_DIR / "figure1_val_ap_grid.png"
final_img.save(out_path)
print(f"Saved: {out_path}")