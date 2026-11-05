"""Draw a compact, large-label overview of the implemented LIBERO V1 policy."""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(16, 5.5))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
ax.set(xlim=(0, 22), ylim=(0, 7.1))
ax.axis("off")
ink, muted = "#233246", "#647487"
blue, green, violet, orange = "#347db6", "#35876f", "#8166a6", "#c88039"


def text(x, y, label, size=17, color=ink, bold=False):
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold" if bold else "normal",
        color=color,
        linespacing=1.4,
    )


def box(x, y, w, h, label, color, face, frozen=False, size=17):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=.025,rounding_size=.13",
            facecolor=face,
            edgecolor=color,
            linewidth=1.6,
            linestyle=(0, (5, 3)) if frozen else "solid",
            zorder=2,
        )
    )
    text(x + w / 2, y + h / 2, label, size, bold=True)


def arrow(points, color=ink):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.8, zorder=1)
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color=color,
            zorder=1,
        )
    )


text(11, 6.7, "myVLA V1", 24, bold=True)
path = next(iter(sorted((ROOT / "data/libero_spatial").glob("*_demo.hdf5"))))
with h5py.File(path) as f:
    demo = next(iter(f["data"]))
    images = [f[f"data/{demo}/obs/{key}"][0][::-1] for key in ("agentview_rgb", "eye_in_hand_rgb")]
for image, x in zip(images, [0.35, 1.7]):
    ax.imshow(image, extent=(x, x + 1.2, 4.1, 5.3), aspect="auto", zorder=2)
    ax.add_patch(Rectangle((x, 4.1), 1.2, 1.2, fill=False, edgecolor="#b4c2ce", lw=1))
text(1.62, 5.65, "Two-view RGB", 16)
box(0.35, 1.65, 2.55, 1.05, "Instruction", green, "#eff8f4", size=16)

box(3.65, 4.0, 3.1, 1.35, "CLIP\nVision Encoder", blue, "#edf3fa", frozen=True)
box(3.65, 1.5, 3.1, 1.35, "CLIP\nText Encoder", green, "#eff8f4", frozen=True)
text(5.2, 5.72, "Frozen", 14, muted)
arrow([(2.9, 4.68), (3.65, 4.68)], blue)
arrow([(2.9, 2.18), (3.65, 2.18)], green)

box(8.0, 4.0, 3.05, 1.35, "Visual\nProjection", blue, "#eef5fc", size=19)
box(8.0, 1.5, 3.05, 1.35, "Pooled Text\nProjection", green, "#eff8f4", size=18)
arrow([(6.75, 4.68), (8.0, 4.68)], blue)
arrow([(6.75, 2.18), (8.0, 2.18)], green)

box(12.35, 2.7, 2.35, 1.95, "Concat\nMemory", violet, "#f5f1fa", size=17)
arrow([(11.05, 4.68), (11.7, 4.68), (11.7, 4.15), (12.35, 4.15)], blue)
arrow([(11.05, 2.18), (11.7, 2.18), (11.7, 3.15), (12.35, 3.15)], green)
box(12.15, 5.25, 2.75, 0.85, "State MLP", violet, "#f5f1fa", size=16)
arrow([(13.53, 5.25), (13.53, 4.65)], violet)

box(15.65, 2.6, 3.05, 2.15, "Action\nDecoder\n× 4", orange, "#fff5e9", size=19)
arrow([(14.7, 3.67), (15.65, 3.67)])
box(15.65, 5.25, 3.05, 0.85, "Action Queries", orange, "#fff5e9", size=16)
arrow([(17.18, 5.25), (17.18, 4.75)], orange)
arrow([(18.7, 3.67), (19.5, 3.67)], orange)

for i in range(8):
    ax.add_patch(
        Rectangle(
            (19.55 + i * 0.25, 3.18),
            0.21,
            1.0,
            facecolor=orange,
            alpha=0.35 + 0.07 * i,
            edgecolor="white",
            lw=0.7,
        )
    )
text(20.5, 4.65, "Action Chunk", 17, bold=True)
text(20.5, 2.75, "8 × 7", 22, bold=True)

# Minimal legend and scope; implementation detail stays in the model-history document.
ax.plot([0.4, 21.55], [0.95, 0.95], color="#d7e0e8", lw=1)
ax.add_patch(
    Rectangle((5.0, 0.32), 0.26, 0.26, fill=False, edgecolor=muted, linestyle=(0, (4, 2)), lw=1.4)
)
text(6.5, 0.46, "Frozen", 14, muted)
ax.add_patch(Rectangle((8.4, 0.32), 0.26, 0.26, facecolor="#eef5fc", edgecolor=blue, lw=1.4))
text(10.15, 0.46, "Trainable", 14, muted)
text(15.9, 0.46, "LIBERO-Spatial", 14, muted)

output = OUT / "v1.png"
fig.savefig(output, dpi=240, bbox_inches="tight", pad_inches=0.1, facecolor="white")
print(output)
