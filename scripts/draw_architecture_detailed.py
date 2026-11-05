"""Render the implemented LIBERO V2 policy architecture as editable SVG and PNG."""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)
fig, ax = plt.subplots(figsize=(22, 9))
fig.subplots_adjust(left=0.012, right=0.99, bottom=0.025, top=0.98)
ax.set(xlim=(0, 22), ylim=(0, 9))
ax.axis("off")
ink = "#203044"
muted = "#627184"
edge = "#aebbc9"
blue = "#3278b5"
green = "#32866d"
purple = "#8265a5"
orange = "#c77e34"


def txt(x, y, s, size=11, color=ink, weight="normal", ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def box(x, y, w, h, title, subtitle="", color=blue, fill="#f3f7fc", fs=12, dashed=False):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.12",
        linewidth=1.3,
        edgecolor=color,
        facecolor=fill,
        zorder=2,
        linestyle=(0, (4, 3)) if dashed else "solid",
    )
    ax.add_patch(p)
    txt(x + w / 2, y + h * 0.62 if subtitle else y + h / 2, title, fs, weight="bold", zorder=3)
    if subtitle:
        txt(x + w / 2, y + h * 0.25, subtitle, 9.3, color=muted, zorder=3, linespacing=1.5)
    return p


def arrow(points, color=ink, lw=1.45, dashed=False, head=True, zorder=1):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=color,
            lw=lw,
            zorder=zorder,
            linestyle="--" if dashed else "-",
        )
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>" if head else "-",
            mutation_scale=12,
            lw=lw,
            color=color,
            linestyle="--" if dashed else "-",
            zorder=zorder,
        )
    )


# Background group for the frozen portion.
ax.add_patch(
    FancyBboxPatch(
        (2.65, 3.35),
        2.75,
        4.35,
        boxstyle="round,pad=.02,rounding_size=.15",
        facecolor="#f5f6f8",
        edgecolor=edge,
        linewidth=1,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
)
txt(4.025, 7.37, "FROZEN ENCODERS", 10, color=muted, weight="bold")
txt(0.25, 8.65, "myVLA V2", 23, weight="bold", ha="left")
txt(
    3.25,
    8.65,
    "Lightweight bidirectional vision-language fusion for action-chunk prediction",
    14,
    ha="left",
)
# Legend.
ax.add_patch(
    Rectangle((17.0, 8.55), 0.18, 0.18, facecolor="#f5f6f8", edgecolor=edge, linestyle="--")
)
txt(17.3, 8.64, "Frozen", 10, ha="left", color=muted)
ax.add_patch(Rectangle((18.7, 8.55), 0.18, 0.18, facecolor="#eef5fc", edgecolor=blue))
txt(19.0, 8.64, "Trainable", 10, ha="left", color=muted)

# Genuine dataset images, already corrected into the model's image orientation.
path = next(iter(sorted((ROOT / "data/libero_spatial").glob("*_demo.hdf5"))), None)
if path:
    with h5py.File(path) as f:
        demo = next(iter(f["data"]))
        images = [f[f"data/{demo}/obs/{k}"][0][::-1] for k in ["agentview_rgb", "eye_in_hand_rgb"]]
else:
    raise FileNotFoundError("Need a LIBERO-Spatial demonstration for the RGB input thumbnails")
for image, y, label in zip(images, [6.15, 4.65], ["External RGB", "Wrist RGB"]):
    ax.imshow(image, extent=(0.35, 1.7, y, y + 1.1), zorder=2, aspect="auto")
    ax.add_patch(Rectangle((0.35, y), 1.35, 1.1, fill=False, edgecolor=edge, lw=1, zorder=3))
    txt(1.025, y - 0.16, label, 9.5, color=muted)
txt(1.03, 7.65, "Visual observations", 11, weight="bold")
arrow([(1.85, 6.7), (2.25, 6.7), (2.25, 6.2), (2.85, 6.2)], blue)
arrow([(1.85, 5.2), (2.25, 5.2), (2.25, 6.2)], blue, head=False)
box(
    2.85,
    5.45,
    2.35,
    1.4,
    "CLIP ViT-B/16",
    "224 × 224 per view\nDrop CLS; retain patch tokens",
    blue,
    "#edf0f4",
    fs=13,
    dashed=True,
)
box(
    5.85,
    5.55,
    1.8,
    1.2,
    "Visual projection",
    "LayerNorm + Linear\n+ camera embeddings",
    blue,
    fs=11,
)
arrow([(5.2, 6.15), (5.85, 6.15)], blue)
txt(5.52, 6.62, "2 × 196\n× 768", 9, color=blue)
arrow([(7.65, 6.15), (8.35, 6.15)], blue)
txt(8.0, 6.72, "392 × 256", 9, color=blue)

box(
    0.3,
    2.95,
    2.1,
    0.95,
    "Language instruction",
    "e.g., “pick up the black bowl”",
    green,
    "#eff8f4",
    fs=10.5,
)
box(
    2.85,
    3.65,
    2.35,
    1.2,
    "CLIP text encoder",
    "Token sequence + pooled feature",
    green,
    "#edf0f4",
    fs=12,
    dashed=True,
)
arrow([(2.4, 3.43), (2.6, 3.43), (2.6, 4.25), (2.85, 4.25)], green)
box(5.85, 3.8, 1.8, 0.95, "Text projection", "Shared Linear: 512 → 256", green, "#eff8f4", fs=11)
arrow([(5.2, 4.25), (5.85, 4.25)], green)
arrow([(7.65, 4.25), (8.35, 4.25)], green)
txt(8, 4.83, "L × 256", 9, color=green)

# Parallel bidirectional interaction in each fusion block.
box(8.35, 3.4, 3.25, 3.8, "", color="#7088a4", fill="#fafcfe")
txt(9.98, 6.9, "Bidirectional fusion × 2", 13, weight="bold")
box(8.6, 5.55, 2.75, 0.85, "Visual ← Text", "Cross-attention: Q = V, K / V = T", blue, fs=11)
box(
    8.6,
    3.85,
    2.75,
    0.85,
    "Text ← Visual",
    "Cross-attention: Q = T, K / V = V",
    green,
    "#eff8f4",
    fs=11,
)
txt(9.98, 5.15, "Gated residuals + FFNs", 11, weight="bold")
txt(9.98, 4.87, "8 heads · d = 256 · FFN = 1024", 9, color=muted)
txt(9.98, 3.05, "Zero-initialized tanh gates; text padding masked", 9, color=muted)
arrow([(8.35, 6.15), (8.6, 6.0)], blue)
arrow([(8.35, 4.25), (8.6, 4.28)], green)
# Outputs and the pooled-language residual bypass.
arrow([(11.35, 6.0), (11.9, 6.0), (11.9, 6.8), (14.55, 6.8)], blue)
txt(13.0, 7.05, "Fused visual tokens", 10, color=blue)
box(
    12.1, 3.85, 1.85, 1.0, "Language summary", "Pooled text + EOS update", green, "#eff8f4", fs=10.6
)
arrow([(11.35, 4.28), (12.1, 4.28)], green)
arrow([(5.2, 4.05), (5.5, 4.05), (5.5, 2.65), (13.03, 2.65), (13.03, 3.85)], green, dashed=True)
txt(8.4, 2.42, "Projected pooled text bypass", 9, color=green)
arrow([(13.95, 4.35), (14.25, 4.35), (14.25, 5.45), (14.55, 5.45)], green)

# State branch.
box(11.8, 7.6, 1.65, 0.65, "Robot state", "", purple, "#f5f0fa", fs=11)
txt(12.62, 7.35, "8D: xyz + axis-angle + gripper joints", 8.6, color=purple)
box(14.15, 7.6, 1.65, 0.65, "State MLP", "", purple, "#f5f0fa", fs=11)
arrow([(13.45, 7.92), (14.15, 7.92)], purple)
arrow([(14.98, 7.6), (14.98, 7.2)], purple)

# Memory and action decoder.
box(14.55, 4.25, 1.4, 2.95, "", color="#71849a", fill="#f8fafc")
txt(15.25, 6.93, "Memory", 11, weight="bold")
for y, c, label in [
    (6.25, blue, "392 visual"),
    (5.5, green, "1 language"),
    (4.75, purple, "1 state"),
]:
    ax.add_patch(
        FancyBboxPatch(
            (14.7, y - 0.23),
            1.1,
            0.48,
            boxstyle="round,pad=.01,rounding_size=.05",
            facecolor=c,
            edgecolor="none",
            alpha=0.15,
            zorder=3,
        )
    )
    txt(15.25, y, label, 8.8, color=c, zorder=4)
txt(15.25, 3.94, "394 × 256", 9, color=muted)
txt(15.25, 3.56, "Language / state\n+ modality embeddings", 8.6, color=muted)
box(
    16.55,
    4.55,
    2.25,
    2.35,
    "Transformer\naction decoder",
    "4 layers · 8 heads\nSelf-attention + cross-attention\nFFN = 1024",
    orange,
    "#fff6ec",
    fs=13,
)
arrow([(15.95, 5.77), (16.55, 5.77)], ink)
box(16.55, 7.6, 2.25, 0.65, "8 learned action queries", "", orange, "#fff6ec", fs=10.5)
arrow([(17.68, 7.6), (17.68, 6.9)], orange)
box(19.35, 5.25, 2.15, 1.0, "Action projection", "Linear: 256 → 7", orange, "#fff6ec", fs=12)
arrow([(18.8, 5.77), (19.35, 5.77)], orange)
arrow([(20.43, 5.25), (20.43, 4.72)], orange)
# Tiny action-sequence matrix glyph.
for i in range(8):
    ax.add_patch(
        Rectangle(
            (19.4 + i * 0.255, 4.05),
            0.22,
            0.5,
            facecolor=orange,
            edgecolor="white",
            lw=0.6,
            alpha=0.35 + 0.06 * i,
            zorder=3,
        )
    )
txt(20.43, 3.78, "8 × 7 action chunk", 12, weight="bold")
txt(20.43, 3.38, "6D motion + gripper logit", 9.5, color=muted)
txt(20.43, 2.94, "Execute first 4 steps; reobserve", 9.4, color=orange)

# Footer notes: keeps dimensions, learning boundaries and scope explicit.
ax.plot([0.3, 21.6], [1.8, 1.8], color="#d4dce5", lw=1)
txt(0.35, 1.35, "TRAINING", 9, weight="bold", ha="left", color=muted)
txt(2.1, 1.35, "Frozen CLIP: 149.62M parameters", 10, ha="left")
txt(8.4, 1.35, "Trainable fusion + action head: 7.78M parameters", 10, ha="left")
txt(16.35, 1.35, "Total: 157.40M (0.157B)", 10, ha="left", weight="bold")
txt(
    0.35,
    0.8,
    "Loss: Smooth-L1 motion regression + binary cross-entropy for gripper; "
    "padded action steps are masked.",
    10,
    ha="left",
    color=muted,
)
txt(
    0.35,
    0.35,
    "LIBERO-Spatial configuration · batch dimension omitted · L ≤ 77 text tokens · "
    "V2 implemented in src/myvla/model.py",
    9.4,
    ha="left",
    color=muted,
)
fig.savefig(OUT / "model_architecture_v2_detailed.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
fig.savefig(OUT / "model_architecture_v2_detailed.svg", bbox_inches="tight", pad_inches=0.12)
print(OUT / "model_architecture_v2_detailed.png")
print(OUT / "model_architecture_v2_detailed.svg")
