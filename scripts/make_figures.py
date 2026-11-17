"""Generate metrics table and pipeline diagram for project submission."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def make_metrics_table(save_path):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    columns = ["Model", "PSNR ↑", "SSIM ↑", "UIQM ↑", "PIQE ↓"]
    row = ["Water-CDM+ (Ours)", "12.19", "0.30", "4.49", "12.25"]
    colors = ["#e8f4fd"] * 5

    table = ax.table(
        cellText=[row],
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colColours=["#2c7bb6"] * 5,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold", fontsize=11)
            cell.set_facecolor("#2c7bb6")
        else:
            cell.set_facecolor("#e8f4fd")
            cell.set_text_props(fontsize=12)
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(1.2)

    fig.text(
        0.5, 0.08,
        "CPU-trained  ·  3.26M params  ·  128×128  ·  20000 iterations  ·  50 DDIM steps",
        ha="center", fontsize=9, color="#666666", style="italic",
    )

    fig.suptitle("Quantitative Results", fontsize=14, fontweight="bold", y=0.92)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {save_path}")


def _draw_box(ax, x, y, w, h, text, color="#4a90d9", text_color="white", fontsize=9):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.12", facecolor=color,
        edgecolor="#333333", linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color,
            wrap=True)


def _draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.8,
                        connectionstyle="arc3,rad=0"),
    )


def make_pipeline_diagram(save_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    # Input
    _draw_box(ax, 1.0, 2.75, 1.6, 1.0, "Degraded\nUnderwater\nImage", color="#e74c3c")

    # Primary branch
    _draw_box(ax, 3.8, 4.0, 2.0, 0.9, "Conditional\nU-Net", color="#2c7bb6")
    _draw_box(ax, 6.3, 4.0, 1.6, 0.9, "DDIM\nSampler", color="#2c7bb6")
    _draw_box(ax, 8.5, 4.0, 1.4, 0.9, "Primary\nOutput", color="#7fb3d8")

    # Secondary branch
    _draw_box(ax, 4.8, 1.5, 1.8, 0.9, "GMSRBC\nEnhancement", color="#27ae60")
    _draw_box(ax, 7.2, 1.5, 1.5, 0.9, "GMSRBC\nOutput", color="#7dcea0")

    # Fusion
    _draw_box(ax, 8.5, 2.75, 1.6, 0.9, "Adaptive\nHSL Fusion", color="#8e44ad", text_color="white")

    # Final output
    _draw_box(ax, 10.2, 2.75, 1.4, 1.0, "Restored\nImage", color="#f39c12")

    # Arrows: Input → branches
    _draw_arrow(ax, 1.8, 3.1, 2.8, 3.9)   # input → unet
    _draw_arrow(ax, 1.8, 2.4, 3.9, 1.6)   # input → gmsrbc

    # Primary branch arrows
    _draw_arrow(ax, 4.8, 4.0, 5.5, 4.0)   # unet → ddim
    _draw_arrow(ax, 7.1, 4.0, 7.8, 4.0)   # ddim → primary output

    # Secondary branch arrow
    _draw_arrow(ax, 5.7, 1.5, 6.45, 1.5)  # gmsrbc → gmsrbc output

    # To fusion
    _draw_arrow(ax, 8.5, 3.55, 8.5, 3.2)  # primary → fusion
    _draw_arrow(ax, 7.8, 1.7, 8.1, 2.3)   # gmsrbc output → fusion

    # Fusion → final
    _draw_arrow(ax, 9.3, 2.75, 9.5, 2.75) # fusion → restored

    # Branch labels
    ax.text(2.3, 4.4, "Primary Branch", fontsize=10, color="#2c7bb6",
            fontweight="bold", style="italic")
    ax.text(3.2, 0.7, "Secondary Branch", fontsize=10, color="#27ae60",
            fontweight="bold", style="italic")

    # Upgrade labels
    ax.text(3.8, 4.7, "Restormer MDTA\nAttention", fontsize=7, ha="center",
            color="#555555", style="italic")
    ax.text(6.3, 4.7, "UPGRADE #2\n(10 steps)", fontsize=7, ha="center",
            color="#555555", style="italic")

    # Noise input to DDIM
    ax.text(6.3, 3.3, "ε ~ N(0, I)", fontsize=8, ha="center",
            color="#888888", style="italic")
    _draw_arrow(ax, 6.3, 3.4, 6.3, 3.55)

    fig.suptitle("Water-CDM+ Pipeline Architecture", fontsize=15, fontweight="bold", y=0.97)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    make_metrics_table("results/metrics_table.png")
    make_pipeline_diagram("results/pipeline.png")
    print("\nAll figures generated.")
