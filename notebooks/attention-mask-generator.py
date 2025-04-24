import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button

# ───────────────────────────────-- CONFIG --─────────────────────────────────── #
GRID_SIZE        = 6           # n  → creates an n×n grid
COLOR_ON         = "#1E90FF"   # color for "on" cells
COLOR_OFF        = "#FFFFFF"   # color for "off" cells
EXPORT_DPI       = 300         # resolution of the saved PNG
# ────────────────────────────────────────────────────────────────────────────── #

def main(n: int = GRID_SIZE):
    # data array: 0 = off, 1 = on
    data = np.zeros((n, n), dtype=int)

    # two-color colormap
    cmap  = colors.ListedColormap([COLOR_OFF, COLOR_ON])
    norm  = colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("Attention-Mask Painter")
    plt.subplots_adjust(bottom=0.15)      # leave room for button

    im = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=.5)
    ax.tick_params(which="both", length=0)  # hide tick marks
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # ─── Cell-toggle callback ──────────────────────────────────────────────── #
    def onclick(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        j = int(np.floor(event.xdata + 0.5))   # column
        i = int(np.floor(event.ydata + 0.5))   # row
        if 0 <= i < n and 0 <= j < n:
            data[i, j] ^= 1    # toggle 0 ↔ 1
            im.set_data(data)
            fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    # ─── Export button ─────────────────────────────────────────────────────── #
    def export(event):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{stamp}_attention_mask.png"
        fig.savefig(fname, dpi=EXPORT_DPI, bbox_inches="tight")
        print(f"Saved: {fname}")

    btn_ax = fig.add_axes([0.35, 0.025, 0.3, 0.07])  # [left, bottom, width, height]
    button = Button(btn_ax, "Export PNG", hovercolor="#DDDDDD")
    button.on_clicked(export)

    plt.show()

if __name__ == "__main__":
    main()
