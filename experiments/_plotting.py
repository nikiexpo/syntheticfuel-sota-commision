"""Shared heatmap rendering for the experiment grids.

One function, so a readability fix lands on every figure rather than on whichever
script was edited last. Three things it gets right that the first versions did
not:

* **No gridlines.** Matplotlib's default grid draws over an `imshow`, and on a
  heatmap the cell boundaries are already unambiguous -- the lines only obscure
  the annotations they cross.
* **Bold annotations.** The numbers are the content; at 8 pt on a saturated
  colormap they need the weight to stay legible.
* **Contrast picked per cell** rather than globally, so a dark value on a light
  patch and a light value on a dark patch are both readable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def heatmap_panels(
    path: Path,
    grids: dict[str, np.ndarray],
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    cmap: str = "RdYlGn",
    fmt: str = ".0f",
    norm=None,
    vmin: float | None = None,
    vmax: float | None = None,
    mask_invalid: bool = False,
    invalid_label: str = "never",
    invalid_colour: str = "#3b0a0a",
    highlight: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Render one panel per key of `grids`, sharing a colour scale.

    `highlight` optionally boxes one `(row, col)` cell per panel. Worth having
    when both axes are design variables and the question the figure answers is
    *where* the optimum sits rather than how deep it is.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    names = list(grids)
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 5.0),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids.values()])
    lo = vmin if vmin is not None else float(finite.min())
    hi = vmax if vmax is not None else float(finite.max())

    im = None
    for ax, name in zip(axes, names):
        g = grids[name]
        shown = np.ma.masked_invalid(g) if mask_invalid else g
        cm = plt.get_cmap(cmap).copy()
        if mask_invalid:
            cm.set_bad(invalid_colour)
        im = ax.imshow(shown, origin="lower", aspect="auto", cmap=cm,
                       norm=norm, vmin=None if norm else lo,
                       vmax=None if norm else hi)
        ax.set_xticks(range(len(col_labels)), col_labels)
        ax.set_yticks(range(len(row_labels)), row_labels)
        ax.set_xlabel(xlabel)
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
        ax.set_title(name, fontweight="bold")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                value = g[i, j]
                if mask_invalid and not np.isfinite(value):
                    ax.text(j, i, invalid_label, ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white")
                    continue
                # Luminance of the patch this label sits on, so the text colour
                # is chosen per cell rather than by one global threshold.
                rgba = im.cmap(im.norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, format(value, fmt), ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="black" if luminance > 0.55 else "white")

        if highlight and name in highlight:
            i, j = highlight[name]
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                                   edgecolor="black", linewidth=2.5, zorder=5))

    fig.colorbar(im, ax=axes, label=cbar_label)
    fig.suptitle(title, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
