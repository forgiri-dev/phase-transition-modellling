"""
Matplotlib figure generators: Pxy/Txy binary diagrams, a ternary tie-line
plot, pure-component Psat parity (model vs. NIST) with the Monte Carlo
confidence band, and the condensate phase envelope.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pxy(rows, comp_a_name, comp_b_name, T, kij_value, path_name):
    x1 = [r[0] for r in rows]
    y1 = [r[1] for r in rows]
    P = [r[2] / 1e5 for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(x1, P, "-", color="#2060a8", label=f"bubble line (liquid x$_{{{comp_a_name}}}$)")
    ax.plot(y1, P, "-", color="#c0392b", label=f"dew line (vapor y$_{{{comp_a_name}}}$)")
    ax.fill_betweenx(P, x1, y1, color="#2060a8", alpha=0.08)
    ax.set_xlabel(f"mole fraction {comp_a_name}")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title(f"Pxy: {comp_a_name} + {comp_b_name} at {T:.0f} K  (PR EOS, k$_{{ij}}$={kij_value:g})")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, path_name)


def plot_psat_parity(name, T_data, P_data_bar, T_model, P_model_bar, mc_band, path_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.plot(T_model, P_model_bar, "-", color="#2060a8", label="PR EOS")
    ax.plot(T_data, P_data_bar, "o", color="#c0392b", ms=4, label="NIST reference EOS")
    if mc_band is not None:
        ax.fill_between(mc_band["T_grid"], mc_band["lower"] / 1e5, mc_band["upper"] / 1e5,
                         color="#2060a8", alpha=0.2, label=f"{int(mc_band['ci']*100)}% MC band")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("P$_{sat}$ (bar)")
    ax.set_title(f"{name}: saturation pressure")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    # Parity / deviation plot
    P_model_interp = np.interp(T_data, T_model, P_model_bar)
    dev = (P_model_interp - P_data_bar) / P_data_bar * 100
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axhline(5, color="gray", ls="--", lw=0.8)
    ax2.axhline(-5, color="gray", ls="--", lw=0.8)
    ax2.plot(T_data, dev, "o-", color="#2060a8", ms=4)
    ax2.set_xlabel("T (K)")
    ax2.set_ylabel("deviation (%)")
    ax2.set_title("PR EOS vs. NIST reference EOS")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, path_name)


def plot_ternary(points, labels, title, path_name):
    """points: list of (a,b,c) fractions summing to 1 for several tie
    lines, e.g. [((xa,xb,xc) liquid, (ya,yb,yc) vapor), ...]."""
    def to_xy(a, b, c):
        # Equilateral-triangle barycentric projection: a at top, b at
        # bottom-left, c at bottom-right.
        x = 0.5 * (2 * c + a) / (a + b + c)
        y = (np.sqrt(3) / 2) * a / (a + b + c)
        return x, y

    fig, ax = plt.subplots(figsize=(5.5, 5))
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(tri_x, tri_y, "k-", lw=1)
    for i, txt in enumerate(labels):
        tx, ty = [(0.5, np.sqrt(3) / 2 + 0.03), (-0.03, -0.03), (1.03, -0.03)][i]
        ax.text(tx, ty, txt, ha="center", fontsize=10)

    for liq, vap in points:
        xl, yl = to_xy(*liq)
        xv, yv = to_xy(*vap)
        ax.plot([xl, xv], [yl, yv], "-", color="#2060a8", lw=1)
        ax.plot(xl, yl, "o", color="#2060a8", ms=4)
        ax.plot(xv, yv, "s", color="#c0392b", ms=4)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    return _save(fig, path_name)


def plot_envelope(T_grid, P_bubble, P_dew, mc_lower=None, mc_upper=None, title="", path_name="envelope.png"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(T_grid, np.array(P_bubble) / 1e5, "-", color="#2060a8", label="bubble line")
    ax.plot(T_grid, np.array(P_dew) / 1e5, "-", color="#c0392b", label="dew line")
    if mc_lower is not None:
        ax.fill_between(T_grid, np.array(mc_lower) / 1e5, np.array(mc_upper) / 1e5,
                         color="gray", alpha=0.25, label="95% MC band")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("P (bar)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, path_name)
