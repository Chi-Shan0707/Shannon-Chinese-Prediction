from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent.parent
FIGS = BASE / "figs"
FIGS.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

COLORS = {
    "Reference": "#4C78A8",
    "News": "#59A14F",
    "Internet": "#F2A104",
    "Classical/Wuxia": "#56B4E9",
    "Literature": "#CC79A7",
    "Dialect": "#D55E00",
    "Historical": "#B8A70D",
}

CATEGORIES = [
    {"key": "wiki", "label": "Wiki", "source": "Wikipedia", "group": "Reference", "steps": 4228, "match": 94.0, "rank1": 27.6, "entropy": 3.53},
    {"key": "news", "label": "News", "source": "News", "group": "News", "steps": 2693, "match": 92.0, "rank1": 24.6, "entropy": 3.80},
    {"key": "internet_twists", "label": "Twists", "source": "Internet twists", "group": "Internet", "steps": 1323, "match": 92.3, "rank1": 25.6, "entropy": 3.94},
    {"key": "human_jianshi", "label": "Sapiens", "source": "Popular science", "group": "Reference", "steps": 6450, "match": 91.6, "rank1": 24.7, "entropy": 3.95},
    {"key": "kfc", "label": "KFC", "source": "KFC memes", "group": "Internet", "steps": 1497, "match": 91.4, "rank1": 27.5, "entropy": 3.95},
    {"key": "sanguo", "label": "SanGuo", "source": "Classical novel", "group": "Classical/Wuxia", "steps": 7818, "match": 95.0, "rank1": 25.7, "entropy": 4.00},
    {"key": "wanli", "label": "WanLi", "source": "Historical essay", "group": "Historical", "steps": 7590, "match": 91.3, "rank1": 23.4, "entropy": 4.07},
    {"key": "tianlongbabu", "label": "TianLong", "source": "Wuxia novel", "group": "Classical/Wuxia", "steps": 7425, "match": 92.0, "rank1": 24.2, "entropy": 4.09},
    {"key": "bailuyuan", "label": "BaiLuYuan", "source": "Modern literature", "group": "Literature", "steps": 7399, "match": 90.1, "rank1": 20.1, "entropy": 4.12},
    {"key": "sishitongtang", "label": "SiShiT", "source": "Beijing dialect", "group": "Dialect", "steps": 6188, "match": 91.0, "rank1": 18.1, "entropy": 4.38},
    {"key": "sushi_qiren", "label": "SuShiQi", "source": "Tianjin dialect", "group": "Dialect", "steps": 6885, "match": 90.0, "rank1": 19.6, "entropy": 4.48},
]

FITS = [
    {"label": "SanGuo", "color": "#F2A104", "h_inf": 3.00, "b": 0.681, "r2": 0.99, "h5": 4.70},
    {"label": "BaiLuYuan", "color": "#CC79A7", "h_inf": 3.22, "b": 0.541, "r2": 0.99, "h5": 4.85},
    {"label": "Sapiens", "color": "#4C78A8", "h_inf": 3.21, "b": 1.035, "r2": 0.93, "h5": 4.75},
    {"label": "TianLong", "color": "#56B4E9", "h_inf": 3.05, "b": 0.955, "r2": 0.93, "h5": 4.55},
]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png")
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)
    print(f"Saved {name}")


def color_for(item):
    return COLORS[item["group"]]


def plot_entropy():
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    labels = [item["label"] for item in CATEGORIES]
    values = [item["entropy"] for item in CATEGORIES]
    colors = [color_for(item) for item in CATEGORIES]
    bars = ax.bar(range(len(CATEGORIES)), values, color=colors, edgecolor="white", linewidth=0.7, width=0.72)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=7.5, color="#444")

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean next-token entropy (nats/step)")
    ax.set_ylim(3.25, 4.70)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.45)
    ax.set_axisbelow(True)

    handles = [mpl.patches.Patch(facecolor=color, label=group) for group, color in COLORS.items()]
    ax.legend(handles=handles, loc="upper left", ncol=3, frameon=False)
    save(fig, "fig_entropy")


def plot_scatter():
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    entropy = np.array([item["entropy"] for item in CATEGORIES])
    match = np.array([item["match"] for item in CATEGORIES])
    sizes = np.array([item["rank1"] for item in CATEGORIES])
    colors = [color_for(item) for item in CATEGORIES]

    ax.scatter(entropy, match, s=(sizes - 12) * 12, c=colors, edgecolors="white", linewidth=0.8, zorder=3)
    offsets = {
        "Wiki": (0.02, 0.12),
        "SanGuo": (0.02, 0.10),
        "KFC": (0.03, -0.45),
        "Sapiens": (-0.10, 0.24),
        "WanLi": (0.02, -0.28),
        "BaiLuYuan": (0.02, 0.08),
        "SiShiT": (0.02, 0.10),
        "SuShiQi": (0.02, 0.08),
    }
    for item in CATEGORIES:
        dx, dy = offsets.get(item["label"], (0.02, 0.10))
        ax.annotate(item["label"], (item["entropy"], item["match"]), xytext=(dx, dy), textcoords="offset fontsize", fontsize=7.5, color="#4a4a4a")

    ax.set_xlabel("Mean next-token entropy (nats/step)")
    ax.set_ylabel("Top-1000 match rate (%)")
    ax.set_xlim(3.40, 4.68)
    ax.set_ylim(89.5, 95.5)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.55)
    ax.set_axisbelow(True)
    ax.text(3.42, 89.72, "bubble area = rank-1 accuracy", fontsize=7, color="#777")
    save(fig, "fig_scatter")


def plot_rank1():
    fig, ax = plt.subplots(figsize=(5.3, 3.8))
    ordered = sorted(CATEGORIES, key=lambda item: item["rank1"])
    labels = [item["label"] for item in ordered]
    rank1 = [item["rank1"] for item in ordered]
    colors = [color_for(item) for item in ordered]
    bars = ax.barh(range(len(ordered)), rank1, color=colors, edgecolor="white", linewidth=0.6, height=0.64)

    for bar, value in zip(bars, rank1):
        ax.text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", ha="left", va="center", fontsize=7.5, color="#444")

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rank-1 accuracy (%)")
    ax.set_xlim(0, 30)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.45)
    ax.set_axisbelow(True)
    save(fig, "fig_rank1")


def fitted_curve(fit, x):
    amplitude = (fit["h5"] - fit["h_inf"]) * (5 ** fit["b"])
    return fit["h_inf"] + amplitude * x ** (-fit["b"])


def plot_hn_decay():
    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    x = np.linspace(5, 220, 300)
    sample_x = np.array([7.5, 15, 30, 55, 90, 135, 180])

    for fit in FITS:
        color = fit["color"]
        y = fitted_curve(fit, x)
        sample_y = fitted_curve(fit, sample_x)
        ax.plot(x, y, color=color, linewidth=1.5, label=f"{fit['label']}  $H_\\infty$={fit['h_inf']:.2f}  $R^2$={fit['r2']:.2f}")
        ax.scatter(sample_x, sample_y, color=color, s=20, alpha=0.72, zorder=3)
        ax.axhline(fit["h_inf"], color=color, linestyle=":", linewidth=0.7, alpha=0.35)

    ax.set_xlabel("Context length N (characters)")
    ax.set_ylabel("Conditional entropy H(N) (nats/step)")
    ax.set_xlim(0, 225)
    ax.set_ylim(2.85, 5.05)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    save(fig, "fig_hn_decay")


def main():
    plot_entropy()
    plot_scatter()
    plot_rank1()
    plot_hn_decay()
    print("Done.")


if __name__ == "__main__":
    main()
