import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']

cats = ['wiki', 'news', 'twists', 'YuJianShi', 'kfc',
        'SanGuo', 'TianLong', 'BaiLuYuan', 'SiShiT', 'SuShiQi', 'WanLi']
entropies = [3.53, 3.80, 3.94, 3.95, 3.95, 4.00, 4.09, 4.12, 4.38, 4.48, 4.07]
match_rates = [94.0, 92.0, 92.3, 91.6, 91.4, 95.0, 92.0, 90.1, 91.0, 90.0, 91.3]
rank1 = [27.6, 24.6, 25.6, 24.7, 27.5, 25.7, 24.2, 20.1, 18.1, 19.6, 23.4]

groups = ['Ency', 'News', 'Meme', 'PopSci', 'Meme',
          'Class', 'Wuxia', 'Lit', 'Dial', 'Dial', 'Hist']
group_colors = {
    'Ency': OKABE_ITO[4], 'News': OKABE_ITO[2], 'Meme': OKABE_ITO[0],
    'PopSci': OKABE_ITO[4], 'Class': OKABE_ITO[1], 'Wuxia': OKABE_ITO[1],
    'Lit': OKABE_ITO[6], 'Dial': OKABE_ITO[5], 'Hist': OKABE_ITO[3],
}
colors = [group_colors[g] for g in groups]

BASE = '/mnt/d/FudanUniversity/Fdu1/Introduction-to-Linguistic-Science/pj/'
FIGS = BASE + 'figs/'

# ===== Fig 1: Entropy bar chart =====
fig1, ax1 = plt.subplots(figsize=(5.5, 3))
bars = ax1.bar(range(len(cats)), entropies, color=colors,
               edgecolor='white', linewidth=0.5, width=0.7)
for bar, val in zip(bars, entropies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{val:.2f}', ha='center', va='bottom', fontsize=7, color='#444')
ax1.set_xticks(range(len(cats)))
ax1.set_xticklabels(cats, rotation=30, ha='right')
ax1.set_ylabel('Average entropy (bits/char)')
ax1.set_ylim(3.0, 4.8)
ax1.axhline(y=3.96, color='#999', linestyle='--', linewidth=0.7, alpha=0.7)
ax1.text(len(cats)-0.5, 3.97, 'Takahashi 3.96 bpc', fontsize=7,
         color='#999', ha='right', va='bottom')
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=OKABE_ITO[4], label='Encyclopaedia/PopSci'),
    Patch(facecolor=OKABE_ITO[2], label='News'),
    Patch(facecolor=OKABE_ITO[0], label='Internet memes'),
    Patch(facecolor=OKABE_ITO[1], label='Classical/Wuxia'),
    Patch(facecolor=OKABE_ITO[3], label='Historical essay'),
    Patch(facecolor=OKABE_ITO[6], label='Modern literature'),
    Patch(facecolor=OKABE_ITO[5], label='Dialect literature'),
]
ax1.legend(handles=legend_elements, loc='upper left', frameon=False, ncol=2, fontsize=7)
fig1.tight_layout()
fig1.savefig(FIGS + 'fig_entropy.png')
fig1.savefig(FIGS + 'fig_entropy.pdf')
print('Saved fig_entropy')

# ===== Fig 2: Scatter — Match Rate vs Entropy =====
fig2, ax2 = plt.subplots(figsize=(5, 3.5))
ax2.scatter(entropies, match_rates, c=colors, s=60, edgecolors='white',
            linewidth=0.8, zorder=3)
for i, cat in enumerate(cats):
    dx, dy = 0.03, 0.3
    if cat == 'kfc': dy = -0.6
    elif cat == 'YuJianShi': dx, dy = -0.08, -0.6
    elif cat == 'twists': dx, dy = -0.05, 0.4
    ax2.annotate(cat, (entropies[i], match_rates[i]),
                 xytext=(dx, dy), textcoords='offset fontsize',
                 fontsize=7, color='#555')
ax2.set_xlabel('Average entropy (bits/char)')
ax2.set_ylabel('Match rate (%)')
ax2.set_xlim(3.3, 4.7)
ax2.set_ylim(89, 96)
ax2.yaxis.grid(True, linestyle=':', linewidth=0.4, alpha=0.5)
ax2.set_axisbelow(True)
fig2.tight_layout()
fig2.savefig(FIGS + 'fig_scatter.png')
fig2.savefig(FIGS + 'fig_scatter.pdf')
print('Saved fig_scatter')

# ===== Fig 3: Rank-1 horizontal bar =====
fig3, ax3 = plt.subplots(figsize=(5, 3.5))
sorted_idx = np.argsort(rank1)
sorted_cats = [cats[i] for i in sorted_idx]
sorted_r1 = [rank1[i] for i in sorted_idx]
sorted_colors = [colors[i] for i in sorted_idx]
bars3 = ax3.barh(range(len(sorted_cats)), sorted_r1, color=sorted_colors,
                 edgecolor='white', linewidth=0.5, height=0.6)
for bar, val in zip(bars3, sorted_r1):
    ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', ha='left', va='center', fontsize=7, color='#444')
ax3.set_yticks(range(len(sorted_cats)))
ax3.set_yticklabels(sorted_cats)
ax3.set_xlabel('Rank-1 accuracy (%)')
ax3.set_xlim(0, 32)
fig3.tight_layout()
fig3.savefig(FIGS + 'fig_rank1.png')
fig3.savefig(FIGS + 'fig_rank1.pdf')
print('Saved fig_rank1')

# ===== Fig 4: H(N) decay with power-law fit =====
import json
from collections import defaultdict
from scipy.optimize import curve_fit

fig4, ax4 = plt.subplots(figsize=(5.5, 3.8))

hn_cats = {
    'sanguo':        ('SanGuo',     OKABE_ITO[0]),
    'bailuyuan':     ('BaiLuYuan',  OKABE_ITO[1]),
    'human_jianshi': ('YuJianShi',  OKABE_ITO[2]),
    'tianlongbabu':  ('TianLong',   OKABE_ITO[5]),
}

LOG_EDGES = [5, 10, 20, 40, 70, 110, 160, 200]

def power_law(N, H_inf, a, b):
    return H_inf + a * N**(-b)

def log_bin_idx(pl):
    for i in range(len(LOG_EDGES) - 1):
        if pl < LOG_EDGES[i + 1]:
            return i
    return len(LOG_EDGES) - 2

for cat_file, (label, color) in hn_cats.items():
    bins = defaultdict(list)
    with open(BASE + f'results/{cat_file}.jsonl') as f:
        for line in f:
            d = json.loads(line)
            pl = d['prefix_len']
            if pl >= 5:
                bins[log_bin_idx(pl)].append(d['entropy'])

    N_data, H_data = [], []
    for bi in sorted(bins.keys()):
        es = bins[bi]
        if len(es) >= 3:
            N_data.append((LOG_EDGES[bi] + LOG_EDGES[bi + 1]) / 2)
            H_data.append(np.mean(es))

    N_arr = np.array(N_data)
    H_arr = np.array(H_data)

    popt, _ = curve_fit(power_law, N_arr, H_arr, p0=[3.0, 5.0, 0.6], maxfev=10000)
    H_inf, a_fit, b_fit = popt
    residuals = H_arr - power_law(N_arr, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((H_arr - np.mean(H_arr))**2)
    r2 = 1 - ss_res / ss_tot

    ax4.scatter(N_arr, H_arr, color=color, s=22, alpha=0.8, zorder=3)

    N_fit = np.linspace(5, 300, 400)
    H_fit = power_law(N_fit, *popt)
    ax4.plot(N_fit, H_fit, color=color, linewidth=1.3,
             label=f'{label}  $H_\\infty$={H_inf:.1f}  $R^2$={r2:.2f}')

    ax4.axhline(y=H_inf, color=color, linestyle=':', linewidth=0.6, alpha=0.4)

ax4.axhline(y=9.62, color='#aaa', linestyle='--', linewidth=0.7, alpha=0.6)
ax4.text(290, 9.82, '$H_0 = 9.62$', fontsize=7, color='#999', ha='right')

ax4.set_xlabel('Context length $N$ (characters)')
ax4.set_ylabel('Conditional entropy $H(N)$ (bits)')
ax4.set_xlim(0, 310)
ax4.set_ylim(0, 11)
ax4.legend(loc='upper right', frameon=False, fontsize=7.5)
ax4.yaxis.grid(True, linestyle=':', linewidth=0.4, alpha=0.4)
ax4.set_axisbelow(True)
fig4.tight_layout()
fig4.savefig(FIGS + 'fig_hn_decay.png')
fig4.savefig(FIGS + 'fig_hn_decay.pdf')
print('Saved fig_hn_decay')

print('Done.')
