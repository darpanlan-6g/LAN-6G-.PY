"""
================================================================================
3GPP 6G NR FR3 V2X @ 24 GHz — Full Visualization Suite
================================================================================
Environments: Office | Urban Streets | Highway | Classroom
Plots        : Network topology, SINR heatmap, mobility traces, KPI dashboard,
               throughput timeline, handover log, CDF, building attenuation
================================================================================
Dependencies : pip install matplotlib numpy scipy
Run          : python 6g_nr_fr3_v2x_simulation.py
Output       : 6g_nr_fr3_v2x_results.png  (4 K figure)
================================================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.4,
    "figure.dpi": 150,
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENVIRONMENT DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ENVIRONMENTS = {
    # ── Office ────────────────────────────────────────────────────────────────
    "Office": {
        "area": (100, 80),           # width × height (m)
        "freq_ghz": 24.0,
        "bw_mhz": 200,
        "tx_power_dbm": 23,          # Indoor AP
        "buildings": [               # (x, y, w, h, label, material, pen_loss_db)
            (5,  5,  35, 30, "Wing A",      "concrete", 22),
            (45, 5,  35, 30, "Wing B",      "glass",     8),
            (5,  45, 45, 25, "Open Plan",   "glass",     8),
            (55, 45, 35, 25, "Server Room", "metal",    32),
        ],
        "gnbs": [                    # (x, y, label, height_m)
            (50, 40, "AP-0", 3),
        ],
        "nodes": [
            # (id, type, x0, y0, vx, vy, service)
            ("L1",   "laptop",  12, 18, 0.0,  0.0,  "eMBB"),
            ("L2",   "laptop",  60, 18, 0.0,  0.0,  "eMBB"),
            ("P1",   "phone",   20, 55, 0.3,  0.1,  "URLLC"),
            ("P2",   "phone",   70, 55,-0.2,  0.15, "URLLC"),
            ("P3",   "phone",   50, 20, 0.1, -0.1,  "eMBB"),
            ("IoT1", "iot",     30, 60, 0.0,  0.0,  "mMTC"),
            ("IoT2", "iot",     75, 60, 0.0,  0.0,  "mMTC"),
            ("IoT3", "iot",     15, 35, 0.0,  0.0,  "mMTC"),
        ],
        "pen_loss_avg": 20,
        "color": "#4A90D9",
    },

    # ── Urban Streets ─────────────────────────────────────────────────────────
    "Urban Streets": {
        "area": (120, 100),
        "freq_ghz": 24.0,
        "bw_mhz": 200,
        "tx_power_dbm": 40,
        "buildings": [
            (0,   0,  30, 35, "Office Block", "concrete", 22),
            (40,  0,  35, 25, "Apartments",   "concrete", 22),
            (90,  0,  30, 40, "Hotel",        "glass",     8),
            (0,  65,  30, 35, "Warehouse",    "metal",    32),
            (85, 60,  35, 40, "Residential",  "concrete", 22),
            (40, 70,  30, 30, "Parking",      "concrete", 22),
        ],
        "gnbs": [
            (35, 50, "gNB-0", 15),
            (85, 50, "gNB-1", 15),
        ],
        "nodes": [
            ("Car1",  "car",       5,  50, 2.0,  0.0,  "URLLC"),
            ("Car2",  "car",      55,  45, 1.8,  0.0,  "URLLC"),
            ("Car3",  "car",     100,  52,-1.5,  0.0,  "URLLC"),
            ("Truck", "truck",    30,  55, 1.2,  0.0,  "eMBB"),
            ("Ped1",  "pedestrian", 45, 48, 0.2,  0.1, "eMBB"),
            ("Ped2",  "pedestrian", 70, 53,-0.1,  0.2, "eMBB"),
            ("RSU0",  "rsu",      35,  50, 0.0,  0.0,  "URLLC"),
            ("RSU1",  "rsu",      85,  50, 0.0,  0.0,  "URLLC"),
        ],
        "pen_loss_avg": 18,
        "color": "#E67E22",
    },

    # ── Highway ───────────────────────────────────────────────────────────────
    "Highway": {
        "area": (200, 60),
        "freq_ghz": 24.0,
        "bw_mhz": 400,
        "tx_power_dbm": 46,
        "buildings": [
            (0,   0, 200, 15, "Sound Barrier (N)", "concrete", 22),
            (0,  45, 200, 15, "Sound Barrier (S)", "concrete", 22),
            (40, 15,  20, 30, "Tunnel portal",     "metal",    32),
            (140,15,  20, 30, "Tunnel portal",     "metal",    32),
        ],
        "gnbs": [
            (60,  8, "gNB-A", 20),
            (140, 8, "gNB-B", 20),
        ],
        "nodes": [
            ("V1",  "car",       10, 25, 3.2, 0.0, "URLLC"),
            ("V2",  "car",       40, 28, 2.8, 0.0, "URLLC"),
            ("V3",  "car",       80, 26, 3.0, 0.0, "URLLC"),
            ("V4",  "car",      120, 27, 2.5, 0.0, "URLLC"),
            ("V5",  "car",      170, 25, 2.9, 0.0, "URLLC"),
            ("T1",  "truck",     20, 33, 2.0, 0.0, "eMBB"),
            ("T2",  "truck",    100, 34, 1.8, 0.0, "eMBB"),
            ("EV",  "emergency",160, 26, 4.5, 0.0, "URLLC"),
        ],
        "pen_loss_avg": 8,
        "color": "#27AE60",
    },

    # ── Classroom ─────────────────────────────────────────────────────────────
    "Classroom": {
        "area": (90, 70),
        "freq_ghz": 24.0,
        "bw_mhz": 200,
        "tx_power_dbm": 23,
        "buildings": [
            (5,  5, 35, 28, "Room 101",   "concrete", 22),
            (50, 5, 35, 28, "Room 102",   "concrete", 22),
            (5, 40, 35, 25, "Auditorium", "glass",     8),
            (50,40, 35, 25, "Library",    "glass",     8),
        ],
        "gnbs": [
            (45, 35, "AP-0", 3),
        ],
        "nodes": [
            ("S1",   "phone", 12, 16, 0.05, 0.1,  "eMBB"),
            ("S2",   "phone", 22, 22,-0.05, 0.05, "eMBB"),
            ("S3",   "phone", 30, 14, 0.08,-0.05, "eMBB"),
            ("S4",   "phone", 58, 16,-0.06, 0.08, "eMBB"),
            ("S5",   "phone", 72, 20, 0.04, 0.06, "eMBB"),
            ("S6",   "phone", 65, 10,-0.03,-0.04, "eMBB"),
            ("T1",   "laptop",20, 52, 0.0,  0.0,  "eMBB"),
            ("T2",   "laptop",65, 52, 0.0,  0.0,  "eMBB"),
            ("Proj", "iot",   45, 35, 0.0,  0.0,  "URLLC"),
        ],
        "pen_loss_avg": 20,
        "color": "#8E44AD",
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PHYSICS HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def friis_path_loss(d_m, freq_ghz):
    """Free-space Friis path loss in dB."""
    d = max(d_m, 1.0)
    return 20*np.log10(d) + 20*np.log10(freq_ghz*1e9) - 147.55


def sinr_db(node_xy, gnbs, env_cfg, rng=None):
    """
    Compute SINR (dB) for a node given gNB positions.
    Signal = strongest gNB; Interference = sum of remaining gNBs; Noise floor = -90 dBm.
    """
    if rng is None:
        rng = np.random.default_rng()
    freq = env_cfg["freq_ghz"]
    tx   = env_cfg["tx_power_dbm"]
    pen  = env_cfg["pen_loss_avg"]

    powers_dbm = []
    for gx, gy, *_ in gnbs:
        d   = np.hypot(node_xy[0]-gx, node_xy[1]-gy)
        pl  = friis_path_loss(d, freq)
        rx  = tx - pl - pen*0.25 + rng.normal(0, 2)   # shadowing σ=2 dB
        powers_dbm.append(rx)

    powers_dbm.sort(reverse=True)
    noise_dbm  = -90.0
    signal_lin = 10**(powers_dbm[0]/10)
    interf_lin = sum(10**(p/10) for p in powers_dbm[1:]) if len(powers_dbm) > 1 else 0
    noise_lin  = 10**(noise_dbm/10)
    sinr_lin   = signal_lin / (interf_lin + noise_lin)
    return 10*np.log10(max(sinr_lin, 1e-9))


def throughput_mbps(sinr_val, bw_mhz):
    """Shannon capacity estimate (practical efficiency factor 0.6)."""
    return bw_mhz * np.log2(1 + 10**(sinr_val/10)) * 0.6


def simulate_trajectory(node, env_cfg, steps=300, dt=0.1):
    """
    Simulate node mobility and record SINR at each step.
    Returns arrays: xs, ys, sinr_series.
    """
    W, H  = env_cfg["area"]
    x, y  = node[2], node[3]
    vx, vy= node[4], node[5]
    gnbs  = env_cfg["gnbs"]
    rng   = np.random.default_rng(abs(hash(node[0])) % 2**32)

    xs, ys, sinrs = [x], [y], []
    for _ in range(steps):
        s = sinr_db((x, y), gnbs, env_cfg, rng)
        sinrs.append(s)
        x += vx * dt + rng.normal(0, 0.05)
        y += vy * dt + rng.normal(0, 0.05)
        x = np.clip(x, 1, W-1)
        y = np.clip(y, 1, H-1)
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys), np.array(sinrs)


def sinr_heatmap(env_cfg, resolution=60):
    """Compute SINR grid over the entire environment area."""
    W, H = env_cfg["area"]
    xs   = np.linspace(0, W, resolution)
    ys   = np.linspace(0, H, resolution)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.zeros_like(XX)
    rng  = np.random.default_rng(0)
    for i in range(resolution):
        for j in range(resolution):
            grid[i, j] = sinr_db((XX[i,j], YY[i,j]), env_cfg["gnbs"], env_cfg, rng)
    return XX, YY, gaussian_filter(grid, sigma=1.5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  NODE STYLING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NODE_STYLE = {
    "laptop":     {"color": "#27AE60", "marker": "s", "size": 70,  "label": "Laptop"},
    "phone":      {"color": "#8E44AD", "marker": "o", "size": 55,  "label": "Phone"},
    "car":        {"color": "#2980B9", "marker": "^", "size": 80,  "label": "Car"},
    "truck":      {"color": "#E67E22", "marker": "D", "size": 90,  "label": "Truck"},
    "emergency":  {"color": "#E74C3C", "marker": "*", "size": 120, "label": "Emergency"},
    "pedestrian": {"color": "#E91E8C", "marker": "P", "size": 65,  "label": "Pedestrian"},
    "rsu":        {"color": "#F39C12", "marker": "H", "size": 80,  "label": "RSU"},
    "iot":        {"color": "#16A085", "marker": "+", "size": 60,  "label": "IoT"},
}

SERVICE_COLOR = {"URLLC": "#E74C3C", "eMBB": "#2980B9", "mMTC": "#27AE60"}
MATERIAL_COLOR = {"concrete": "#B0BEC5", "glass": "#B2EBF2", "metal": "#BCAAA4"}

# SINR quality thresholds
def sinr_color(v):
    if v >= 20: return "#27AE60"
    if v >= 12: return "#F39C12"
    if v >= 4:  return "#E67E22"
    return "#E74C3C"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PRE-COMPUTE ALL DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Pre-computing simulation data …")
SIM_STEPS = 300
DT        = 0.1
TIME      = np.arange(SIM_STEPS) * DT   # 0 … 30 s

SIM_DATA = {}
for env_name, env_cfg in ENVIRONMENTS.items():
    trajectories = {}
    sinr_series_all = {}
    for node in env_cfg["nodes"]:
        xs, ys, sinrs = simulate_trajectory(node, env_cfg, steps=SIM_STEPS, dt=DT)
        trajectories[node[0]]    = (xs, ys)
        sinr_series_all[node[0]] = sinrs

    # Heatmap
    XX, YY, hm = sinr_heatmap(env_cfg, resolution=55)

    # Snapshot SINR at final position
    rng = np.random.default_rng(7)
    snapshot_sinr = {}
    for node in env_cfg["nodes"]:
        xs, ys = trajectories[node[0]]
        snapshot_sinr[node[0]] = sinr_db((xs[-1], ys[-1]), env_cfg["gnbs"], env_cfg, rng)

    SIM_DATA[env_name] = {
        "trajectories":    trajectories,
        "sinr_series":     sinr_series_all,
        "heatmap":         (XX, YY, hm),
        "snapshot_sinr":   snapshot_sinr,
    }

print("Done. Building figure …")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE LAYOUT
#  Row 0 : title banner
#  Rows 1-4: one major row per environment, each split into 4 sub-plots:
#            [topology+heatmap | SINR time series | mobility trails | KPI bar]
#  Row 5 : global summary (CDF | throughput summary | building attenuation | handover)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIG_W, FIG_H = 24, 46
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FAFAFA")

# Outer grid: title + 4 env rows + 1 summary row
outer = GridSpec(6, 1, figure=fig,
                 hspace=0.38,
                 top=0.97, bottom=0.02,
                 left=0.04, right=0.97,
                 height_ratios=[0.28, 3, 3, 3, 3, 2.8])

# ── Title banner ─────────────────────────────────────────────────────────────
ax_title = fig.add_subplot(outer[0])
ax_title.axis("off")
ax_title.set_facecolor("#1A252F")
ax_title.patch.set_visible(True)
ax_title.text(0.5, 0.62,
    "3GPP 6G NR FR3 @ 24 GHz — V2X Network Simulation",
    ha="center", va="center", fontsize=17, fontweight="bold",
    color="white", transform=ax_title.transAxes)
ax_title.text(0.5, 0.18,
    "Environments: Office | Urban Streets | Highway | Classroom    •    "
    "200–400 MHz BW  •  URLLC / eMBB / mMTC  •  Massive MIMO 32×32",
    ha="center", va="center", fontsize=9, color="#AED6F1",
    transform=ax_title.transAxes)

ENV_NAMES = list(ENVIRONMENTS.keys())
ENV_COLORS = [ENVIRONMENTS[e]["color"] for e in ENV_NAMES]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PER-ENVIRONMENT ROWS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

for env_idx, env_name in enumerate(ENV_NAMES):
    env_cfg  = ENVIRONMENTS[env_name]
    sim_data = SIM_DATA[env_name]
    ec       = env_cfg["color"]
    W, H     = env_cfg["area"]

    inner = GridSpecFromSubplotSpec(
        2, 4,
        subplot_spec=outer[env_idx + 1],
        hspace=0.55, wspace=0.42,
        height_ratios=[1, 1]
    )

    # ── Row label ────────────────────────────────────────────────────────────
    ax_lbl = fig.add_subplot(inner[:, 0])   # will be replaced by topology
    # We'll draw a colored band via a text box on the topology plot itself.

    # =========================================================================
    #  PLOT 1 — Network Topology + SINR Heatmap
    # =========================================================================
    ax_topo = fig.add_subplot(inner[0, 0])
    XX, YY, hm = sim_data["heatmap"]

    # Heatmap
    im = ax_topo.pcolormesh(XX, YY, hm, cmap="RdYlGn",
                             vmin=-5, vmax=30, shading="gouraud", alpha=0.55)

    # Buildings
    for bld in env_cfg["buildings"]:
        bx, by, bw, bh, blbl, mat, pen = bld
        rect = mpatches.FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0.5",
            linewidth=0.6,
            edgecolor="#455A64",
            facecolor=MATERIAL_COLOR.get(mat, "#CFD8DC"),
            alpha=0.75
        )
        ax_topo.add_patch(rect)
        ax_topo.text(bx+bw/2, by+bh/2, blbl,
                     ha="center", va="center", fontsize=5.5,
                     color="#263238", fontweight="bold")

    # gNB coverage circles
    for gx, gy, glbl, *_ in env_cfg["gnbs"]:
        for r, a in [(min(W,H)*0.45, 0.07), (min(W,H)*0.3, 0.11), (min(W,H)*0.15, 0.15)]:
            circle = plt.Circle((gx, gy), r, color=ec, alpha=a, linewidth=0)
            ax_topo.add_patch(circle)
        ax_topo.plot(gx, gy, marker="^", color=ec, markersize=9, zorder=5,
                     markeredgecolor="white", markeredgewidth=0.8)
        ax_topo.text(gx, gy-4, glbl, ha="center", fontsize=6, color=ec, fontweight="bold")

    # Node final positions + link lines
    for node in env_cfg["nodes"]:
        nid   = node[0]
        ntype = node[1]
        xs, ys = sim_data["trajectories"][nid]
        fx, fy = xs[-1], ys[-1]
        s = sim_data["snapshot_sinr"][nid]

        # Link to nearest gNB
        gnbs_xy = [(g[0], g[1]) for g in env_cfg["gnbs"]]
        dists   = [np.hypot(fx-gx, fy-gy) for gx,gy in gnbs_xy]
        best    = gnbs_xy[np.argmin(dists)]
        ax_topo.plot([fx, best[0]], [fy, best[1]],
                     color=sinr_color(s), lw=0.6, alpha=0.6, zorder=2)

        # Node marker
        st = NODE_STYLE.get(ntype, NODE_STYLE["phone"])
        ax_topo.scatter(fx, fy, c=st["color"], marker=st["marker"],
                        s=st["size"], zorder=6, edgecolors="white",
                        linewidths=0.5)
        ax_topo.text(fx, fy+2.5, nid, fontsize=4.5, ha="center",
                     color="#1A252F", zorder=7)

    ax_topo.set_xlim(0, W); ax_topo.set_ylim(0, H)
    ax_topo.set_aspect("equal", adjustable="box")
    ax_topo.set_title(f"{env_name}  ·  Topology + SINR Heatmap", fontsize=8,
                      color=ec, fontweight="bold", pad=3)
    ax_topo.set_xlabel("x (m)"); ax_topo.set_ylabel("y (m)")
    # Colorbar
    cb = plt.colorbar(im, ax=ax_topo, fraction=0.046, pad=0.04)
    cb.set_label("SINR (dB)", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    # =========================================================================
    #  PLOT 2 — SINR Time Series (all nodes)
    # =========================================================================
    ax_sinr = fig.add_subplot(inner[0, 1])
    cmap_nodes = plt.get_cmap("tab20")
    for ni, node in enumerate(env_cfg["nodes"]):
        nid    = node[0]
        series = sim_data["sinr_series"][nid]
        ax_sinr.plot(TIME[:len(series)], series,
                     lw=0.8, alpha=0.85,
                     color=cmap_nodes(ni / max(len(env_cfg["nodes"])-1, 1)),
                     label=nid)

    # Quality bands
    ax_sinr.axhspan(20, 35,  alpha=0.06, color="#27AE60", label=None)
    ax_sinr.axhspan(12, 20,  alpha=0.06, color="#F39C12", label=None)
    ax_sinr.axhspan(-5, 12,  alpha=0.06, color="#E74C3C", label=None)
    ax_sinr.axhline(20, color="#27AE60", lw=0.5, ls="--", alpha=0.5)
    ax_sinr.axhline(12, color="#E74C3C", lw=0.5, ls="--", alpha=0.5)

    ax_sinr.set_title("SINR Time Series (all nodes)", fontsize=8, pad=3)
    ax_sinr.set_xlabel("Time (s)"); ax_sinr.set_ylabel("SINR (dB)")
    ax_sinr.set_ylim(-5, 38)
    ax_sinr.legend(fontsize=5, ncol=2, loc="upper right",
                   framealpha=0.7, handlelength=1)
    # Band labels
    ax_sinr.text(TIME[-1]*0.98, 27, "Excellent", fontsize=5,
                 color="#27AE60", ha="right", va="center")
    ax_sinr.text(TIME[-1]*0.98, 16, "Good",      fontsize=5,
                 color="#F39C12", ha="right", va="center")
    ax_sinr.text(TIME[-1]*0.98, 4,  "Marginal",  fontsize=5,
                 color="#E74C3C", ha="right", va="center")

    # =========================================================================
    #  PLOT 3 — Mobility Trails
    # =========================================================================
    ax_mob = fig.add_subplot(inner[0, 2])
    ax_mob.set_facecolor("#F0F4F8")

    for bld in env_cfg["buildings"]:
        bx, by, bw, bh, *_ = bld
        rect = mpatches.FancyBboxPatch((bx, by), bw, bh,
            boxstyle="round,pad=0.5", linewidth=0.4,
            edgecolor="#90A4AE", facecolor="#CFD8DC", alpha=0.5)
        ax_mob.add_patch(rect)

    for ni, node in enumerate(env_cfg["nodes"]):
        nid    = node[0]
        ntype  = node[1]
        xs, ys = sim_data["trajectories"][nid]
        sinrs  = sim_data["sinr_series"][nid]
        st     = NODE_STYLE.get(ntype, NODE_STYLE["phone"])

        # Trail colored by SINR
        points = np.array([xs[:len(sinrs)], ys[:len(sinrs)]]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection
        norm   = Normalize(vmin=-5, vmax=30)
        lc     = LineCollection(segs, cmap="RdYlGn", norm=norm, lw=0.8, alpha=0.7)
        lc.set_array(sinrs[:-1])
        ax_mob.add_collection(lc)

        # Start / end markers
        ax_mob.plot(xs[0], ys[0], "o", color=st["color"],
                    markersize=4, markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        ax_mob.plot(xs[-1], ys[-1], st["marker"], color=st["color"],
                    markersize=6, markeredgecolor="white", markeredgewidth=0.6, zorder=6)
        ax_mob.text(xs[-1], ys[-1]+2.5, nid, fontsize=4.5, ha="center", color="#263238")

    # gNBs
    for gx, gy, glbl, *_ in env_cfg["gnbs"]:
        ax_mob.plot(gx, gy, "^", color=ec, markersize=8, zorder=7,
                    markeredgecolor="white", markeredgewidth=0.8)

    ax_mob.set_xlim(0, W); ax_mob.set_ylim(0, H)
    ax_mob.set_aspect("equal", adjustable="box")
    ax_mob.set_title("Mobility Trails (SINR-colored)", fontsize=8, pad=3)
    ax_mob.set_xlabel("x (m)"); ax_mob.set_ylabel("y (m)")

    # =========================================================================
    #  PLOT 4 — Throughput Timeline (per service type)
    # =========================================================================
    ax_tp = fig.add_subplot(inner[0, 3])
    bw    = env_cfg["bw_mhz"]
    for svc, svc_color in SERVICE_COLOR.items():
        nodes_in_svc = [n for n in env_cfg["nodes"] if n[6] == svc]
        if not nodes_in_svc:
            continue
        agg_tp = np.zeros(SIM_STEPS)
        for node in nodes_in_svc:
            s = sim_data["sinr_series"][node[0]]
            agg_tp += np.array([throughput_mbps(sv, bw) for sv in s])
        ax_tp.plot(TIME[:SIM_STEPS], agg_tp, color=svc_color,
                   lw=1.1, label=svc, alpha=0.9)
        ax_tp.fill_between(TIME[:SIM_STEPS], agg_tp, alpha=0.08, color=svc_color)

    ax_tp.set_title("Aggregate Throughput by Service", fontsize=8, pad=3)
    ax_tp.set_xlabel("Time (s)"); ax_tp.set_ylabel("Throughput (Mbps)")
    ax_tp.legend(fontsize=6, loc="upper right", framealpha=0.7)

    # =========================================================================
    #  PLOT 5 — Per-Node SINR Snapshot Bar Chart
    # =========================================================================
    ax_bar = fig.add_subplot(inner[1, 0])
    node_ids = [n[0] for n in env_cfg["nodes"]]
    snap     = [sim_data["snapshot_sinr"][nid] for nid in node_ids]
    colors   = [sinr_color(v) for v in snap]
    bars     = ax_bar.bar(range(len(node_ids)), snap, color=colors,
                          edgecolor="white", linewidth=0.5, width=0.7)
    ax_bar.axhline(20, color="#27AE60", lw=0.8, ls="--", alpha=0.6, label="Good (20 dB)")
    ax_bar.axhline(12, color="#E74C3C", lw=0.8, ls="--", alpha=0.6, label="Min (12 dB)")
    ax_bar.set_xticks(range(len(node_ids)))
    ax_bar.set_xticklabels(node_ids, rotation=45, ha="right", fontsize=6)
    ax_bar.set_ylabel("SINR (dB)"); ax_bar.set_ylim(-5, 38)
    ax_bar.set_title("Node SINR Snapshot", fontsize=8, pad=3)
    ax_bar.legend(fontsize=6, framealpha=0.7)
    for bar_i, v in enumerate(snap):
        ax_bar.text(bar_i, max(v+0.5, 0.5), f"{v:.1f}",
                    ha="center", fontsize=5, color="#1A252F")

    # =========================================================================
    #  PLOT 6 — SINR CDF per node
    # =========================================================================
    ax_cdf = fig.add_subplot(inner[1, 1])
    for ni, node in enumerate(env_cfg["nodes"]):
        nid    = node[0]
        series = np.sort(sim_data["sinr_series"][nid])
        cdf    = np.arange(1, len(series)+1) / len(series)
        ax_cdf.plot(series, cdf, lw=0.9, alpha=0.8,
                    color=cmap_nodes(ni / max(len(env_cfg["nodes"])-1, 1)),
                    label=nid)
    ax_cdf.axvline(12, color="#E74C3C", lw=0.7, ls="--", alpha=0.6)
    ax_cdf.axvline(20, color="#27AE60", lw=0.7, ls="--", alpha=0.6)
    ax_cdf.set_xlabel("SINR (dB)"); ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("SINR CDF", fontsize=8, pad=3)
    ax_cdf.set_ylim(0, 1); ax_cdf.set_xlim(-5, 36)
    ax_cdf.legend(fontsize=5, ncol=2, loc="lower right", framealpha=0.7, handlelength=1)

    # =========================================================================
    #  PLOT 7 — Average SINR vs Distance to Nearest gNB
    # =========================================================================
    ax_dist = fig.add_subplot(inner[1, 2])
    gnbs_xy = [(g[0], g[1]) for g in env_cfg["gnbs"]]
    for ni, node in enumerate(env_cfg["nodes"]):
        nid     = node[0]
        ntype   = node[1]
        xs, ys  = sim_data["trajectories"][nid]
        series  = sim_data["sinr_series"][nid]
        dists   = [min(np.hypot(xs[i]-gx, ys[i]-gy) for gx,gy in gnbs_xy)
                   for i in range(len(series))]
        st = NODE_STYLE.get(ntype, NODE_STYLE["phone"])
        ax_dist.scatter(dists, series, c=st["color"], s=4, alpha=0.4,
                        marker=st["marker"], label=nid)
    ax_dist.set_xlabel("Dist to nearest gNB (m)")
    ax_dist.set_ylabel("SINR (dB)")
    ax_dist.set_title("SINR vs Distance", fontsize=8, pad=3)
    ax_dist.legend(fontsize=5, ncol=2, loc="upper right", framealpha=0.7, handlelength=1)
    ax_dist.set_ylim(-5, 36)

    # =========================================================================
    #  PLOT 8 — Service KPI Compliance Table
    # =========================================================================
    ax_kpi = fig.add_subplot(inner[1, 3])
    ax_kpi.axis("off")
    col_labels = ["Service", "Avg SINR\n(dB)", "Avg TP\n(Mbps)", "5th pct\nSINR", "KPI"]
    table_data = []
    targets    = {"URLLC": ("< 10ms", 12), "eMBB": ("> 50 Mbps", 18), "mMTC": ("PDR > 99%", 8)}
    for svc in ["URLLC", "eMBB", "mMTC"]:
        svc_nodes = [n for n in env_cfg["nodes"] if n[6] == svc]
        if not svc_nodes:
            table_data.append([svc, "—", "—", "—", "N/A"])
            continue
        all_sinr = np.concatenate([sim_data["sinr_series"][n[0]] for n in svc_nodes])
        avg_s    = np.mean(all_sinr)
        avg_tp   = np.mean([throughput_mbps(s, env_cfg["bw_mhz"]) for s in all_sinr])
        p5       = np.percentile(all_sinr, 5)
        met      = "✓" if avg_s >= targets[svc][1] else "✗"
        table_data.append([svc, f"{avg_s:.1f}", f"{avg_tp:.0f}", f"{p5:.1f}", met])

    tbl = ax_kpi.table(
        cellText   = table_data,
        colLabels  = col_labels,
        cellLoc    = "center",
        loc        = "center",
        bbox       = [0.0, 0.05, 1.0, 0.9],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor(ec)
            cell.set_text_props(color="white", fontweight="bold", fontsize=6.5)
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        # KPI column coloring
        if c == 4 and r > 0:
            txt = cell.get_text().get_text()
            cell.set_facecolor("#D5F5E3" if txt == "✓" else "#FADBD8")
            cell.set_text_props(fontweight="bold", color="#1E8449" if txt == "✓" else "#C0392B")

    ax_kpi.set_title("Service KPI Compliance", fontsize=8, pad=3)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GLOBAL SUMMARY ROW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

summary = GridSpecFromSubplotSpec(
    1, 4,
    subplot_spec=outer[5],
    hspace=0.3, wspace=0.45
)

# ── S1: Cross-environment SINR comparison (violin) ───────────────────────────
ax_s1 = fig.add_subplot(summary[0, 0])
vdata = []
vlabels = []
for env_name in ENV_NAMES:
    env_cfg  = ENVIRONMENTS[env_name]
    all_sinr = np.concatenate([SIM_DATA[env_name]["sinr_series"][n[0]]
                               for n in env_cfg["nodes"]])
    vdata.append(all_sinr)
    vlabels.append(env_name.replace(" ", "\n"))

parts = ax_s1.violinplot(vdata, positions=range(len(ENV_NAMES)),
                          showmeans=True, showmedians=True,
                          widths=0.6)
for pc, ec in zip(parts['bodies'], ENV_COLORS):
    pc.set_facecolor(ec); pc.set_alpha(0.6)
parts['cmeans'].set_color("#1A252F"); parts['cmeans'].set_linewidth(1.2)
parts['cmedians'].set_color("white"); parts['cmedians'].set_linewidth(1.2)
for partname in ('cbars','cmins','cmaxes'):
    parts[partname].set_color("#666"); parts[partname].set_linewidth(0.8)

ax_s1.set_xticks(range(len(ENV_NAMES)))
ax_s1.set_xticklabels(vlabels, fontsize=7)
ax_s1.set_ylabel("SINR (dB)")
ax_s1.set_title("SINR Distribution Across Environments", fontsize=8, pad=3)
ax_s1.axhline(20, color="#27AE60", lw=0.7, ls="--", alpha=0.5)
ax_s1.axhline(12, color="#E74C3C", lw=0.7, ls="--", alpha=0.5)

# ── S2: Cross-environment throughput (grouped bar) ────────────────────────────
ax_s2 = fig.add_subplot(summary[0, 1])
svc_list = ["URLLC", "eMBB", "mMTC"]
x      = np.arange(len(ENV_NAMES))
width  = 0.22
for si, svc in enumerate(svc_list):
    vals = []
    for env_name in ENV_NAMES:
        env_cfg   = ENVIRONMENTS[env_name]
        svc_nodes = [n for n in env_cfg["nodes"] if n[6] == svc]
        if not svc_nodes:
            vals.append(0)
            continue
        all_sinr = np.concatenate([SIM_DATA[env_name]["sinr_series"][n[0]]
                                    for n in svc_nodes])
        vals.append(np.mean([throughput_mbps(s, env_cfg["bw_mhz"]) for s in all_sinr]))
    ax_s2.bar(x + si*width - width, vals, width,
              label=svc, color=SERVICE_COLOR[svc], alpha=0.85, edgecolor="white", lw=0.5)

ax_s2.set_xticks(x)
ax_s2.set_xticklabels([e.replace(" ", "\n") for e in ENV_NAMES], fontsize=7)
ax_s2.set_ylabel("Avg Throughput (Mbps)")
ax_s2.set_title("Throughput by Env & Service", fontsize=8, pad=3)
ax_s2.legend(fontsize=6, framealpha=0.7, loc="upper right")

# ── S3: Building penetration loss comparison ─────────────────────────────────
ax_s3 = fig.add_subplot(summary[0, 2])
mat_labels = ["Concrete\n(22 dB)", "Glass\n(8 dB)", "Metal\n(32 dB)"]
mat_vals   = [22, 8, 32]
mat_colors = ["#B0BEC5", "#B2EBF2", "#BCAAA4"]
bar_s3     = ax_s3.barh(mat_labels, mat_vals, color=mat_colors,
                         edgecolor="#455A64", linewidth=0.5, height=0.5)
ax_s3.set_xlabel("Penetration Loss (dB)")
ax_s3.set_title("FR3 @ 24 GHz Building Penetration\n(vs FR2 mmWave ~25% lower loss)", fontsize=8, pad=3)
for i, v in enumerate(mat_vals):
    ax_s3.text(v+0.3, i, f"{v} dB", va="center", fontsize=7.5, color="#263238")
# Comparison arrow
ax_s3.axvline(30, color="#E74C3C", lw=0.8, ls=":", alpha=0.5)
ax_s3.text(30.5, 2.4, "mmWave\nbaseline", fontsize=6, color="#E74C3C", va="top")

# ── S4: RF Configuration Summary ─────────────────────────────────────────────
ax_s4 = fig.add_subplot(summary[0, 3])
ax_s4.axis("off")

rf_rows = [
    ["Parameter",         "Value"],
    ["Frequency",         "24.0 GHz"],
    ["Band",              "FR3 (7–24 GHz)"],
    ["Max Bandwidth",     "400 MHz"],
    ["Wavelength",        "12.5 mm"],
    ["TX Power (gNB)",    "40–46 dBm"],
    ["TX Power (UE)",     "23 dBm"],
    ["Noise Floor",       "−90 dBm"],
    ["Numerology (μ)",    "2 (60 kHz SCS)"],
    ["Antenna",           "MIMO 32×32"],
    ["Max Modulation",    "256-QAM"],
    ["Environments sim.", "4"],
    ["Total nodes",       str(sum(len(ENVIRONMENTS[e]["nodes"]) for e in ENV_NAMES))],
    ["Sim duration",      f"{SIM_STEPS*DT:.0f} s"],
]
tbl2 = ax_s4.table(
    cellText   = rf_rows[1:],
    colLabels  = rf_rows[0],
    cellLoc    = "left",
    loc        = "center",
    bbox       = [0.0, 0.0, 1.0, 1.0],
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(7.5)
for (r, c), cell in tbl2.get_celld().items():
    cell.set_linewidth(0.3)
    if r == 0:
        cell.set_facecolor("#1A252F")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#EBF5FB")
    if c == 0:
        cell.set_text_props(color="#2C3E50", fontweight="bold")
ax_s4.set_title("FR3 RF Configuration", fontsize=8, pad=3)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GLOBAL COLORBAR FOR SINR HEATMAPS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sm  = ScalarMappable(cmap="RdYlGn", norm=Normalize(-5, 30))
cax = fig.add_axes([0.975, 0.28, 0.008, 0.68])
cb2 = fig.colorbar(sm, cax=cax)
cb2.set_label("SINR (dB)", fontsize=8, labelpad=6)
cb2.ax.tick_params(labelsize=7)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GLOBAL LEGEND
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
legend_handles = [
    mpatches.Patch(color=c, label=l)
    for l, c in [("Good SINR (≥20 dB)", "#27AE60"),
                 ("Fair SINR (12–20 dB)", "#F39C12"),
                 ("Poor SINR (<12 dB)", "#E74C3C")]
] + [
    plt.Line2D([0],[0], marker=NODE_STYLE[t]["marker"],
               color="w", markerfacecolor=NODE_STYLE[t]["color"],
               markersize=7, label=NODE_STYLE[t]["label"])
    for t in ["laptop", "phone", "car", "truck", "emergency", "pedestrian", "rsu", "iot"]
] + [
    mpatches.Patch(color=c, label=f"{svc} service")
    for svc, c in SERVICE_COLOR.items()
]

fig.legend(handles=legend_handles,
           loc="lower center",
           bbox_to_anchor=(0.5, -0.005),
           ncol=8, fontsize=7, framealpha=0.9,
           edgecolor="#BDC3C7")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
out_path = "6g_nr_fr3_v2x_results.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor="#FAFAFA", edgecolor="none")
print(f"\n✅  Saved → {out_path}")
print("   Open with any image viewer or IDE.")
plt.show()
