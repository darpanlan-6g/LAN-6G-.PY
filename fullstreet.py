"""
================================================================================
  3GPP 6G NR FR3 @ 24 GHz — INTERACTIVE LIVE SIMULATION
================================================================================
  Environments : Office | Urban Streets | Highway | Classroom
  Features     : Live node animation, real-time SINR heatmap refresh,
                 per-node SINR bars, throughput & handover counters,
                 pause/resume, speed control, environment switcher,
                 service filter (URLLC / eMBB / mMTC), CSV export
--------------------------------------------------------------------------------
  Run          : python 6g_nr_fr3_live_sim.py
  Dependencies : pip install matplotlib numpy scipy
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg is unavailable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, RadioButtons, Slider, CheckButtons
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import warnings, time, csv, os
warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COLOUR / STYLE CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG        = "#0D1117"
PANEL_BG  = "#161B22"
GRID_COL  = "#21262D"
TEXT_COL  = "#E6EDF3"
MUTED_COL = "#7D8590"
ACC_BLUE  = "#388BFD"
ACC_GREEN = "#3FB950"
ACC_RED   = "#F85149"
ACC_ORG   = "#D29922"
ACC_PRP   = "#BC8CFF"
ACC_TEAL  = "#39D353"

SERVICE_COLOR = {"URLLC": ACC_RED, "eMBB": ACC_BLUE, "mMTC": ACC_GREEN}

NODE_STYLE = {
    "laptop"    : {"color": ACC_GREEN, "marker": "s",  "size": 90,  "label": "Laptop"},
    "phone"     : {"color": ACC_PRP,   "marker": "o",  "size": 70,  "label": "Phone"},
    "car"       : {"color": ACC_BLUE,  "marker": "^",  "size": 100, "label": "Car"},
    "truck"     : {"color": ACC_ORG,   "marker": "D",  "size": 110, "label": "Truck"},
    "emergency" : {"color": ACC_RED,   "marker": "*",  "size": 160, "label": "Emergency"},
    "pedestrian": {"color": "#E91E8C", "marker": "P",  "size": 80,  "label": "Pedestrian"},
    "rsu"       : {"color": "#F39C12", "marker": "H",  "size": 90,  "label": "RSU"},
    "iot"       : {"color": ACC_TEAL,  "marker": "+",  "size": 70,  "label": "IoT"},
}

MAT_COLOR = {"concrete": "#30363D", "glass": "#1C2A38", "metal": "#2D2016"}
MAT_EDGE  = {"concrete": "#484F58", "glass": "#388BFD", "metal": "#D29922"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENVIRONMENT DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENVIRONMENTS = {
    "Office": {
        "area": (100, 80), "freq_ghz": 24.0, "bw_mhz": 200,
        "tx_power_dbm": 23, "pen_loss_avg": 20, "color": "#388BFD",
        "buildings": [
            (3,  3, 35, 30, "Wing A",      "concrete", 22),
            (45, 3, 35, 30, "Wing B",      "glass",     8),
            (3,  45,45, 28, "Open Plan",   "glass",     8),
            (55, 45,38, 28, "Server Room", "metal",    32),
        ],
        "gnbs": [(50, 40, "AP-0", 3)],
        "nodes": [
            # (id, type, x0, y0, vx, vy, service, bounce)
            ("L1",   "laptop",      12, 18,  0.00,  0.00, "eMBB",  False),
            ("L2",   "laptop",      60, 18,  0.00,  0.00, "eMBB",  False),
            ("P1",   "phone",       20, 55,  0.35,  0.12, "URLLC", True),
            ("P2",   "phone",       70, 55, -0.28,  0.18, "URLLC", True),
            ("P3",   "phone",       50, 20,  0.15, -0.12, "eMBB",  True),
            ("IoT1", "iot",         30, 60,  0.00,  0.00, "mMTC",  False),
            ("IoT2", "iot",         75, 60,  0.00,  0.00, "mMTC",  False),
            ("IoT3", "iot",         15, 35,  0.00,  0.00, "mMTC",  False),
        ],
    },
    "Urban Streets": {
        "area": (120, 100), "freq_ghz": 24.0, "bw_mhz": 200,
        "tx_power_dbm": 40, "pen_loss_avg": 18, "color": "#D29922",
        "buildings": [
            (0,  0,  28, 34, "Office Block","concrete", 22),
            (38, 0,  34, 24, "Apartments",  "concrete", 22),
            (88, 0,  32, 40, "Hotel",       "glass",     8),
            (0,  66, 30, 34, "Warehouse",   "metal",    32),
            (85, 62, 35, 38, "Residential", "concrete", 22),
            (40, 72, 28, 28, "Parking",     "concrete", 22),
        ],
        "gnbs": [(35, 50, "gNB-0", 15), (85, 50, "gNB-1", 15)],
        "nodes": [
            ("Car1",  "car",        5,  50,  2.2,  0.0,  "URLLC", False),
            ("Car2",  "car",       55,  46,  1.9,  0.0,  "URLLC", False),
            ("Car3",  "car",      115,  53, -1.6,  0.0,  "URLLC", False),
            ("Truck", "truck",     30,  56,  1.3,  0.0,  "eMBB",  False),
            ("Ped1",  "pedestrian",45,  48,  0.22, 0.12, "eMBB",  True),
            ("Ped2",  "pedestrian",72,  54, -0.12, 0.20, "eMBB",  True),
            ("RSU0",  "rsu",       35,  50,  0.0,  0.0,  "URLLC", False),
            ("RSU1",  "rsu",       85,  50,  0.0,  0.0,  "URLLC", False),
        ],
    },
    "Highway": {
        "area": (200, 60), "freq_ghz": 24.0, "bw_mhz": 400,
        "tx_power_dbm": 46, "pen_loss_avg": 8,  "color": "#3FB950",
        "buildings": [
            (0,   0, 200, 14, "Sound Barrier N", "concrete", 22),
            (0,  46, 200, 14, "Sound Barrier S", "concrete", 22),
            (38, 14,  20, 32, "Tunnel W",        "metal",    32),
            (142,14,  20, 32, "Tunnel E",        "metal",    32),
        ],
        "gnbs": [(60, 7, "gNB-A", 20), (140, 7, "gNB-B", 20)],
        "nodes": [
            ("V1", "car",        8,  25,  3.4, 0.0, "URLLC", False),
            ("V2", "car",       45,  28,  3.0, 0.0, "URLLC", False),
            ("V3", "car",       88,  26,  3.2, 0.0, "URLLC", False),
            ("V4", "car",      125,  27,  2.8, 0.0, "URLLC", False),
            ("V5", "car",      172,  25,  3.1, 0.0, "URLLC", False),
            ("T1", "truck",     22,  33,  2.2, 0.0, "eMBB",  False),
            ("T2", "truck",    102,  35,  2.0, 0.0, "eMBB",  False),
            ("EV", "emergency",162,  26,  4.8, 0.0, "URLLC", False),
        ],
    },
    "Classroom": {
        "area": (90, 70), "freq_ghz": 24.0, "bw_mhz": 200,
        "tx_power_dbm": 23, "pen_loss_avg": 20, "color": "#BC8CFF",
        "buildings": [
            (4,  4, 36, 28, "Room 101",  "concrete", 22),
            (50, 4, 36, 28, "Room 102",  "concrete", 22),
            (4, 40, 36, 26, "Auditorium","glass",      8),
            (50,40, 36, 26, "Library",   "glass",      8),
        ],
        "gnbs": [(45, 35, "AP-0", 3)],
        "nodes": [
            ("S1",  "phone",  12, 16,  0.06,  0.10, "eMBB",  True),
            ("S2",  "phone",  22, 22, -0.05,  0.06, "eMBB",  True),
            ("S3",  "phone",  30, 14,  0.08, -0.05, "eMBB",  True),
            ("S4",  "phone",  58, 16, -0.06,  0.08, "eMBB",  True),
            ("S5",  "phone",  72, 20,  0.05,  0.06, "eMBB",  True),
            ("S6",  "phone",  65, 10, -0.04, -0.04, "eMBB",  True),
            ("T1",  "laptop", 20, 52,  0.00,  0.00, "eMBB",  False),
            ("T2",  "laptop", 65, 52,  0.00,  0.00, "eMBB",  False),
            ("Proj","iot",    45, 35,  0.00,  0.00, "URLLC", False),
        ],
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PHYSICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def friis_pl(d_m, freq_ghz):
    return 20*np.log10(max(d_m, 0.5)) + 20*np.log10(freq_ghz*1e9) - 147.55

def compute_sinr(nx, ny, gnbs, cfg, rng):
    freq, tx, pen = cfg["freq_ghz"], cfg["tx_power_dbm"], cfg["pen_loss_avg"]
    rxs = []
    for gx, gy, *_ in gnbs:
        d  = np.hypot(nx-gx, ny-gy)
        pl = friis_pl(d, freq)
        rxs.append(tx - pl - pen*0.25 + rng.normal(0, 1.5))
    rxs.sort(reverse=True)
    sig  = 10**(rxs[0]/10)
    intf = sum(10**(p/10) for p in rxs[1:]) if len(rxs)>1 else 0
    nois = 10**(-90/10)
    return 10*np.log10(max(sig/(intf+nois), 1e-9))

def shannon_tp(sinr_db, bw_mhz):
    return bw_mhz * np.log2(1 + 10**(sinr_db/10)) * 0.6

def sinr_color_map(v):
    if v >= 20: return ACC_GREEN
    if v >= 12: return ACC_ORG
    if v >= 4:  return "#E67E22"
    return ACC_RED

def build_heatmap(cfg, res=40):
    W, H = cfg["area"]
    xs = np.linspace(0, W, res)
    ys = np.linspace(0, H, res)
    XX, YY = np.meshgrid(xs, ys)
    rng = np.random.default_rng(0)
    G = np.array([[compute_sinr(XX[i,j], YY[i,j], cfg["gnbs"], cfg, rng)
                   for j in range(res)] for i in range(res)])
    return XX, YY, gaussian_filter(G, sigma=1.2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIMULATION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimState:
    def __init__(self):
        self.env_name    = "Office"
        self.paused      = False
        self.speed       = 1.0
        self.t           = 0.0
        self.frame       = 0
        self.show_hm     = True
        self.show_trails = True
        self.show_links  = True
        self.svc_filter  = {"URLLC": True, "eMBB": True, "mMTC": True}
        self.heatmaps    = {}
        self.reset()

    @property
    def cfg(self):
        return ENVIRONMENTS[self.env_name]

    def reset(self):
        cfg = self.cfg
        self.rng         = np.random.default_rng(42)
        self.positions   = {n[0]: [n[2], n[3]] for n in cfg["nodes"]}
        self.velocities  = {n[0]: [n[4]*self.speed, n[5]*self.speed] for n in cfg["nodes"]}
        self.bounce      = {n[0]: n[7] for n in cfg["nodes"]}
        self.sinr_hist   = {n[0]: [] for n in cfg["nodes"]}
        self.tp_hist     = {n[0]: [] for n in cfg["nodes"]}
        self.trail_x     = {n[0]: [] for n in cfg["nodes"]}
        self.trail_y     = {n[0]: [] for n in cfg["nodes"]}
        self.prev_cell   = {n[0]: None for n in cfg["nodes"]}
        self.handovers   = 0
        self.total_tp    = []
        self.t           = 0.0
        self.frame       = 0
        # Pre-build heatmap if not cached
        if self.env_name not in self.heatmaps:
            print(f"  Building heatmap for {self.env_name}…", end="", flush=True)
            self.heatmaps[self.env_name] = build_heatmap(cfg)
            print(" done")

    def step(self, dt=0.05):
        if self.paused:
            return
        cfg = self.cfg
        W, H = cfg["area"]
        self.t     += dt * self.speed
        self.frame += 1

        total_tp_frame = 0.0
        for node in cfg["nodes"]:
            nid = node[0]
            x, y   = self.positions[nid]
            vx, vy = self.velocities[nid]

            # Move
            x += vx * dt
            y += vy * dt

            # Boundary handling
            if self.bounce[nid]:
                if x <= 0 or x >= W: vx *= -1; x = np.clip(x, 0.5, W-0.5)
                if y <= 0 or y >= H: vy *= -1; y = np.clip(y, 0.5, H-0.5)
            else:
                # Highway wrap-around for fast vehicles
                if vx > 0 and x > W+5:  x = -5
                if vx < 0 and x < -5:   x = W+5
                y = np.clip(y, 0.5, H-0.5)

            self.positions[nid]  = [x, y]
            self.velocities[nid] = [vx, vy]

            # SINR
            sv = compute_sinr(x, y, cfg["gnbs"], cfg, self.rng)
            tp = shannon_tp(sv, cfg["bw_mhz"])
            self.sinr_hist[nid].append(sv)
            self.tp_hist[nid].append(tp)
            if len(self.sinr_hist[nid]) > 300:
                self.sinr_hist[nid].pop(0)
                self.tp_hist[nid].pop(0)
            total_tp_frame += tp

            # Trail
            self.trail_x[nid].append(x)
            self.trail_y[nid].append(y)
            if len(self.trail_x[nid]) > 80:
                self.trail_x[nid].pop(0)
                self.trail_y[nid].pop(0)

            # Handover detection
            gnbs   = cfg["gnbs"]
            dists  = [np.hypot(x-g[0], y-g[1]) for g in gnbs]
            cell   = int(np.argmin(dists))
            if self.prev_cell[nid] is not None and self.prev_cell[nid] != cell:
                self.handovers += 1
            self.prev_cell[nid] = cell

        self.total_tp.append(total_tp_frame)
        if len(self.total_tp) > 300:
            self.total_tp.pop(0)

    def export_csv(self):
        fname = f"sinr_export_{self.env_name.replace(' ','_')}_{int(time.time())}.csv"
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Node","Type","Service","SINR_dB","Throughput_Mbps","X","Y"])
            cfg = self.cfg
            for node in cfg["nodes"]:
                nid   = node[0]
                ntype = node[1]
                svc   = node[6]
                x, y  = self.positions[nid]
                sv    = self.sinr_hist[nid][-1] if self.sinr_hist[nid] else 0
                tp    = self.tp_hist[nid][-1]   if self.tp_hist[nid]   else 0
                w.writerow([nid, ntype, svc, f"{sv:.2f}", f"{tp:.1f}", f"{x:.1f}", f"{y:.1f}"])
        print(f"\n✅  Exported → {fname}")
        return fname

SIM = SimState()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.facecolor": PANEL_BG, "figure.facecolor": BG,
    "axes.edgecolor": GRID_COL, "axes.labelcolor": TEXT_COL,
    "xtick.color": MUTED_COL,  "ytick.color": MUTED_COL,
    "text.color": TEXT_COL,    "axes.titlecolor": TEXT_COL,
    "axes.grid": True,         "grid.color": GRID_COL,
    "grid.linewidth": 0.4,     "axes.spines.top": False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(22, 13), facecolor=BG)
fig.canvas.manager.set_window_title("6G NR FR3 @ 24 GHz — Live V2X Simulation")

# ── Main grid ────────────────────────────────────────────────────────────────
gs_main = gridspec.GridSpec(
    3, 4,
    figure=fig,
    left=0.01, right=0.99,
    top=0.91,  bottom=0.28,
    hspace=0.52, wspace=0.38,
)

# Row 0: topology (tall, spans 2 rows in col 0), SINR bars, TP timeline, stats
ax_topo = fig.add_subplot(gs_main[0:2, 0])   # topology + heatmap
ax_bars = fig.add_subplot(gs_main[0,   1])   # live SINR bars
ax_tp   = fig.add_subplot(gs_main[0,   2])   # throughput timeline
ax_stat = fig.add_subplot(gs_main[0,   3])   # KPI stats table

# Row 1: SINR time series (col 1-2), scatter SINR vs dist (col 3)
ax_sinr = fig.add_subplot(gs_main[1,   1])   # SINR time series
ax_dist = fig.add_subplot(gs_main[1,   2])   # SINR vs distance
ax_cdf  = fig.add_subplot(gs_main[1,   3])   # CDF

# Row 2: full-width aggregate throughput
ax_agg = fig.add_subplot(gs_main[2, :])      # aggregate total TP

# ── Control panel (bottom) ───────────────────────────────────────────────────
gs_ctrl = gridspec.GridSpec(
    1, 7,
    figure=fig,
    left=0.01, right=0.99,
    bottom=0.01, top=0.26,
    wspace=0.35,
)

ax_env_radio  = fig.add_subplot(gs_ctrl[0, 0])
ax_svc_chk    = fig.add_subplot(gs_ctrl[0, 1])
ax_opts_chk   = fig.add_subplot(gs_ctrl[0, 2])
ax_speed_sl   = fig.add_subplot(gs_ctrl[0, 3])
ax_btn_area   = fig.add_subplot(gs_ctrl[0, 4])
ax_legend     = fig.add_subplot(gs_ctrl[0, 5])
ax_info       = fig.add_subplot(gs_ctrl[0, 6])

for ax in [ax_btn_area, ax_legend, ax_info, ax_stat, ax_cdf]:
    ax.axis("off")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TITLE BAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig.text(0.5, 0.965,
         "3GPP 6G NR FR3 @ 24 GHz  ·  Interactive Live V2X Simulation",
         ha="center", fontsize=14, fontweight="bold", color=TEXT_COL)
fig.text(0.5, 0.948,
         "Office  |  Urban Streets  |  Highway  |  Classroom     "
         "·     200–400 MHz BW  ·  Massive MIMO 32×32  ·  URLLC / eMBB / mMTC",
         ha="center", fontsize=8.5, color=MUTED_COL)

# Time / status label
txt_status = fig.text(0.98, 0.965, "t = 0.00 s  |  ▶ RUNNING",
                      ha="right", fontsize=9, color=ACC_GREEN, fontweight="bold")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WIDGET: Environment selector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_env_radio.set_title("Environment", fontsize=8, color=TEXT_COL, pad=2)
radio_env = RadioButtons(
    ax_env_radio,
    list(ENVIRONMENTS.keys()),
    activecolor=ACC_BLUE,
)
for lbl in radio_env.labels:
    lbl.set_color(TEXT_COL); lbl.set_fontsize(8)

def on_env_change(label):
    SIM.env_name = label
    SIM.reset()
    print(f"  Switched → {label}")
radio_env.on_clicked(on_env_change)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WIDGET: Service filter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_svc_chk.set_title("Service Filter", fontsize=8, color=TEXT_COL, pad=2)
chk_svc = CheckButtons(
    ax_svc_chk, ["URLLC", "eMBB", "mMTC"], [True, True, True]
)
for lbl in chk_svc.labels:
    lbl.set_color(TEXT_COL); lbl.set_fontsize(8)

def on_svc_toggle(label):
    SIM.svc_filter[label] = not SIM.svc_filter[label]
chk_svc.on_clicked(on_svc_toggle)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WIDGET: Display options
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_opts_chk.set_title("Display", fontsize=8, color=TEXT_COL, pad=2)
chk_opts = CheckButtons(
    ax_opts_chk,
    ["Heatmap", "Trails", "Links"],
    [True, True, True]
)
for lbl in chk_opts.labels:
    lbl.set_color(TEXT_COL); lbl.set_fontsize(8)

def on_opts_toggle(label):
    if label == "Heatmap": SIM.show_hm     = not SIM.show_hm
    if label == "Trails":  SIM.show_trails = not SIM.show_trails
    if label == "Links":   SIM.show_links  = not SIM.show_links
chk_opts.on_clicked(on_opts_toggle)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WIDGET: Speed slider
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_speed_sl.set_title("Sim Speed", fontsize=8, color=TEXT_COL, pad=2)
slider_speed = Slider(
    ax_speed_sl, "", 0.1, 5.0, valinit=1.0,
    color=ACC_BLUE, track_color=GRID_COL,
)
slider_speed.label.set_color(TEXT_COL)
slider_speed.valtext.set_color(ACC_BLUE)

def on_speed(val):
    SIM.speed = val
slider_speed.on_changed(on_speed)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WIDGET: Buttons
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_btn_area.set_facecolor(PANEL_BG)
btn_ax_pause  = fig.add_axes([0.595, 0.17, 0.075, 0.035])
btn_ax_reset  = fig.add_axes([0.595, 0.12, 0.075, 0.035])
btn_ax_export = fig.add_axes([0.595, 0.07, 0.075, 0.035])

btn_pause  = Button(btn_ax_pause,  "⏸  Pause",  color=PANEL_BG, hovercolor=GRID_COL)
btn_reset  = Button(btn_ax_reset,  "↺  Reset",  color=PANEL_BG, hovercolor=GRID_COL)
btn_export = Button(btn_ax_export, "⬇  Export CSV", color=PANEL_BG, hovercolor=GRID_COL)
for btn in [btn_pause, btn_reset, btn_export]:
    btn.label.set_color(TEXT_COL); btn.label.set_fontsize(8)

def on_pause(ev):
    SIM.paused = not SIM.paused
    btn_pause.label.set_text("▶  Resume" if SIM.paused else "⏸  Pause")
    txt_status.set_text(
        f"t = {SIM.t:.1f} s  |  {'⏸ PAUSED' if SIM.paused else '▶ RUNNING'}"
    )
    txt_status.set_color(ACC_ORG if SIM.paused else ACC_GREEN)
btn_pause.on_clicked(on_pause)

def on_reset(ev):
    SIM.reset()
    print(f"  Reset: {SIM.env_name}")
btn_reset.on_clicked(on_reset)

def on_export(ev):
    SIM.export_csv()
btn_export.on_clicked(on_export)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WIDGET: Legend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_legend.set_facecolor(PANEL_BG)
ax_legend.set_title("Node Types", fontsize=8, color=TEXT_COL, pad=2)
yl = 0.95
for ntype, st in NODE_STYLE.items():
    ax_legend.plot(0.08, yl, marker=st["marker"], color=st["color"],
                   markersize=7, transform=ax_legend.transAxes, clip_on=False)
    ax_legend.text(0.18, yl, st["label"], fontsize=7.5, color=TEXT_COL,
                   va="center", transform=ax_legend.transAxes)
    yl -= 0.115
# SINR quality bands
ax_legend.text(0.08, 0.15, "─── SINR quality:", fontsize=7, color=MUTED_COL,
               transform=ax_legend.transAxes)
for col, lbl, yy in [(ACC_GREEN,"≥20 dB  Excellent",0.09),
                     (ACC_ORG,  "12–20   Good",      0.04),
                     (ACC_RED,  "<12 dB  Marginal",  -0.01)]:
    ax_legend.plot(0.08, yy, "s", color=col, markersize=6,
                   transform=ax_legend.transAxes, clip_on=False)
    ax_legend.text(0.18, yy, lbl, fontsize=7, color=col,
                   va="center", transform=ax_legend.transAxes)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INFO PANEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_info.set_facecolor(PANEL_BG)
ax_info.set_title("RF Parameters", fontsize=8, color=TEXT_COL, pad=2)
info_lines = [
    ("Frequency",  "24.0 GHz"),
    ("Band",       "FR3 (7–24 GHz)"),
    ("BW",         "200–400 MHz"),
    ("Wavelength", "12.5 mm"),
    ("Numerology", "μ=2  60 kHz SCS"),
    ("TX (gNB)",   "40–46 dBm"),
    ("TX (UE)",    "23 dBm"),
    ("Noise",      "−90 dBm"),
    ("Antenna",    "32×32 MIMO"),
    ("Mod",        "256-QAM"),
]
for i, (k, v) in enumerate(info_lines):
    yp = 0.95 - i*0.094
    ax_info.text(0.02, yp, k+":", fontsize=7.5, color=MUTED_COL,
                 transform=ax_info.transAxes, va="center")
    ax_info.text(0.48, yp, v,   fontsize=7.5, color=TEXT_COL,
                 transform=ax_info.transAxes, va="center", fontweight="bold")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ANIMATION UPDATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cmap_node_hist = plt.get_cmap("tab20")
norm_sinr      = Normalize(vmin=-5, vmax=30)

def draw_frame(_frame_num):
    SIM.step(dt=0.05)
    cfg  = SIM.cfg
    W, H = cfg["area"]
    nodes_def = cfg["nodes"]

    # ── Topology + Heatmap ───────────────────────────────────────────────────
    ax_topo.cla()
    ax_topo.set_facecolor(PANEL_BG)
    ax_topo.set_xlim(0, W); ax_topo.set_ylim(0, H)
    ax_topo.set_aspect("equal", adjustable="box")
    ax_topo.set_title(f"Network Topology  ·  {SIM.env_name}", fontsize=9,
                      color=cfg["color"], fontweight="bold", pad=3)
    ax_topo.set_xlabel("x (m)", color=MUTED_COL)
    ax_topo.set_ylabel("y (m)", color=MUTED_COL)
    ax_topo.grid(True, color=GRID_COL, lw=0.3)

    # Heatmap
    if SIM.show_hm and SIM.env_name in SIM.heatmaps:
        XX, YY, hm = SIM.heatmaps[SIM.env_name]
        ax_topo.pcolormesh(XX, YY, hm, cmap="RdYlGn",
                           vmin=-5, vmax=30, shading="gouraud", alpha=0.40)

    # Buildings
    for bld in cfg["buildings"]:
        bx, by, bw, bh, blbl, mat, pen = bld
        rect = mpatches.FancyBboxPatch(
            (bx, by), bw, bh, boxstyle="round,pad=0.4",
            lw=0.8, edgecolor=MAT_EDGE.get(mat,"#484F58"),
            facecolor=MAT_COLOR.get(mat, "#30363D"), alpha=0.85
        )
        ax_topo.add_patch(rect)
        ax_topo.text(bx+bw/2, by+bh/2, blbl,
                     ha="center", va="center", fontsize=5.5,
                     color=MUTED_COL)

    # gNB pulse
    for gx, gy, glbl, *_ in cfg["gnbs"]:
        for r, a in [(min(W,H)*0.42, 0.05), (min(W,H)*0.28, 0.08), (min(W,H)*0.14, 0.12)]:
            circle = plt.Circle((gx, gy), r, color=cfg["color"], alpha=a, lw=0)
            ax_topo.add_patch(circle)
        # Pulse ring animation
        pr = (SIM.frame * 0.6 % min(W,H)*0.4) + 5
        ax_topo.add_patch(plt.Circle((gx,gy), pr, color=cfg["color"],
                                     alpha=max(0, 0.3-pr/(min(W,H)*0.5)),
                                     fill=False, lw=1.0))
        ax_topo.plot(gx, gy, "^", color=cfg["color"], markersize=11, zorder=8,
                     markeredgecolor="white", markeredgewidth=0.8)
        ax_topo.text(gx, gy-4.5, glbl, ha="center", fontsize=6.5,
                     color=cfg["color"], fontweight="bold")

    # Nodes
    gnbs_xy = [(g[0], g[1]) for g in cfg["gnbs"]]
    for node in nodes_def:
        nid, ntype, _, _, _, _, svc, _ = node
        if not SIM.svc_filter.get(svc, True):
            continue
        x, y = SIM.positions[nid]
        hist = SIM.sinr_hist[nid]
        sv   = hist[-1] if hist else 15.0
        st   = NODE_STYLE.get(ntype, NODE_STYLE["phone"])

        # Trail
        if SIM.show_trails and len(SIM.trail_x[nid]) > 2:
            tx_ = SIM.trail_x[nid]
            ty_ = SIM.trail_y[nid]
            pts = np.array([tx_, ty_]).T.reshape(-1,1,2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap="RdYlGn", norm=norm_sinr,
                                lw=1.0, alpha=0.55)
            lc.set_array(np.array(SIM.sinr_hist[nid][-len(segs):]))
            ax_topo.add_collection(lc)

        # Link line
        if SIM.show_links:
            dists  = [np.hypot(x-gx, y-gy) for gx,gy in gnbs_xy]
            best   = gnbs_xy[int(np.argmin(dists))]
            ax_topo.plot([x, best[0]], [y, best[1]],
                         color=sinr_color_map(sv), lw=0.6, alpha=0.5, zorder=2)

        # Node marker
        ax_topo.scatter(x, y, c=st["color"], marker=st["marker"],
                        s=st["size"], zorder=7,
                        edgecolors="white", linewidths=0.7)
        # SINR dot (top-right)
        ax_topo.scatter(x+W*0.012, y+H*0.015, c=sinr_color_map(sv),
                        s=18, zorder=9, edgecolors="none")
        ax_topo.text(x, y+H*0.035, nid, fontsize=5, ha="center",
                     color=TEXT_COL, zorder=10)

    # ── Live SINR Bars ───────────────────────────────────────────────────────
    ax_bars.cla()
    ax_bars.set_facecolor(PANEL_BG)
    ax_bars.set_title("Live SINR per Node (dB)", fontsize=8, pad=3)
    ax_bars.set_xlim(-5, 35)
    active_nodes = [n for n in nodes_def if SIM.svc_filter.get(n[6], True)]
    ypos = np.arange(len(active_nodes))
    for i, node in enumerate(active_nodes):
        nid = node[0]; svc = node[6]
        hist = SIM.sinr_hist[nid]
        sv   = hist[-1] if hist else 0
        bar_color = sinr_color_map(sv)
        ax_bars.barh(i, sv, color=bar_color, height=0.6,
                     edgecolor="none", alpha=0.85)
        ax_bars.text(max(sv+0.3, 0.5), i, f"{sv:.1f}",
                     va="center", fontsize=6.5, color=TEXT_COL)
        ax_bars.text(-4.5, i, node[0], va="center", fontsize=6.5,
                     color=SERVICE_COLOR.get(svc, MUTED_COL), fontweight="bold")
    ax_bars.set_yticks([])
    ax_bars.axvline(20, color=ACC_GREEN, lw=0.7, ls="--", alpha=0.5)
    ax_bars.axvline(12, color=ACC_RED,   lw=0.7, ls="--", alpha=0.5)
    ax_bars.set_xlabel("SINR (dB)", color=MUTED_COL)
    ax_bars.grid(axis="x", color=GRID_COL, lw=0.3)
    ax_bars.text(20.3, len(active_nodes)-0.3, "20", fontsize=6,
                 color=ACC_GREEN, alpha=0.7)
    ax_bars.text(12.3, len(active_nodes)-0.3, "12", fontsize=6,
                 color=ACC_RED, alpha=0.7)

    # ── Throughput Timeline ──────────────────────────────────────────────────
    ax_tp.cla()
    ax_tp.set_facecolor(PANEL_BG)
    ax_tp.set_title("Per-Node Throughput (Mbps)", fontsize=8, pad=3)
    ax_tp.grid(color=GRID_COL, lw=0.3)
    cmap_tab = plt.get_cmap("tab20")
    for ni, node in enumerate(active_nodes):
        nid = node[0]
        tp_h = SIM.tp_hist[nid]
        if len(tp_h) < 2: continue
        t_arr = np.arange(len(tp_h)) * 0.05
        ax_tp.plot(t_arr, tp_h,
                   color=cmap_tab(ni / max(len(active_nodes)-1, 1)),
                   lw=0.9, alpha=0.85, label=nid)
    ax_tp.set_xlabel("Time (s)", color=MUTED_COL)
    ax_tp.set_ylabel("Mbps", color=MUTED_COL)
    ax_tp.legend(fontsize=5.5, ncol=2, loc="upper left",
                 framealpha=0.5, labelcolor=TEXT_COL,
                 facecolor=PANEL_BG, edgecolor=GRID_COL)

    # ── KPI Stats ────────────────────────────────────────────────────────────
    ax_stat.cla(); ax_stat.axis("off")
    ax_stat.set_facecolor(PANEL_BG)
    ax_stat.set_title("Live KPIs", fontsize=8, pad=3, color=TEXT_COL)

    kpi_items = []
    for svc, sc in SERVICE_COLOR.items():
        svc_ns = [n for n in nodes_def if n[6]==svc and SIM.svc_filter.get(svc,True)]
        if not svc_ns: continue
        vals = [SIM.sinr_hist[n[0]][-1] for n in svc_ns if SIM.sinr_hist[n[0]]]
        if not vals: continue
        avg_s  = np.mean(vals)
        avg_tp = np.mean([shannon_tp(v, cfg["bw_mhz"]) for v in vals])
        kpi_items.append((svc, avg_s, avg_tp, sc))

    y0 = 0.93
    for svc, avg_s, avg_tp, sc in kpi_items:
        ax_stat.text(0.02, y0, svc, fontsize=8, color=sc, fontweight="bold",
                     transform=ax_stat.transAxes)
        ax_stat.text(0.35, y0, f"SINR {avg_s:.1f} dB", fontsize=7.5,
                     color=sinr_color_map(avg_s), transform=ax_stat.transAxes)
        ax_stat.text(0.70, y0, f"{avg_tp:.0f} Mbps", fontsize=7.5,
                     color=TEXT_COL, transform=ax_stat.transAxes)
        y0 -= 0.18

    ax_stat.plot([0, 1], [y0+0.05, y0+0.05], color=GRID_COL, lw=0.4,
                 transform=ax_stat.transAxes, clip_on=False)

    # Handover + time
    ax_stat.text(0.02, y0-0.05,
                 f"Handovers: {SIM.handovers}",
                 fontsize=8, color=ACC_ORG, transform=ax_stat.transAxes)
    ax_stat.text(0.02, y0-0.18,
                 f"t = {SIM.t:.1f} s",
                 fontsize=8, color=MUTED_COL, transform=ax_stat.transAxes)
    ax_stat.text(0.02, y0-0.31,
                 f"Env: {SIM.env_name}",
                 fontsize=8, color=cfg["color"], fontweight="bold",
                 transform=ax_stat.transAxes)

    # ── SINR Time Series ─────────────────────────────────────────────────────
    ax_sinr.cla()
    ax_sinr.set_facecolor(PANEL_BG)
    ax_sinr.set_title("SINR Time Series (all nodes)", fontsize=8, pad=3)
    ax_sinr.grid(color=GRID_COL, lw=0.3)
    ax_sinr.axhspan(20, 35,  alpha=0.06, color=ACC_GREEN)
    ax_sinr.axhspan(12, 20,  alpha=0.06, color=ACC_ORG)
    ax_sinr.axhspan(-5, 12,  alpha=0.06, color=ACC_RED)
    for ni, node in enumerate(active_nodes):
        nid    = node[0]
        series = SIM.sinr_hist[nid]
        if len(series) < 2: continue
        t_arr = np.arange(len(series)) * 0.05
        ax_sinr.plot(t_arr, series,
                     color=cmap_tab(ni / max(len(active_nodes)-1, 1)),
                     lw=0.85, alpha=0.85, label=nid)
    ax_sinr.axhline(20, color=ACC_GREEN, lw=0.6, ls="--", alpha=0.4)
    ax_sinr.axhline(12, color=ACC_RED,   lw=0.6, ls="--", alpha=0.4)
    ax_sinr.set_ylim(-5, 38)
    ax_sinr.set_xlabel("Time (s)", color=MUTED_COL)
    ax_sinr.set_ylabel("SINR (dB)", color=MUTED_COL)
    ax_sinr.legend(fontsize=5.5, ncol=3, loc="upper right",
                   framealpha=0.4, labelcolor=TEXT_COL,
                   facecolor=PANEL_BG, edgecolor=GRID_COL)

    # ── SINR vs Distance ─────────────────────────────────────────────────────
    ax_dist.cla()
    ax_dist.set_facecolor(PANEL_BG)
    ax_dist.set_title("SINR vs Distance to gNB", fontsize=8, pad=3)
    ax_dist.grid(color=GRID_COL, lw=0.3)
    for ni, node in enumerate(active_nodes):
        nid, ntype = node[0], node[1]
        hist = SIM.sinr_hist[nid]
        if not hist: continue
        x, y  = SIM.positions[nid]
        dists = [np.hypot(x-g[0], y-g[1]) for g in cfg["gnbs"]]
        d     = min(dists)
        sv    = hist[-1]
        st    = NODE_STYLE.get(ntype, NODE_STYLE["phone"])
        ax_dist.scatter(d, sv, c=st["color"], marker=st["marker"],
                        s=65, zorder=5, edgecolors="none", alpha=0.9, label=nid)
    ax_dist.set_xlabel("Distance (m)", color=MUTED_COL)
    ax_dist.set_ylabel("SINR (dB)",    color=MUTED_COL)
    ax_dist.set_ylim(-5, 36)
    ax_dist.legend(fontsize=5.5, ncol=2, loc="upper right",
                   framealpha=0.4, labelcolor=TEXT_COL,
                   facecolor=PANEL_BG, edgecolor=GRID_COL)

    # ── CDF ──────────────────────────────────────────────────────────────────
    ax_cdf.cla()
    ax_cdf.set_facecolor(PANEL_BG)
    ax_cdf.set_title("SINR CDF", fontsize=8, pad=3)
    ax_cdf.grid(color=GRID_COL, lw=0.3)
    for ni, node in enumerate(active_nodes):
        nid    = node[0]
        series = SIM.sinr_hist[nid]
        if len(series) < 5: continue
        sorted_s = np.sort(series)
        cdf      = np.arange(1, len(sorted_s)+1) / len(sorted_s)
        ax_cdf.plot(sorted_s, cdf,
                    color=cmap_tab(ni / max(len(active_nodes)-1, 1)),
                    lw=0.9, alpha=0.85, label=nid)
    ax_cdf.axvline(20, color=ACC_GREEN, lw=0.7, ls="--", alpha=0.5)
    ax_cdf.axvline(12, color=ACC_RED,   lw=0.7, ls="--", alpha=0.5)
    ax_cdf.set_xlabel("SINR (dB)", color=MUTED_COL)
    ax_cdf.set_ylabel("CDF",       color=MUTED_COL)
    ax_cdf.set_xlim(-5, 36); ax_cdf.set_ylim(0, 1)
    ax_cdf.legend(fontsize=5.5, ncol=2, loc="lower right",
                  framealpha=0.4, labelcolor=TEXT_COL,
                  facecolor=PANEL_BG, edgecolor=GRID_COL)

    # ── Aggregate Throughput ─────────────────────────────────────────────────
    ax_agg.cla()
    ax_agg.set_facecolor(PANEL_BG)
    ax_agg.set_title("Aggregate System Throughput (Mbps) — all services", fontsize=8, pad=3)
    ax_agg.grid(color=GRID_COL, lw=0.3)
    if len(SIM.total_tp) > 2:
        t_agg  = np.arange(len(SIM.total_tp)) * 0.05
        tp_arr = np.array(SIM.total_tp)
        ax_agg.plot(t_agg, tp_arr, color=cfg["color"], lw=1.2, alpha=0.9, label="Total TP")
        ax_agg.fill_between(t_agg, tp_arr, alpha=0.12, color=cfg["color"])
        # Per-service overlay
        for svc, sc in SERVICE_COLOR.items():
            svc_ns = [n for n in nodes_def if n[6]==svc and SIM.svc_filter.get(svc,True)]
            if not svc_ns: continue
            svc_tp = []
            for i in range(len(SIM.total_tp)):
                row_tp = 0.0
                for nd in svc_ns:
                    h = SIM.tp_hist[nd[0]]
                    if i < len(h): row_tp += h[i]
                svc_tp.append(row_tp)
            ax_agg.plot(t_agg[:len(svc_tp)], svc_tp,
                        color=sc, lw=0.8, ls="--", alpha=0.65, label=svc)
    ax_agg.set_xlabel("Time (s)", color=MUTED_COL)
    ax_agg.set_ylabel("Mbps",     color=MUTED_COL)
    ax_agg.legend(fontsize=7, loc="upper left",
                  framealpha=0.4, labelcolor=TEXT_COL,
                  facecolor=PANEL_BG, edgecolor=GRID_COL)

    # ── Status label ─────────────────────────────────────────────────────────
    if not SIM.paused:
        txt_status.set_text(f"t = {SIM.t:.1f} s  |  ▶ RUNNING")
        txt_status.set_color(ACC_GREEN)

    return []   # blit=False works better with cla() pattern


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KEYBOARD SHORTCUTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def on_key(event):
    if event.key == " ":
        on_pause(None)
    elif event.key == "r":
        on_reset(None)
    elif event.key == "e":
        on_export(None)
    elif event.key in ("1","2","3","4"):
        names = list(ENVIRONMENTS.keys())
        idx   = int(event.key) - 1
        if idx < len(names):
            radio_env.set_active(idx)
            on_env_change(names[idx])
    elif event.key == "+" or event.key == "=":
        new_v = min(5.0, SIM.speed + 0.5)
        slider_speed.set_val(new_v)
    elif event.key == "-":
        new_v = max(0.1, SIM.speed - 0.5)
        slider_speed.set_val(new_v)

fig.canvas.mpl_connect("key_press_event", on_key)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PRE-COMPUTE ALL HEATMAPS (background)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("  3GPP 6G NR FR3 @ 24 GHz — Live Simulation")
print("="*60)
print("\n  Pre-computing heatmaps for all environments…")
for ename in ENVIRONMENTS:
    SIM_TMP = SIM
    if ename not in SIM.heatmaps:
        print(f"    {ename}…", end="", flush=True)
        SIM.heatmaps[ename] = build_heatmap(ENVIRONMENTS[ename])
        print(" done")

print("\n  Controls:")
print("    SPACE       Pause / Resume")
print("    R           Reset current environment")
print("    E           Export SINR data to CSV")
print("    1/2/3/4     Switch environment")
print("    + / -       Increase / decrease sim speed")
print("\n  Starting animation …\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LAUNCH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
anim = FuncAnimation(fig, draw_frame, interval=60, blit=False, cache_frame_data=False)
plt.show()
