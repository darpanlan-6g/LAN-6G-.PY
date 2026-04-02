"""
================================================================================
  3GPP 6G THz Network — SINR Radar Charts + Multi-Layer Heatmaps (Live)
================================================================================
  NEW in this version:
    ▸ SINR Radar / Spider chart  — per-node KPI wheel (SINR, TP, Latency,
                                    Coverage, BF-gain, Reliability)
    ▸ Dynamic SINR heatmap       — plasma + contour overlays, refreshes live
    ▸ Polar SINR map             — beamforming angular SINR in polar coords
    ▸ 3-D surface heatmap        — 2-D SINR field rendered as filled contourf
    ▸ Per-service heatmap grid   — URLLC / eMBB / mMTC / XR / V2X side-by-side
    ▸ Rolling SINR histogram     — live distribution per environment
    ▸ gNB SINR gradient rings    — visual coverage quality overlay on topology

  Physics (NS3-aligned):
    • Friis path loss  +  THz molecular absorption
    • Massive MIMO beamforming gain
    • Shannon capacity  (η = 0.65)
    • Handover detection (SINR threshold)
    • Shadowing σ = 1.8 dB

  Use Cases (6):
    1. XR Surgery        300 GHz  100 GHz BW  <1 ms
    2. Auto Factory      140 GHz   50 GHz BW  <2 ms
    3. Smart Intersection 300 GHz  80 GHz BW  <0.5 ms
    4. THz Backhaul        1 THz  300 GHz BW  <0.1 ms
    5. Tunnel Rescue      100 GHz  30 GHz BW  <5 ms
    6. Holo Classroom     300 GHz  60 GHz BW  <3 ms

  Controls: SPACE pause | R reset | E export | 1-6 env
            H heatmap | T trails | L links | + / - speed
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")           # change to Qt5Agg / MacOSX / Agg if needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.widgets import Button, RadioButtons, Slider, CheckButtons
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import warnings, time, csv
warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PALETTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG       = "#0D1117"
PANEL_BG = "#161B22"
GRID_COL = "#21262D"
TEXT_COL = "#E6EDF3"
MUTED    = "#7D8590"
BLUE     = "#388BFD"
GREEN    = "#3FB950"
RED      = "#F85149"
ORANGE   = "#D29922"
PURPLE   = "#BC8CFF"
TEAL     = "#39D353"
CYAN     = "#39C5CF"
PINK     = "#FF6EB4"

SVC_COLOR = {"URLLC": RED, "eMBB": BLUE, "mMTC": GREEN,
             "XR": PURPLE, "V2X": ORANGE}

NODE_STYLE = {
    "surgeon"   : {"color": RED,    "marker": "*",  "sz": 160, "label": "Surgeon"},
    "robot"     : {"color": PURPLE, "marker": "h",  "sz": 120, "label": "Robot Arm"},
    "sensor"    : {"color": TEAL,   "marker": "+",  "sz":  80, "label": "IoT Sensor"},
    "agv"       : {"color": ORANGE, "marker": "D",  "sz": 110, "label": "AGV"},
    "car"       : {"color": BLUE,   "marker": "^",  "sz": 110, "label": "Car"},
    "drone"     : {"color": CYAN,   "marker": "v",  "sz": 100, "label": "Drone"},
    "backhaul"  : {"color": GREEN,  "marker": "s",  "sz": 120, "label": "Backhaul"},
    "rescuer"   : {"color": RED,    "marker": "P",  "sz": 100, "label": "Rescuer"},
    "holo_disp" : {"color": PINK,   "marker": "8",  "sz": 110, "label": "Holo Display"},
    "student"   : {"color": PURPLE, "marker": "o",  "sz":  80, "label": "Student"},
    "camera"    : {"color": TEAL,   "marker": "x",  "sz":  80, "label": "Camera"},
    "rsu"       : {"color": ORANGE, "marker": "H",  "sz": 100, "label": "RSU"},
}

MAT_COLOR = {"wall":"#1C2227","glass":"#0D2030","metal":"#2D1F10",
             "concrete":"#1E2228","free":"#0D1117"}
MAT_EDGE  = {"wall":"#484F58","glass":"#388BFD","metal":"#D29922",
             "concrete":"#555D64","free":"#21262D"}

# Custom diverging SINR colormap: deep-red → yellow → bright-green
_SINR_CMAP = LinearSegmentedColormap.from_list(
    "sinr_thz",
    ["#8B0000","#C0392B","#E74C3C","#E67E22",
     "#F39C12","#F1C40F","#2ECC71","#27AE60","#1ABC9C"],
    N=256
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  THz PHYSICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_THZ_ABS = {0.10e12:0.40, 0.14e12:0.50, 0.30e12:1.20, 1.00e12:8.00}

def thz_abs_db(d, f):
    k = _THZ_ABS.get(f, 1.0)
    return k * max(d, 0.01) * 10 / np.log(10)

def friis_db(d, f):
    c = 3e8
    return 20*np.log10(max(4*np.pi*max(d,0.01)*f/c, 1e-30))

def compute_sinr(nx, ny, gnbs, cfg, rng):
    f, tx, pen, bf = (cfg["freq_hz"], cfg["tx_power_dbm"],
                      cfg["pen_loss_avg"], cfg["beamforming_gain_db"])
    noise = cfg["noise_floor_dbm"]
    rxs = []
    for gx, gy, *_ in gnbs:
        d  = np.hypot(nx-gx, ny-gy)
        pl = friis_db(d, f) + thz_abs_db(d, f) + pen*0.20
        rxs.append(tx - pl + bf + rng.normal(0, 1.8))
    rxs.sort(reverse=True)
    sig  = 10**(rxs[0]/10)
    intf = sum(10**(p/10) for p in rxs[1:]) if len(rxs)>1 else 0
    nois = 10**(noise/10)
    return 10*np.log10(max(sig/(intf+nois), 1e-12))

def shannon_tp(sinr_db, bw_ghz):
    return bw_ghz * np.log2(1 + 10**(sinr_db/10)) * 0.65

def sinr_qcolor(v):
    if v >= 18: return GREEN
    if v >= 10: return ORANGE
    if v >=  3: return "#E67E22"
    return RED

def build_heatmap(cfg, res=45):
    W, H = cfg["area"]
    xs = np.linspace(0, W, res)
    ys = np.linspace(0, H, res)
    XX, YY = np.meshgrid(xs, ys)
    rng = np.random.default_rng(0)
    G = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            G[i,j] = compute_sinr(XX[i,j], YY[i,j], cfg["gnbs"], cfg, rng)
    return XX, YY, gaussian_filter(G, sigma=1.5)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENVIRONMENTS  (same 6 use-cases, compact form)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENVIRONMENTS = {
    "XR Surgery": {
        "desc":"Holographic remote surgery · 300 GHz · <1 ms URLLC",
        "area":(12,10), "freq_hz":300e9, "bw_ghz":100,
        "tx_power_dbm":20, "pen_loss_avg":8,
        "beamforming_gain_db":30, "noise_floor_dbm":-80,
        "color":RED, "latency_target_ms":1.0,
        "buildings":[
            (0,0,12,10,"OR Chamber","wall",28),
            (1,1,4,2,"Instrument Bay","metal",35),
            (7,1,4,2,"Imaging Suite","metal",35),
        ],
        "gnbs":[(6.0,9.5,"THz-AP",2.5)],
        "nodes":[
            ("Surg1","surgeon",  3.0,5.0, 0.00, 0.00,"URLLC",False),
            ("Surg2","surgeon",  9.0,5.0, 0.00, 0.00,"URLLC",False),
            ("RobotL","robot",   4.5,4.5, 0.01, 0.00,"XR",   True),
            ("RobotR","robot",   7.5,4.5,-0.01, 0.00,"XR",   True),
            ("Cam1","camera",    2.0,8.5, 0.00, 0.00,"URLLC",False),
            ("Cam2","camera",   10.0,8.5, 0.00, 0.00,"URLLC",False),
            ("Holo","holo_disp", 6.0,7.5, 0.00, 0.00,"XR",   False),
        ],
    },
    "Auto Factory": {
        "desc":"Industry 6.0 smart factory · 140 GHz · URLLC + mMTC",
        "area":(80,60), "freq_hz":140e9, "bw_ghz":50,
        "tx_power_dbm":30, "pen_loss_avg":25,
        "beamforming_gain_db":25, "noise_floor_dbm":-85,
        "color":ORANGE, "latency_target_ms":2.0,
        "buildings":[
            (0,0,80,60,"Factory Shell","metal",32),
            (5,5,20,15,"Line A","metal",30),
            (30,5,20,15,"Line B","metal",30),
            (55,5,20,15,"Line C","metal",30),
            (5,40,30,15,"Warehouse","concrete",22),
            (45,40,30,15,"Control Room","wall",20),
        ],
        "gnbs":[(20,30,"gNB-F0",8),(60,30,"gNB-F1",8),(40,10,"gNB-F2",8)],
        "nodes":[
            ("AGV1","agv",    10,30, 0.80, 0.00,"URLLC",False),
            ("AGV2","agv",    40,30,-0.70, 0.00,"URLLC",False),
            ("AGV3","agv",    65,20, 0.00, 0.60,"URLLC",False),
            ("Arm1","robot",  15,12, 0.00, 0.00,"URLLC",False),
            ("Arm2","robot",  40,12, 0.00, 0.00,"URLLC",False),
            ("Arm3","robot",  65,12, 0.00, 0.00,"URLLC",False),
            ("S1","sensor",   10,45, 0.00, 0.00,"mMTC", False),
            ("S2","sensor",   30,45, 0.00, 0.00,"mMTC", False),
            ("S3","sensor",   55,45, 0.00, 0.00,"mMTC", False),
            ("S4","sensor",   70,45, 0.00, 0.00,"mMTC", False),
        ],
    },
    "Smart Intersection": {
        "desc":"6G V2X crossroad · 300 GHz · zero-accident <0.5 ms",
        "area":(100,100), "freq_hz":300e9, "bw_ghz":80,
        "tx_power_dbm":38, "pen_loss_avg":15,
        "beamforming_gain_db":28, "noise_floor_dbm":-82,
        "color":CYAN, "latency_target_ms":0.5,
        "buildings":[
            (0,0,38,38,"Block NW","concrete",22),
            (62,0,38,38,"Block NE","concrete",22),
            (0,62,38,38,"Block SW","concrete",22),
            (62,62,38,38,"Block SE","concrete",22),
        ],
        "gnbs":[(50,50,"V2X-gNB",12),(20,50,"RSU-W",6),
                (80,50,"RSU-E",6),(50,20,"RSU-N",6),(50,80,"RSU-S",6)],
        "nodes":[
            ("C1","car",  5,50, 2.5, 0.0,"V2X",  False),
            ("C2","car", 95,50,-2.2, 0.0,"V2X",  False),
            ("C3","car", 50, 5, 0.0, 2.3,"V2X",  False),
            ("C4","car", 50,95, 0.0,-2.0,"V2X",  False),
            ("Dr1","drone",30,30, 0.15, 0.10,"URLLC",True),
            ("Dr2","drone",70,70,-0.12, 0.08,"URLLC",True),
            ("Ped1","student",48,48,0.08,0.05,"URLLC",True),
            ("Cam1","camera",38,38,0.0, 0.0,"URLLC",False),
            ("Cam2","camera",62,62,0.0, 0.0,"URLLC",False),
        ],
    },
    "THz Backhaul": {
        "desc":"1 THz rooftop P2P backhaul · 300 GHz BW · 1 Tbps",
        "area":(500,100), "freq_hz":1.00e12, "bw_ghz":300,
        "tx_power_dbm":45, "pen_loss_avg":5,
        "beamforming_gain_db":40, "noise_floor_dbm":-75,
        "color":GREEN, "latency_target_ms":0.1,
        "buildings":[
            (0,35,30,30,"Building A","concrete",22),
            (235,35,30,30,"Relay","concrete",22),
            (470,35,30,30,"Building B","concrete",22),
        ],
        "gnbs":[(15,65,"THz-TX",15),(250,65,"THz-Relay",15),(485,65,"THz-RX",15)],
        "nodes":[
            ("BH0","backhaul", 15, 65, 0.0, 0.0,"eMBB",False),
            ("BH1","backhaul",250, 65, 0.0, 0.0,"eMBB",False),
            ("BH2","backhaul",485, 65, 0.0, 0.0,"eMBB",False),
            ("Dr1","drone",    80, 55, 1.5, 0.08,"eMBB",False),
            ("Dr2","drone",   300, 70,-1.2, 0.05,"eMBB",False),
            ("Dr3","drone",   420, 60, 1.0,-0.06,"eMBB",False),
        ],
    },
    "Tunnel Rescue": {
        "desc":"6G rescue in confined tunnel · 100 GHz · URLLC critical",
        "area":(150,15), "freq_hz":100e9, "bw_ghz":30,
        "tx_power_dbm":27, "pen_loss_avg":30,
        "beamforming_gain_db":18, "noise_floor_dbm":-88,
        "color":ORANGE, "latency_target_ms":5.0,
        "buildings":[
            (0,0,150,15,"Tunnel Walls","metal",32),
            (0,0,150,2,"Floor","concrete",20),
            (0,13,150,2,"Ceiling","concrete",20),
            (60,2,10,11,"Debris","concrete",28),
        ],
        "gnbs":[(5,7,"THz-R0",2),(75,7,"THz-R1",2),(145,7,"THz-R2",2)],
        "nodes":[
            ("Rs1","rescuer",  8, 7, 0.50,0.0,"URLLC",False),
            ("Rs2","rescuer", 25, 7, 0.45,0.0,"URLLC",False),
            ("Rs3","rescuer", 45, 7, 0.40,0.0,"URLLC",False),
            ("Dr1","drone",   15,10, 0.60,0.0,"URLLC",False),
            ("Dr2","drone",   35,10, 0.55,0.0,"URLLC",False),
            ("S1","sensor",   70, 7, 0.0, 0.0,"mMTC", False),
            ("S2","sensor",  110, 7, 0.0, 0.0,"mMTC", False),
        ],
    },
    "Holo Classroom": {
        "desc":"Holographic tele-education · 300 GHz · 10 Gbps/user eMBB",
        "area":(20,15), "freq_hz":300e9, "bw_ghz":60,
        "tx_power_dbm":22, "pen_loss_avg":12,
        "beamforming_gain_db":26, "noise_floor_dbm":-82,
        "color":PURPLE, "latency_target_ms":3.0,
        "buildings":[
            (0,0,20,15,"Classroom","wall",20),
            (0,0,2,15,"W Wall","concrete",22),
            (18,0,2,15,"E Wall","concrete",22),
            (1,11,18,3,"Pres Wall","wall",20),
        ],
        "gnbs":[(10,13.5,"THz-AP0",2),(3,5.5,"THz-AP1",2),(17,5.5,"THz-AP2",2)],
        "nodes":[
            ("St1","student", 4, 3, 0.04, 0.03,"eMBB",True),
            ("St2","student", 8, 3,-0.03, 0.04,"eMBB",True),
            ("St3","student",12, 3, 0.05,-0.02,"eMBB",True),
            ("St4","student",16, 3,-0.04, 0.03,"eMBB",True),
            ("St5","student", 4, 7, 0.03, 0.04,"eMBB",True),
            ("St6","student", 8, 7,-0.04,-0.03,"eMBB",True),
            ("St7","student",12, 7, 0.04, 0.02,"eMBB",True),
            ("St8","student",16, 7,-0.03, 0.04,"eMBB",True),
            ("Holo1","holo_disp", 5,12, 0.0, 0.0,"XR",False),
            ("Holo2","holo_disp",10,12, 0.0, 0.0,"XR",False),
            ("Holo3","holo_disp",15,12, 0.0, 0.0,"XR",False),
        ],
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIMULATION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimState:
    def __init__(self):
        self.env_name    = "XR Surgery"
        self.paused      = False
        self.speed       = 1.0
        self.t           = 0.0
        self.frame       = 0
        self.show_hm     = True
        self.show_trails = True
        self.show_links  = True
        self.svc_filter  = {s:True for s in SVC_COLOR}
        self.heatmaps    = {}
        self.reset()

    @property
    def cfg(self): return ENVIRONMENTS[self.env_name]

    def reset(self):
        cfg = self.cfg
        self.rng       = np.random.default_rng(42)
        self.pos       = {n[0]:[float(n[2]),float(n[3])] for n in cfg["nodes"]}
        self.vel       = {n[0]:[n[4]*self.speed,n[5]*self.speed] for n in cfg["nodes"]}
        self.bounce    = {n[0]:n[7] for n in cfg["nodes"]}
        self.sinr_hist = {n[0]:[] for n in cfg["nodes"]}
        self.tp_hist   = {n[0]:[] for n in cfg["nodes"]}
        self.lat_hist  = {n[0]:[] for n in cfg["nodes"]}
        self.trail_x   = {n[0]:[] for n in cfg["nodes"]}
        self.trail_y   = {n[0]:[] for n in cfg["nodes"]}
        self.prev_cell = {n[0]:None for n in cfg["nodes"]}
        self.handovers = 0
        self.total_tp  = []
        self.t = 0.0; self.frame = 0
        if self.env_name not in self.heatmaps:
            print(f"  [HM] {self.env_name}…",end="",flush=True)
            self.heatmaps[self.env_name] = build_heatmap(cfg)
            print(" done")

    def step(self, dt=0.04):
        if self.paused: return
        cfg = self.cfg; W,H = cfg["area"]
        self.t += dt*self.speed; self.frame += 1
        total_tp = 0.0
        for node in cfg["nodes"]:
            nid = node[0]
            x,y   = self.pos[nid]; vx,vy = self.vel[nid]
            x+=vx*dt; y+=vy*dt
            if self.bounce[nid]:
                if x<=0 or x>=W: vx*=-1; x=np.clip(x,0.1,W-0.1)
                if y<=0 or y>=H: vy*=-1; y=np.clip(y,0.1,H-0.1)
            else:
                if vx>0 and x>W+2: x=-2
                if vx<0 and x<-2:  x=W+2
                if vy>0 and y>H+2: y=-2
                if vy<0 and y<-2:  y=H+2
                x=np.clip(x,0.1,W-0.1); y=np.clip(y,0.1,H-0.1)
            self.pos[nid]=[x,y]; self.vel[nid]=[vx,vy]
            sv  = compute_sinr(x,y,cfg["gnbs"],cfg,self.rng)
            tp  = shannon_tp(sv,cfg["bw_ghz"])
            lat = max(0.05, cfg["latency_target_ms"]*2*np.exp(-sv/15))
            for hist,val,mx in [(self.sinr_hist[nid],sv,400),
                                 (self.tp_hist[nid],tp*1000,400),
                                 (self.lat_hist[nid],lat,400)]:
                hist.append(val)
                if len(hist)>mx: hist.pop(0)
            total_tp += tp
            self.trail_x[nid].append(x); self.trail_y[nid].append(y)
            if len(self.trail_x[nid])>60: self.trail_x[nid].pop(0); self.trail_y[nid].pop(0)
            gnbs = cfg["gnbs"]
            cell = int(np.argmin([np.hypot(x-g[0],y-g[1]) for g in gnbs]))
            if self.prev_cell[nid] is not None and self.prev_cell[nid]!=cell:
                self.handovers+=1
            self.prev_cell[nid]=cell
        self.total_tp.append(total_tp*1000)
        if len(self.total_tp)>400: self.total_tp.pop(0)

    def export_csv(self):
        fname = f"thz_sinr_{self.env_name.replace(' ','_')}_{int(time.time())}.csv"
        cfg = self.cfg
        with open(fname,"w",newline="") as f:
            w = csv.writer(f)
            w.writerow(["Node","Type","Service","X","Y","SINR_dB","TP_Gbps","Lat_ms"])
            for node in cfg["nodes"]:
                nid=node[0]; x,y=self.pos[nid]
                sv = self.sinr_hist[nid][-1] if self.sinr_hist[nid] else 0
                tp = self.tp_hist[nid][-1]   if self.tp_hist[nid]   else 0
                lt = self.lat_hist[nid][-1]  if self.lat_hist[nid]  else 0
                w.writerow([nid,node[1],node[6],f"{x:.2f}",f"{y:.2f}",
                            f"{sv:.2f}",f"{tp:.1f}",f"{lt:.3f}"])
        print(f"\n✅ Exported → {fname}")

SIM = SimState()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RADAR HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RADAR_CATS = ["SINR\n(dB)", "Throughput\n(Gbps)", "Latency\nScore",
              "Coverage", "BF Gain\nScore", "Reliability"]

def node_radar_values(nid, cfg, sim):
    """Return 0-1 normalised scores for each radar category."""
    h_sinr = sim.sinr_hist[nid]
    h_tp   = sim.tp_hist[nid]
    h_lat  = sim.lat_hist[nid]

    sv  = np.mean(h_sinr[-30:]) if h_sinr else 5.0
    tp  = np.mean(h_tp[-30:])   if h_tp   else 0.0
    lat = np.mean(h_lat[-30:])  if h_lat  else cfg["latency_target_ms"]

    # distance to nearest gNB, normalised
    x, y = sim.pos[nid]
    dists = [np.hypot(x-g[0], y-g[1]) for g in cfg["gnbs"]]
    d_min = min(dists)
    W, H  = cfg["area"]
    cov_score = max(0, 1 - d_min / (0.5*np.hypot(W, H)))

    bf_score  = (cfg["beamforming_gain_db"] - 10) / 35   # 10-45 dBi → 0-1
    rel_score = min(1.0, max(0.0, (sv + 5) / 30))

    sinr_norm = min(1.0, max(0.0, (sv + 5) / 30))
    tp_norm   = min(1.0, tp / max(cfg["bw_ghz"] * 20, 1))   # relative to peak
    lat_score = min(1.0, max(0.0, 1 - lat / (cfg["latency_target_ms"] * 5)))

    return [sinr_norm, tp_norm, lat_score, cov_score, bf_score, rel_score]


def draw_radar(ax, values, labels, color, title="", fill_alpha=0.25):
    """Draw a single radar/spider chart on ax (must be polar)."""
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals   = values + [values[0]]
    angs   = angles + [angles[0]]

    ax.set_facecolor(PANEL_BG)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Grid rings
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(np.linspace(0,2*np.pi,200), [r]*200,
                color=GRID_COL, lw=0.4, ls="--")

    # Spokes
    for a in angles:
        ax.plot([a, a], [0, 1], color=GRID_COL, lw=0.4)

    # Data
    ax.plot(angs, vals, color=color, lw=1.5, zorder=5)
    ax.fill(angs, vals, color=color, alpha=fill_alpha, zorder=4)

    # Labels
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=6, color=TEXT_COL)
    ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["25","50","75","100"], fontsize=5, color=MUTED)
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_color(GRID_COL)
    if title:
        ax.set_title(title, fontsize=7, color=color, pad=8, fontweight="bold")


def draw_polar_sinr(ax, cfg, sim, n_angles=72):
    """
    Polar SINR map: for each angle from the primary gNB,
    compute SINR at increasing radii and plot as a filled contour.
    """
    ax.set_facecolor(PANEL_BG)
    ax.spines["polar"].set_color(GRID_COL)

    W, H = cfg["area"]
    gx0, gy0 = cfg["gnbs"][0][0], cfg["gnbs"][0][1]
    r_max = min(W, H) * 0.50
    n_r   = 30

    radii  = np.linspace(1, r_max, n_r)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    SINR_POLAR = np.zeros((n_r, n_angles))
    rng = np.random.default_rng(SIM.frame % 50)   # slowly drift for animation

    for ri, r in enumerate(radii):
        for ai, a in enumerate(angles):
            nx = np.clip(gx0 + r*np.cos(a), 0.1, W-0.1)
            ny = np.clip(gy0 + r*np.sin(a), 0.1, H-0.1)
            SINR_POLAR[ri, ai] = compute_sinr(nx, ny, cfg["gnbs"], cfg, rng)

    SINR_POLAR = gaussian_filter(SINR_POLAR, sigma=1.0)
    A, R = np.meshgrid(angles, radii)
    cf = ax.contourf(A, R, SINR_POLAR,
                     levels=np.linspace(-5, 25, 20),
                     cmap=_SINR_CMAP, alpha=0.88)

    # Mark active nodes
    for node in cfg["nodes"]:
        nid = node[0]
        if not SIM.svc_filter.get(node[6], True): continue
        nx, ny = SIM.pos[nid]
        dx, dy = nx-gx0, ny-gy0
        r_  = np.hypot(dx, dy)
        a_  = np.arctan2(dy, dx)
        if r_ <= r_max:
            st = NODE_STYLE.get(node[1], NODE_STYLE["sensor"])
            ax.plot(a_, r_, marker=st["marker"], color=st["color"],
                    markersize=6, zorder=8)

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_yticks([r_max*0.25, r_max*0.5, r_max*0.75, r_max])
    ax.set_yticklabels([f"{r_max*0.25:.0f}m","","",""], fontsize=5, color=MUTED)
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels(["N","NE","E","SE","S","SW","W","NW"],
                       fontsize=6, color=MUTED)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED)
    ax.spines["polar"].set_color(GRID_COL)
    ax.grid(color=GRID_COL, lw=0.3)
    ax.set_title("Polar SINR Map\n(from primary gNB)", fontsize=7,
                 color=cfg["color"], pad=6)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE LAYOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":7.5,
    "axes.facecolor":PANEL_BG,"figure.facecolor":BG,
    "axes.edgecolor":GRID_COL,"axes.labelcolor":TEXT_COL,
    "xtick.color":MUTED,"ytick.color":MUTED,
    "text.color":TEXT_COL,"axes.titlecolor":TEXT_COL,
    "axes.grid":True,"grid.color":GRID_COL,
    "grid.linewidth":0.35,"axes.spines.top":False,"axes.spines.right":False,
})

fig = plt.figure(figsize=(26,16), facecolor=BG)
try: fig.canvas.manager.set_window_title("6G THz — SINR Radar & Heatmaps")
except: pass

# ── Row structure ─────────────────────────────────────────────────────────────
#  Row 0 : topology (2×2) | SINR heatmap contourf | polar SINR | SINR radar ×2
#  Row 1 : SINR bars      | TP timeline           | lat timeline | SINR ts | CDF
#  Row 2 : per-service heatmap grid (5 panels)
#  Row 3 : aggregate TP + SINR histogram (2 panels)
#  ctrl  : bottom strip

outer = gridspec.GridSpec(5, 1, figure=fig,
    hspace=0.52, top=0.93, bottom=0.10, left=0.01, right=0.99,
    height_ratios=[3.2, 2.2, 2.0, 1.8, 0.0])

# Row 0
gs0 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0],
    wspace=0.35)
ax_topo   = fig.add_subplot(gs0[0:2])       # topology spans 2 cols
ax_hm2    = fig.add_subplot(gs0[2])         # SINR filled-contour heatmap
ax_polar  = fig.add_subplot(gs0[3], projection="polar")  # polar SINR
ax_radar0 = fig.add_subplot(gs0[4], projection="polar")  # radar chart (node 0)

# Row 1
gs1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1], wspace=0.40)
ax_sinr_bars = fig.add_subplot(gs1[0])
ax_tp_line   = fig.add_subplot(gs1[1])
ax_lat_line  = fig.add_subplot(gs1[2])
ax_sinr_ts   = fig.add_subplot(gs1[3])
ax_cdf       = fig.add_subplot(gs1[4])

# Row 2 — per-service SINR heatmaps
gs2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[2], wspace=0.38)
ax_svc_hm = [fig.add_subplot(gs2[i]) for i in range(5)]
SVC_ORDER  = ["URLLC","eMBB","mMTC","XR","V2X"]

# Row 3 — agg TP + SINR histogram
gs3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3], wspace=0.38)
ax_agg  = fig.add_subplot(gs3[0])
ax_hist = fig.add_subplot(gs3[1])

# ── Control strip (absolute axes at bottom) ──────────────────────────────────
ctrl_y0, ctrl_h = 0.01, 0.085
ax_radio = fig.add_axes([0.00, ctrl_y0, 0.13, ctrl_h])
ax_svc   = fig.add_axes([0.14, ctrl_y0, 0.09, ctrl_h])
ax_disp  = fig.add_axes([0.24, ctrl_y0, 0.09, ctrl_h])
ax_spd   = fig.add_axes([0.34, ctrl_y0+0.035, 0.09, 0.025])
ax_leg   = fig.add_axes([0.44, ctrl_y0, 0.28, ctrl_h])
ax_inf   = fig.add_axes([0.73, ctrl_y0, 0.27, ctrl_h])

for ax in [ax_leg, ax_inf]:
    ax.axis("off"); ax.set_facecolor(PANEL_BG)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.970,
    "3GPP 6G THz  ·  SINR Radar Charts + Multi-Layer Heatmaps  ·  Live Simulation",
    ha="center", fontsize=13, fontweight="bold", color=TEXT_COL)
fig.text(0.5, 0.953,
    "100 GHz – 1 THz  |  Friis + THz absorption  |  Massive MIMO BF  |  "
    "Radar KPI  ·  Polar SINR  ·  Contour Heatmaps  ·  Per-Service Maps",
    ha="center", fontsize=7.5, color=MUTED)
txt_status = fig.text(0.99,0.970,"t=0.0s ▶",
    ha="right", fontsize=9, fontweight="bold", color=GREEN)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WIDGETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENV_NAMES = list(ENVIRONMENTS.keys())
ax_radio.set_facecolor(PANEL_BG)
ax_radio.set_title("Use Case", fontsize=8, color=TEXT_COL, pad=1)
radio_env = RadioButtons(ax_radio, ENV_NAMES, activecolor=BLUE)
for l in radio_env.labels: l.set_color(TEXT_COL); l.set_fontsize(7)

def on_env(lbl):
    SIM.env_name=lbl; SIM.reset()
radio_env.on_clicked(on_env)

ax_svc.set_facecolor(PANEL_BG)
ax_svc.set_title("Service", fontsize=8, color=TEXT_COL, pad=1)
chk_svc = CheckButtons(ax_svc, list(SVC_COLOR.keys()), [True]*5)
for l in chk_svc.labels: l.set_color(TEXT_COL); l.set_fontsize(7)
chk_svc.on_clicked(lambda lbl: SIM.svc_filter.update({lbl: not SIM.svc_filter[lbl]}))

ax_disp.set_facecolor(PANEL_BG)
ax_disp.set_title("Display", fontsize=8, color=TEXT_COL, pad=1)
chk_disp = CheckButtons(ax_disp,["Heatmap","Trails","Links"],[True,True,True])
for l in chk_disp.labels: l.set_color(TEXT_COL); l.set_fontsize(7)
def on_disp(lbl):
    if lbl=="Heatmap": SIM.show_hm=not SIM.show_hm
    if lbl=="Trails":  SIM.show_trails=not SIM.show_trails
    if lbl=="Links":   SIM.show_links=not SIM.show_links
chk_disp.on_clicked(on_disp)

ax_spd.set_facecolor(PANEL_BG)
sl_spd = Slider(ax_spd,"Speed",0.1,6.0,valinit=1.0,color=BLUE,track_color=GRID_COL)
sl_spd.label.set_color(TEXT_COL); sl_spd.valtext.set_color(BLUE)
sl_spd.on_changed(lambda v: setattr(SIM,"speed",v))

for pos,lbl,attr in [
    ([0.34,ctrl_y0+0.002,0.045,0.028],"⏸ Pause","pause"),
    ([0.385,ctrl_y0+0.002,0.045,0.028],"↺ Reset","reset"),
    ([0.34,ctrl_y0+0.034,0.09,0.028],"⬇ Export CSV","export"),
]:
    bax = fig.add_axes(pos)
    btn = Button(bax,lbl,color=PANEL_BG,hovercolor=GRID_COL)
    btn.label.set_color(TEXT_COL); btn.label.set_fontsize(7.5)
    if attr=="pause":  btn_pause=btn
    elif attr=="reset": btn_reset=btn
    else: btn_export=btn

def on_pause(ev):
    SIM.paused=not SIM.paused
    btn_pause.label.set_text("▶ Resume" if SIM.paused else "⏸ Pause")
    txt_status.set_color(ORANGE if SIM.paused else GREEN)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(lambda ev: SIM.reset())
btn_export.on_clicked(lambda ev: SIM.export_csv())

# Legend
ax_leg.set_title("Node Types", fontsize=8, color=TEXT_COL, pad=1)
yl=0.97
for nt,st in NODE_STYLE.items():
    ax_leg.plot(0.04,yl,marker=st["marker"],color=st["color"],
                markersize=6,transform=ax_leg.transAxes,clip_on=False)
    ax_leg.text(0.10,yl,st["label"],fontsize=6.5,color=TEXT_COL,
                va="center",transform=ax_leg.transAxes)
    yl-=0.082

# RF Info
ax_inf.set_title("RF / NS3 Config", fontsize=8, color=TEXT_COL, pad=1)
for i,(k,v) in enumerate([
    ("Freq range","100 GHz – 1 THz"),("BW","30 – 300 GHz"),
    ("BF gain","18 – 40 dBi"),("PL model","Friis + THz Absorption"),
    ("NS3","ThzSpectrumPropagationLoss"),("Mobility","Const/RandWalk/WP"),
    ("FlowMon","FlowMonitorHelper"),("SINR HO","LteHandoverAlgorithm"),
    ("Heatmap","SINR contourf + polar"),("Radar","6-axis KPI spider"),
]):
    yp=0.95-i*0.095
    ax_inf.text(0.01,yp,k+":",fontsize=6.5,color=MUTED,
                transform=ax_inf.transAxes,va="center")
    ax_inf.text(0.42,yp,v,fontsize=6.5,color=TEXT_COL,fontweight="bold",
                transform=ax_inf.transAxes,va="center")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DRAW FRAME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cmap_tab  = plt.get_cmap("tab20")
norm_sinr = Normalize(vmin=-5, vmax=25)
_POLAR_SKIP = 4   # recompute polar every N frames (expensive)

def draw_frame(_fn):
    SIM.step(dt=0.04)
    cfg       = SIM.cfg
    W, H      = cfg["area"]
    nodes_def = cfg["nodes"]
    active    = [n for n in nodes_def if SIM.svc_filter.get(n[6], True)]
    ec        = cfg["color"]
    na        = max(len(active)-1, 1)
    freq_str  = f"{cfg['freq_hz']/1e9:.0f} GHz" if cfg["freq_hz"]<1e12 \
                else f"{cfg['freq_hz']/1e12:.1f} THz"

    # ── 1. TOPOLOGY ──────────────────────────────────────────────────────────
    ax_topo.cla(); ax_topo.set_facecolor(PANEL_BG)
    ax_topo.set_xlim(0,W); ax_topo.set_ylim(0,H)
    ax_topo.set_aspect("equal","box")
    ax_topo.set_title(f"{SIM.env_name}  [{freq_str} · {cfg['bw_ghz']} GHz BW]",
                      fontsize=9,color=ec,fontweight="bold",pad=3)
    ax_topo.set_xlabel("x (m)",color=MUTED); ax_topo.set_ylabel("y (m)",color=MUTED)
    ax_topo.text(0.01,0.99,cfg["desc"],transform=ax_topo.transAxes,
                 fontsize=5.5,color=MUTED,va="top",
                 bbox=dict(boxstyle="round,pad=0.3",fc=BG,ec=GRID_COL,alpha=0.8))

    if SIM.show_hm and SIM.env_name in SIM.heatmaps:
        XX,YY,hm = SIM.heatmaps[SIM.env_name]
        ax_topo.pcolormesh(XX,YY,hm,cmap=_SINR_CMAP,
                           vmin=-5,vmax=25,shading="gouraud",alpha=0.38)

    for bld in cfg["buildings"]:
        bx,by,bw,bh,blbl,mat,_ = bld
        ax_topo.add_patch(mpatches.FancyBboxPatch(
            (bx,by),bw,bh,boxstyle="round,pad=0.3",lw=0.7,
            edgecolor=MAT_EDGE.get(mat,"#484F58"),
            facecolor=MAT_COLOR.get(mat,"#1E2228"),alpha=0.80))
        ax_topo.text(bx+bw/2,by+bh/2,blbl,ha="center",va="center",
                     fontsize=5,color=MUTED)

    gnbs_xy = [(g[0],g[1]) for g in cfg["gnbs"]]
    for gx,gy,glbl,*_ in cfg["gnbs"]:
        rmax = min(W,H)*0.28
        for r,a in [(rmax,0.04),(rmax*0.6,0.07),(rmax*0.3,0.12)]:
            ax_topo.add_patch(plt.Circle((gx,gy),r,color=ec,alpha=a,lw=0))
        pr = (SIM.frame*0.55)%(rmax*1.2)+2
        ax_topo.add_patch(plt.Circle((gx,gy),pr,color=ec,
                          alpha=max(0,0.35-pr/(rmax*1.5)),fill=False,lw=0.9))
        ax_topo.plot(gx,gy,"^",color=ec,ms=10,zorder=8,
                     markeredgecolor="white",markeredgewidth=0.8)
        ax_topo.text(gx,gy-H*0.04,glbl,ha="center",fontsize=6,
                     color=ec,fontweight="bold")

    for ni,node in enumerate(active):
        nid,ntype,svc = node[0],node[1],node[6]
        x,y = SIM.pos[nid]; hist=SIM.sinr_hist[nid]
        sv  = hist[-1] if hist else 12.0
        st  = NODE_STYLE.get(ntype,NODE_STYLE["sensor"])
        if SIM.show_trails and len(SIM.trail_x[nid])>2:
            pts  = np.array([SIM.trail_x[nid],SIM.trail_y[nid]]).T.reshape(-1,1,2)
            segs = np.concatenate([pts[:-1],pts[1:]],axis=1)
            nseg = len(segs)
            lc   = LineCollection(segs,cmap=_SINR_CMAP,norm=norm_sinr,lw=1.0,alpha=0.55)
            lc.set_array(np.array(hist[-nseg:]) if len(hist)>=nseg else np.full(nseg,sv))
            ax_topo.add_collection(lc)
        if SIM.show_links:
            d_  = [np.hypot(x-gx,y-gy) for gx,gy in gnbs_xy]
            bst = gnbs_xy[int(np.argmin(d_))]
            ax_topo.plot([x,bst[0]],[y,bst[1]],
                         color=sinr_qcolor(sv),lw=0.55,alpha=0.50,zorder=2)
        ax_topo.scatter(x,y,c=st["color"],marker=st["marker"],
                        s=st["sz"],zorder=7,edgecolors="white",linewidths=0.6)
        ax_topo.scatter(x+W*0.012,y+H*0.018,c=sinr_qcolor(sv),
                        s=16,zorder=9,edgecolors="none")
        ax_topo.text(x,y+H*0.04,nid,fontsize=5,ha="center",
                     color=TEXT_COL,zorder=10)

    # ── 2. SINR FILLED-CONTOUR HEATMAP ───────────────────────────────────────
    ax_hm2.cla(); ax_hm2.set_facecolor(PANEL_BG)
    ax_hm2.set_title("SINR Heatmap + Contours",fontsize=8,color=TEXT_COL,pad=3)
    if SIM.env_name in SIM.heatmaps:
        XX,YY,hm = SIM.heatmaps[SIM.env_name]
        cf = ax_hm2.contourf(XX,YY,hm,levels=np.linspace(-5,25,18),
                              cmap=_SINR_CMAP,alpha=0.90)
        ct = ax_hm2.contour(XX,YY,hm,levels=[-5,0,5,10,15,18,22,25],
                             colors="white",linewidths=0.5,alpha=0.35)
        ax_hm2.clabel(ct,fmt="%d dB",fontsize=5,colors="white",inline=True)
        # gNB markers on heatmap
        for gx,gy,glbl,*_ in cfg["gnbs"]:
            ax_hm2.plot(gx,gy,"^w",ms=8,zorder=5)
            ax_hm2.text(gx,gy+H*0.04,glbl,ha="center",fontsize=5.5,color="white")
        # Active node dots
        for node in active:
            nid=node[0]; x,y=SIM.pos[nid]
            sv=SIM.sinr_hist[nid][-1] if SIM.sinr_hist[nid] else 10
            ax_hm2.scatter(x,y,c=sinr_qcolor(sv),s=30,zorder=8,
                           edgecolors="white",linewidths=0.5)
        plt.colorbar(cf,ax=ax_hm2,fraction=0.046,pad=0.04,
                     label="SINR (dB)").ax.tick_params(labelsize=5,colors=MUTED)
    ax_hm2.set_xlabel("x (m)",color=MUTED); ax_hm2.set_ylabel("y (m)",color=MUTED)
    ax_hm2.set_xlim(0,W); ax_hm2.set_ylim(0,H)
    ax_hm2.set_aspect("equal","box")

    # ── 3. POLAR SINR MAP ─────────────────────────────────────────────────────
    ax_polar.cla()
    if SIM.frame % _POLAR_SKIP == 0:
        draw_polar_sinr(ax_polar, cfg, SIM, n_angles=60)
    else:
        ax_polar.set_facecolor(PANEL_BG)
        ax_polar.set_title("Polar SINR Map",fontsize=7,color=ec,pad=6)

    # ── 4. RADAR (first active node) ─────────────────────────────────────────
    ax_radar0.cla()
    if active:
        first = active[0]
        vals  = node_radar_values(first[0], cfg, SIM)
        st    = NODE_STYLE.get(first[1], NODE_STYLE["sensor"])
        draw_radar(ax_radar0, vals, RADAR_CATS, st["color"],
                   title=f"KPI Radar: {first[0]}")

    # ── 5. SINR BARS ─────────────────────────────────────────────────────────
    ax_sinr_bars.cla(); ax_sinr_bars.set_facecolor(PANEL_BG)
    ax_sinr_bars.set_title("Live SINR (dB)",fontsize=8,pad=3)
    ax_sinr_bars.set_xlim(-5,28)
    for i,node in enumerate(active):
        nid=node[0]; svc=node[6]
        sv=SIM.sinr_hist[nid][-1] if SIM.sinr_hist[nid] else 0
        ax_sinr_bars.barh(i,sv,color=sinr_qcolor(sv),height=0.65,
                          edgecolor="none",alpha=0.88)
        ax_sinr_bars.text(max(sv+0.2,0.3),i,f"{sv:.1f}",
                          va="center",fontsize=5.5,color=TEXT_COL)
        ax_sinr_bars.text(-4.8,i,nid,va="center",fontsize=5.5,
                          color=SVC_COLOR.get(svc,MUTED),fontweight="bold")
    ax_sinr_bars.set_yticks([])
    ax_sinr_bars.axvline(18,color=GREEN,lw=0.7,ls="--",alpha=0.5)
    ax_sinr_bars.axvline(10,color=RED,  lw=0.7,ls="--",alpha=0.5)
    ax_sinr_bars.set_xlabel("SINR (dB)",color=MUTED)
    ax_sinr_bars.grid(axis="x",color=GRID_COL,lw=0.3)

    # ── 6. THROUGHPUT TIMELINE ────────────────────────────────────────────────
    ax_tp_line.cla(); ax_tp_line.set_facecolor(PANEL_BG)
    ax_tp_line.set_title("Throughput (Gbps)",fontsize=8,pad=3)
    ax_tp_line.grid(color=GRID_COL,lw=0.3)
    for ni,node in enumerate(active):
        h=SIM.tp_hist[node[0]]
        if len(h)<2: continue
        ax_tp_line.plot(np.arange(len(h))*0.04,h,
                        color=cmap_tab(ni/na),lw=0.9,alpha=0.85,label=node[0])
    ax_tp_line.set_xlabel("t (s)",color=MUTED); ax_tp_line.set_ylabel("Gbps",color=MUTED)
    ax_tp_line.legend(fontsize=5,ncol=2,loc="upper left",
                      framealpha=0.4,facecolor=PANEL_BG,edgecolor=GRID_COL,
                      labelcolor=TEXT_COL)

    # ── 7. LATENCY TIMELINE ───────────────────────────────────────────────────
    ax_lat_line.cla(); ax_lat_line.set_facecolor(PANEL_BG)
    ax_lat_line.set_title("Latency (ms)",fontsize=8,pad=3)
    ax_lat_line.grid(color=GRID_COL,lw=0.3)
    tgt=cfg["latency_target_ms"]
    ax_lat_line.axhline(tgt,color=RED,lw=0.8,ls="--",alpha=0.6)
    for ni,node in enumerate(active):
        h=SIM.lat_hist[node[0]]
        if len(h)<2: continue
        ax_lat_line.plot(np.arange(len(h))*0.04,h,
                         color=cmap_tab(ni/na),lw=0.9,alpha=0.85)
    ax_lat_line.set_xlabel("t (s)",color=MUTED); ax_lat_line.set_ylabel("ms",color=MUTED)

    # ── 8. SINR TIME SERIES ───────────────────────────────────────────────────
    ax_sinr_ts.cla(); ax_sinr_ts.set_facecolor(PANEL_BG)
    ax_sinr_ts.set_title("SINR History",fontsize=8,pad=3)
    ax_sinr_ts.grid(color=GRID_COL,lw=0.3)
    ax_sinr_ts.axhspan(18,30,alpha=0.06,color=GREEN)
    ax_sinr_ts.axhspan(10,18,alpha=0.06,color=ORANGE)
    ax_sinr_ts.axhspan(-5,10,alpha=0.06,color=RED)
    for ni,node in enumerate(active):
        h=SIM.sinr_hist[node[0]]
        if len(h)<2: continue
        ax_sinr_ts.plot(np.arange(len(h))*0.04,h,
                        color=cmap_tab(ni/na),lw=0.85,alpha=0.85)
    ax_sinr_ts.axhline(18,color=GREEN,lw=0.6,ls="--",alpha=0.4)
    ax_sinr_ts.axhline(10,color=RED,  lw=0.6,ls="--",alpha=0.4)
    ax_sinr_ts.set_ylim(-5,30)
    ax_sinr_ts.set_xlabel("t (s)",color=MUTED)
    ax_sinr_ts.set_ylabel("SINR (dB)",color=MUTED)

    # ── 9. CDF ────────────────────────────────────────────────────────────────
    ax_cdf.cla(); ax_cdf.set_facecolor(PANEL_BG)
    ax_cdf.set_title("SINR CDF",fontsize=8,pad=3)
    ax_cdf.grid(color=GRID_COL,lw=0.3)
    for ni,node in enumerate(active):
        h=SIM.sinr_hist[node[0]]
        if len(h)<5: continue
        s=np.sort(h); cdf=np.arange(1,len(s)+1)/len(s)
        ax_cdf.plot(s,cdf,color=cmap_tab(ni/na),lw=0.9,alpha=0.85,label=node[0])
    ax_cdf.axvline(18,color=GREEN,lw=0.7,ls="--",alpha=0.5)
    ax_cdf.axvline(10,color=RED,  lw=0.7,ls="--",alpha=0.5)
    ax_cdf.set_xlim(-5,28); ax_cdf.set_ylim(0,1)
    ax_cdf.set_xlabel("SINR (dB)",color=MUTED); ax_cdf.set_ylabel("CDF",color=MUTED)
    ax_cdf.legend(fontsize=5,ncol=2,loc="lower right",
                  framealpha=0.4,facecolor=PANEL_BG,edgecolor=GRID_COL,
                  labelcolor=TEXT_COL)

    # ── 10. PER-SERVICE SINR HEATMAPS ────────────────────────────────────────
    for si,(ax_s,svc) in enumerate(zip(ax_svc_hm, SVC_ORDER)):
        ax_s.cla(); ax_s.set_facecolor(PANEL_BG)
        sc = SVC_COLOR.get(svc, MUTED)
        ax_s.set_title(f"{svc} SINR",fontsize=7.5,color=sc,pad=3)
        svc_nodes = [n for n in nodes_def if n[6]==svc]
        if svc_nodes and SIM.env_name in SIM.heatmaps:
            XX,YY,hm = SIM.heatmaps[SIM.env_name]
            # Blend heatmap with service-specific node positions
            ax_s.pcolormesh(XX,YY,hm,cmap=_SINR_CMAP,
                            vmin=-5,vmax=25,shading="gouraud",alpha=0.75)
            # Overlay service-specific nodes only
            for node in svc_nodes:
                nid=node[0]; x,y=SIM.pos[nid]
                sv_=SIM.sinr_hist[nid][-1] if SIM.sinr_hist[nid] else 10
                st_=NODE_STYLE.get(node[1],NODE_STYLE["sensor"])
                ax_s.scatter(x,y,c=st_["color"],marker=st_["marker"],
                             s=55,zorder=6,edgecolors="white",linewidths=0.5)
                ax_s.text(x,y+H*0.05,f"{sv_:.0f}",fontsize=5,
                          ha="center",color=TEXT_COL,zorder=7)
            # gNBs
            for gx,gy,glbl,*_ in cfg["gnbs"]:
                ax_s.plot(gx,gy,"^w",ms=6,zorder=8)
        else:
            ax_s.text(0.5,0.5,f"No {svc}\nnodes here",ha="center",va="center",
                      fontsize=7,color=MUTED,transform=ax_s.transAxes)
        ax_s.set_xlim(0,W); ax_s.set_ylim(0,H)
        ax_s.set_aspect("equal","box")
        ax_s.set_xticks([]); ax_s.set_yticks([])
        # Color border = service color
        for spine in ax_s.spines.values():
            spine.set_edgecolor(sc); spine.set_linewidth(1.2)

    # ── 11. AGGREGATE THROUGHPUT ─────────────────────────────────────────────
    ax_agg.cla(); ax_agg.set_facecolor(PANEL_BG)
    ax_agg.set_title(
        f"System Throughput (Gbps)  ·  Handovers: {SIM.handovers}",
        fontsize=8,pad=3)
    ax_agg.grid(color=GRID_COL,lw=0.3)
    if len(SIM.total_tp)>2:
        t_a = np.arange(len(SIM.total_tp))*0.04
        tp_a= np.array(SIM.total_tp)
        ax_agg.plot(t_a,tp_a,color=ec,lw=1.3,alpha=0.9,label="Total")
        ax_agg.fill_between(t_a,tp_a,alpha=0.10,color=ec)
        for svc,sc in SVC_COLOR.items():
            svc_ns=[n for n in nodes_def if n[6]==svc and SIM.svc_filter.get(svc,True)]
            if not svc_ns: continue
            svc_tp=[]
            for i in range(len(SIM.total_tp)):
                row=sum(SIM.tp_hist[nd[0]][i] for nd in svc_ns
                        if i<len(SIM.tp_hist[nd[0]]))
                svc_tp.append(row)
            ax_agg.plot(t_a[:len(svc_tp)],svc_tp,color=sc,lw=0.8,
                        ls="--",alpha=0.7,label=svc)
    ax_agg.set_xlabel("Time (s)",color=MUTED); ax_agg.set_ylabel("Gbps",color=MUTED)
    ax_agg.legend(fontsize=6,loc="upper left",ncol=4,framealpha=0.4,
                  facecolor=PANEL_BG,edgecolor=GRID_COL,labelcolor=TEXT_COL)

    # ── 12. SINR HISTOGRAM (rolling) ─────────────────────────────────────────
    ax_hist.cla(); ax_hist.set_facecolor(PANEL_BG)
    ax_hist.set_title("SINR Distribution (rolling)",fontsize=8,pad=3)
    ax_hist.grid(color=GRID_COL,lw=0.3,axis="y")
    _tmp = [SIM.sinr_hist[n[0]] for n in active if SIM.sinr_hist[n[0]]]
    all_sinr = np.concatenate(_tmp) if _tmp else np.array([])
    if len(all_sinr)>10:
        # Per-service stacked histogram
        def _svc_arr(svc):
            parts=[SIM.sinr_hist[n[0]] for n in active
                   if n[6]==svc and SIM.sinr_hist[n[0]]]
            return np.concatenate(parts) if parts else np.array([])
        svc_data = {svc: _svc_arr(svc) for svc in SVC_ORDER}
        bins = np.linspace(-5, 28, 28)
        bottom = np.zeros(len(bins)-1)
        for svc in SVC_ORDER:
            d = svc_data.get(svc, np.array([]))
            if len(d) < 2: continue
            counts, _ = np.histogram(d, bins=bins)
            ax_hist.bar(bins[:-1], counts, width=np.diff(bins),
                        bottom=bottom, color=SVC_COLOR.get(svc, MUTED),
                        alpha=0.75, label=svc, align="edge", edgecolor="none")
            bottom += counts
        # Mean & median lines
        mn = np.mean(all_sinr); md = np.median(all_sinr)
        ax_hist.axvline(mn,color=TEXT_COL,lw=1.2,ls="-",
                        label=f"mean={mn:.1f}")
        ax_hist.axvline(md,color=CYAN,lw=1.0,ls="--",
                        label=f"median={md:.1f}")
        ax_hist.axvline(18,color=GREEN,lw=0.7,ls=":",alpha=0.6)
        ax_hist.axvline(10,color=RED,  lw=0.7,ls=":",alpha=0.6)
    ax_hist.set_xlabel("SINR (dB)",color=MUTED)
    ax_hist.set_ylabel("Count",    color=MUTED)
    ax_hist.legend(fontsize=5.5,ncol=3,loc="upper left",
                   framealpha=0.4,facecolor=PANEL_BG,edgecolor=GRID_COL,
                   labelcolor=TEXT_COL)
    ax_hist.set_xlim(-5,28)

    # ── Status ───────────────────────────────────────────────────────────────
    if not SIM.paused:
        txt_status.set_text(f"t={SIM.t:.1f}s ▶ {SIM.env_name}")
        txt_status.set_color(GREEN)
    return []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KEYBOARD SHORTCUTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def on_key(ev):
    if ev.key==" ":          on_pause(None)
    elif ev.key=="r":        SIM.reset()
    elif ev.key=="e":        SIM.export_csv()
    elif ev.key in [str(i) for i in range(1,7)]:
        idx=int(ev.key)-1
        if idx<len(ENV_NAMES):
            radio_env.set_active(idx); on_env(ENV_NAMES[idx])
    elif ev.key in ("+","="): sl_spd.set_val(min(6.0,SIM.speed+0.5))
    elif ev.key=="-":         sl_spd.set_val(max(0.1,SIM.speed-0.5))
    elif ev.key=="h":         on_disp("Heatmap")
    elif ev.key=="t":         on_disp("Trails")
    elif ev.key=="l":         on_disp("Links")
fig.canvas.mpl_connect("key_press_event",on_key)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STARTUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n"+"="*68)
print("  3GPP 6G THz  ·  SINR Radar + Heatmaps  ·  Live Simulation")
print("="*68)
print("\n  Pre-computing THz heatmaps …")
for ename in ENV_NAMES:
    if ename not in SIM.heatmaps:
        print(f"    [{ename}]…",end="",flush=True)
        SIM.heatmaps[ename]=build_heatmap(ENVIRONMENTS[ename])
        print(" done")
print("\n  Controls:")
print("    SPACE / R / E     pause | reset | export CSV")
print("    1–6               switch use case")
print("    + / -             speed up / down")
print("    H / T / L         heatmap | trails | links")
print("\n  Panels:")
print("    Topology + trails + links | SINR contourf heatmap")
print("    Polar SINR map            | KPI Radar (spider chart)")
print("    Live SINR bars            | TP / Latency timelines")
print("    SINR time series + CDF    | Per-service SINR heatmaps (×5)")
print("    Aggregate TP              | Rolling SINR histogram")
print("\n  Launching …\n")

anim = FuncAnimation(fig, draw_frame, interval=60,
                     blit=False, cache_frame_data=False)
plt.show()