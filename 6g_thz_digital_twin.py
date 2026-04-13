#!/usr/bin/env python3
"""
================================================================================
  6G THz DIGITAL TWIN  —  Real-World ↔ Virtual Network Synchronisation
================================================================================
  A Digital Twin (DT) mirrors a physical network in real time.
  This simulation shows:

    PHYSICAL WORLD  ──────── sync ──────────►  DIGITAL TWIN
    (real 6G nodes)         (latency)          (virtual replica)

  Twin Concept:
    • Physical Asset  = the actual device (car, robot, drone …)
    • Digital Replica = a virtual model updated from live sensor data
    • Sync Latency    = how long it takes to update the twin (ms)
    • Twin Fidelity   = how accurately the twin reflects reality (%)
    • Prediction Engine = DT can predict future state from physics
    • Anomaly Detector  = DT flags when real ≠ twin (fault detection)

  6 Real-Life Use Cases:
    1. XR Surgery        300 GHz  800 Gbps   <0.8 ms  — holographic OR
    2. Auto Factory      140 GHz  200 Gbps   <1.5 ms  — AGVs + robots
    3. V2X Crossroad     300 GHz  400 Gbps   <0.3 ms  — zero-accident driving
    4. THz Backhaul        1 THz 1800 Gbps   <0.1 ms  — rooftop P2P
    5. Tunnel Rescue     100 GHz   80 Gbps   <3.0 ms  — SAR operations
    6. Holo Classroom    300 GHz  300 Gbps   <2.0 ms  — tele-education

  Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │   TITLE + DT STATUS BANNER                                  │
    ├────────────────┬────────────────┬────────────────────────────┤
    │  PHYSICAL      │  DIGITAL TWIN  │  SYNC STATUS + KPI GAUGES  │
    │  WORLD MAP     │  REPLICA MAP   │  SINR RADAR  FIDELITY BAR  │
    ├────────────────┴────────────────┤────────────────────────────┤
    │  SINR HISTORY (twin vs real)    │  THROUGHPUT TIMELINE       │
    ├─────────────────────────────────┴────────────────────────────┤
    │  DT PREDICTION ENGINE  |  ANOMALY PANEL  |  HEATMAP DIFF    │
    ├───────────────────────────────────────────────────────────────┤
    │  CONTROL STRIP  (environment · service · speed · buttons)    │
    └───────────────────────────────────────────────────────────────┘

  Controls: SPACE pause | R reset | 1-6 switch env | +/- speed
            H heatmap   | A anomaly inject          | Q quit
================================================================================
  Run : python 6g_thz_digital_twin.py
  Deps: pip install matplotlib numpy scipy
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.widgets import Button, RadioButtons, Slider, CheckButtons
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter
import warnings, time, sys
warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COLOUR PALETTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG       = "#050A12"
PBG      = "#0D1520"      # physical world panel
TBG      = "#0A1530"      # digital twin panel (slightly bluer)
GC       = "#1A2540"
TC       = "#E8F0FF"
MT       = "#5A7080"
BLUE     = "#2D7CF6"
GREEN    = "#1FD16A"
RED      = "#F04545"
ORANGE   = "#E89010"
PURPLE   = "#9B6BF0"
TEAL     = "#10C0B0"
CYAN     = "#08C8E8"
PINK     = "#F060B0"
YELLOW   = "#E8C010"
LIME     = "#80F030"

# Digital Twin accent colors
DT_BLUE  = "#00BFFF"    # twin replica color
DT_GREEN = "#00FF9F"    # synced / healthy
DT_AMBER = "#FFB800"    # slight drift
DT_RED   = "#FF3030"    # anomaly / desync

SVC_C = {"URLLC":RED,"eMBB":BLUE,"mMTC":GREEN,"XR":PURPLE,"V2X":ORANGE}

NS = {
    "surgeon"   :{"c":RED,   "m":"*","s":180,"l":"Surgeon"},
    "robot"     :{"c":PURPLE,"m":"h","s":150,"l":"Robot"},
    "sensor"    :{"c":TEAL,  "m":"P","s": 90,"l":"Sensor"},
    "agv"       :{"c":ORANGE,"m":"D","s":150,"l":"AGV"},
    "car"       :{"c":BLUE,  "m":"^","s":150,"l":"Car"},
    "drone"     :{"c":CYAN,  "m":"v","s":130,"l":"Drone"},
    "backhaul"  :{"c":GREEN, "m":"s","s":160,"l":"Backhaul"},
    "rescuer"   :{"c":RED,   "m":"P","s":130,"l":"Rescuer"},
    "holo_disp" :{"c":PINK,  "m":"8","s":150,"l":"HoloDisp"},
    "student"   :{"c":PURPLE,"m":"o","s": 90,"l":"Student"},
    "camera"    :{"c":TEAL,  "m":"x","s": 90,"l":"Camera"},
    "rsu"       :{"c":YELLOW,"m":"H","s":120,"l":"RSU"},
}
MC = {"wall":"#0F1A25","glass":"#0A1C30","metal":"#201508",
      "concrete":"#101820","free":"#050A12"}
ME = {"wall":"#253545","glass":"#2060A0","metal":"#805020",
      "concrete":"#304050","free":"#1A2540"}

_SINR_CM = LinearSegmentedColormap.from_list("dt_sinr",[
    "#050015","#150050","#300090","#1000C0","#0040F0","#0080F0",
    "#00C0C0","#00E080","#80FF20","#FFFF00","#FFFFFF"],N=512)

_DIFF_CM = LinearSegmentedColormap.from_list("dt_diff",[
    "#00FF00","#80FF00","#FFFF00","#FF8000","#FF0000"],N=256)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  THz PHYSICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_KF = {0.060e12:0.011,0.100e12:0.003,0.140e12:0.003,0.183e12:0.174,
       0.220e12:0.004,0.300e12:0.013,0.340e12:0.069,0.500e12:0.035,
       1.000e12:0.868}

def mol_abs(d,f):
    fs=sorted(_KF); k=_KF[fs[0]]
    if f>=fs[-1]: k=_KF[fs[-1]]
    else:
        for i in range(len(fs)-1):
            if fs[i]<=f<fs[i+1]:
                t=(np.log10(f)-np.log10(fs[i]))/(np.log10(fs[i+1])-np.log10(fs[i]))
                k=_KF[fs[i]]*(1-t)+_KF[fs[i+1]]*t; break
    return k*max(d,0.01)

def friis(d,f): return 20*np.log10(max(4*np.pi*max(d,0.01)*f/3e8,1e-30))

def calc_sinr(nx,ny,gnbs,cfg,rng,spd=0.0,noise_offset=0.0):
    f=cfg["freq_hz"]; tx=cfg["tx_dbm"]; bf=cfg["bf_db"]
    K=cfg.get("K",4.0); noise=cfg["noise_dbm"]+noise_offset
    # Doppler penalty
    dop=max(0,10*np.log10(max(spd*f/(3e8*240e3),1e-9)))
    # NLOS blockage
    nlos=0.0
    for b in cfg["buildings"]:
        if b[0]<nx<b[0]+b[2] and b[1]<ny<b[1]+b[3]: nlos=b[6]*0.5; break
    rxs=[]
    for gx,gy,*_ in gnbs:
        d=max(np.hypot(nx-gx,ny-gy),0.1)
        pl=friis(d,f)+mol_abs(d,f)+cfg["pen"]*0.2
        nu=np.sqrt(K/(K+1)); sg=1/np.sqrt(2*(K+1))
        fade=20*np.log10(max(np.sqrt((nu+sg*rng.normal())**2+(sg*rng.normal())**2),1e-6))
        rxs.append(tx-pl+bf+fade-dop-nlos)
    rxs.sort(reverse=True)
    sig=10**(rxs[0]/10); intf=sum(10**(p/10) for p in rxs[1:]) if len(rxs)>1 else 0
    return 10*np.log10(max(sig/(intf+10**(noise/10)),1e-12))

def thz_tp(sv,bw,et="indoor"):
    eta={"indoor":0.62,"outdoor":0.58,"vehicular":0.52,"tunnel":0.48,"p2p":0.70}.get(et,0.60)
    return bw*min(np.log2(1+10**(sv/10))*eta,12.0)

def lat_ms(sv,cfg):
    h=0 if sv>=20 else 1 if sv>=12 else 2 if sv>=5 else 3
    return max(0.01,cfg["lat_ms"]*0.4*np.exp(-sv/18)+h*0.125+0.003)

def sinr_col(v):
    if v>=22: return GREEN
    if v>=15: return "#60EF90"
    if v>=10: return ORANGE
    if v>=4:  return "#F08020"
    return RED

def build_heatmap(cfg,res=36):
    W,H=cfg["area"]; rng=np.random.default_rng(0)
    xs=np.linspace(0,W,res); ys=np.linspace(0,H,res)
    XX,YY=np.meshgrid(xs,ys); G=np.zeros((res,res))
    for i in range(res):
        for j in range(res): G[i,j]=calc_sinr(XX[i,j],YY[i,j],cfg["gnbs"],cfg,rng)
    return XX,YY,gaussian_filter(G,sigma=1.5)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENVIRONMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENV = {
  "XR Surgery":{
    "label":"Holographic Remote Surgery · OR Room · 300 GHz",
    "color":RED,"env_t":"indoor","area":(12,10),
    "freq_hz":300e9,"bw_ghz":100,"tx_dbm":20,"bf_db":36,
    "noise_dbm":-80,"lat_ms":0.8,"K":8.0,"pen":8,"peak_gbps":800,
    "buildings":[(0,0,12,10,"OR","wall",28),(0.5,0.5,4,2,"Instruments","metal",35),(7.5,0.5,4,2,"Imaging","metal",35)],
    "gnbs":[(6.0,9.6,"AP",2.5)],
    "nodes":[("Sg1","surgeon",3.0,5.2,0.00,0.00,"URLLC",False,0.0),
             ("Sg2","surgeon",9.0,5.2,0.00,0.00,"URLLC",False,0.0),
             ("RbL","robot",4.5,4.5,0.012,0.004,"XR",True,0.05),
             ("RbR","robot",7.5,4.5,-0.012,0.004,"XR",True,0.05),
             ("Cm1","camera",1.5,8.8,0.00,0.00,"URLLC",False,0.0),
             ("Hlo","holo_disp",6.0,7.8,0.00,0.00,"XR",False,0.0)],
  },
  "Auto Factory":{
    "label":"Industry 6.0 Smart Factory · 140 GHz · AGVs+Robots",
    "color":ORANGE,"env_t":"indoor","area":(80,60),
    "freq_hz":140e9,"bw_ghz":50,"tx_dbm":30,"bf_db":28,
    "noise_dbm":-85,"lat_ms":1.5,"K":3.0,"pen":25,"peak_gbps":200,
    "buildings":[(0,0,80,60,"Factory","metal",32),(4,4,22,16,"LineA","metal",30),
                 (30,4,20,16,"LineB","metal",30),(56,4,20,16,"LineC","metal",30),
                 (4,40,32,16,"WH","concrete",22),(46,40,30,16,"Ctrl","wall",20)],
    "gnbs":[(20,30,"F0",8),(60,30,"F1",8),(40,8,"F2",8)],
    "nodes":[("AG1","agv",10,30,1.0,0.0,"URLLC",False,1.0),
             ("AG2","agv",42,30,-0.9,0.0,"URLLC",False,0.9),
             ("AG3","agv",68,18,0.0,0.8,"URLLC",False,0.8),
             ("Ar1","robot",15,12,0.0,0.0,"URLLC",False,0.0),
             ("Ar2","robot",40,12,0.0,0.0,"URLLC",False,0.0),
             ("S1","sensor",8,48,0.0,0.0,"mMTC",False,0.0),
             ("S2","sensor",28,48,0.0,0.0,"mMTC",False,0.0),
             ("S3","sensor",55,48,0.0,0.0,"mMTC",False,0.0)],
  },
  "V2X Crossroad":{
    "label":"Zero-Accident Autonomous V2X · 300 GHz · <0.3 ms",
    "color":CYAN,"env_t":"outdoor","area":(100,100),
    "freq_hz":300e9,"bw_ghz":80,"tx_dbm":38,"bf_db":30,
    "noise_dbm":-82,"lat_ms":0.3,"K":2.0,"pen":12,"peak_gbps":400,
    "buildings":[(0,0,38,38,"NW","concrete",22),(62,0,38,38,"NE","concrete",22),
                 (0,62,38,38,"SW","concrete",22),(62,62,38,38,"SE","concrete",22)],
    "gnbs":[(50,50,"V2X",12),(18,50,"RW",6),(82,50,"RE",6),(50,18,"RN",6),(50,82,"RS",6)],
    "nodes":[("C1","car",4,50,3.5,0.0,"V2X",False,3.5),
             ("C2","car",96,50,-3.2,0.0,"V2X",False,3.2),
             ("C3","car",50,4,0.0,3.3,"V2X",False,3.3),
             ("C4","car",50,96,0.0,-3.0,"V2X",False,3.0),
             ("Dr1","drone",32,28,0.18,0.12,"URLLC",True,0.3),
             ("Dr2","drone",68,72,-0.15,0.10,"URLLC",True,0.25),
             ("Cm1","camera",38,38,0.0,0.0,"URLLC",False,0.0)],
  },
  "THz Backhaul":{
    "label":"1 THz Rooftop P2P Backhaul · 300 GHz BW · 1.8 Tbps",
    "color":GREEN,"env_t":"p2p","area":(500,100),
    "freq_hz":1.00e12,"bw_ghz":300,"tx_dbm":48,"bf_db":44,
    "noise_dbm":-74,"lat_ms":0.1,"K":20.0,"pen":4,"peak_gbps":1800,
    "buildings":[(0,30,35,40,"BldA","concrete",22),(232,30,36,40,"Relay","concrete",22),(465,30,35,40,"BldB","concrete",22)],
    "gnbs":[(17,68,"TX",18),(250,68,"Rly",18),(483,68,"RX",18)],
    "nodes":[("BH0","backhaul",17,68,0.0,0.0,"eMBB",False,0.0),
             ("BH1","backhaul",250,68,0.0,0.0,"eMBB",False,0.0),
             ("BH2","backhaul",483,68,0.0,0.0,"eMBB",False,0.0),
             ("Dr1","drone",90,55,2.0,0.10,"eMBB",False,2.0),
             ("Dr2","drone",310,72,-1.8,0.08,"eMBB",False,1.8)],
  },
  "Tunnel Rescue":{
    "label":"Emergency SAR in Tunnel · 100 GHz · URLLC Critical",
    "color":ORANGE,"env_t":"tunnel","area":(150,15),
    "freq_hz":100e9,"bw_ghz":30,"tx_dbm":27,"bf_db":20,
    "noise_dbm":-90,"lat_ms":3.0,"K":1.5,"pen":28,"peak_gbps":80,
    "buildings":[(0,0,150,15,"Tunnel","metal",32),(0,0,150,2,"Floor","concrete",20),
                 (0,13,150,2,"Ceiling","concrete",20),(58,2,12,11,"Debris","concrete",30)],
    "gnbs":[(5,7.5,"R0",2),(75,7.5,"R1",2),(145,7.5,"R2",2)],
    "nodes":[("Rs1","rescuer",8,7.5,0.55,0.0,"URLLC",False,0.55),
             ("Rs2","rescuer",26,7.5,0.5,0.0,"URLLC",False,0.50),
             ("Rs3","rescuer",46,7.5,0.45,0.0,"URLLC",False,0.45),
             ("Dr1","drone",16,11,0.65,0.0,"URLLC",False,0.65),
             ("S1","sensor",72,7.5,0.0,0.0,"mMTC",False,0.0)],
  },
  "Holo Classroom":{
    "label":"Holographic Tele-Education · 300 GHz · 10 Gbps/user",
    "color":PURPLE,"env_t":"indoor","area":(20,15),
    "freq_hz":300e9,"bw_ghz":60,"tx_dbm":22,"bf_db":28,
    "noise_dbm":-82,"lat_ms":2.0,"K":6.0,"pen":10,"peak_gbps":300,
    "buildings":[(0,0,20,15,"Room","wall",20),(0,0,2,15,"W","concrete",22),
                 (18,0,2,15,"E","concrete",22),(1,11,18,3,"Pres","wall",20)],
    "gnbs":[(10,13.8,"AP0",2),(3,5.5,"AP1",2),(17,5.5,"AP2",2)],
    "nodes":[("St1","student",4,3,0.04,0.03,"eMBB",True,0.05),
             ("St2","student",8,3,-0.03,0.04,"eMBB",True,0.05),
             ("St3","student",12,3,0.05,-0.02,"eMBB",True,0.06),
             ("St4","student",16,3,-0.04,0.03,"eMBB",True,0.05),
             ("St5","student",4,7,0.03,0.04,"eMBB",True,0.05),
             ("St6","student",8,7,-0.04,-0.03,"eMBB",True,0.05),
             ("Hl1","holo_disp",5,12,0.0,0.0,"XR",False,0.0),
             ("Hl2","holo_disp",10,12,0.0,0.0,"XR",False,0.0)],
  },
}
EN = list(ENV.keys())
HIST = 300

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DIGITAL TWIN STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DigitalTwin:
    """
    Represents both the Physical World and its Digital Twin replica.

    Physical World:
      - Real node positions and SINR (with actual motion + physics)

    Digital Twin:
      - Delayed/predicted replica of physical world
      - Adds sync_latency delay to position updates
      - Twin noise models drift from physical
      - Fidelity score measures how close twin is to reality
      - Anomaly injection: deliberately desyncs a node
    """
    def __init__(self):
        self.ename    = "XR Surgery"
        self.paused   = False
        self.speed    = 1.0
        self.show_hm  = True
        self.inject_anomaly = False
        self.anomaly_node   = None
        self.anomaly_timer  = 0
        self.heatmaps = {}
        self.t        = 0.0
        self.frame    = 0
        self.reset()

    @property
    def cfg(self): return ENV[self.ename]

    def reset(self):
        cfg = self.cfg
        self.rng   = np.random.default_rng(42)
        self.rng_t = np.random.default_rng(99)   # twin has different noise seed

        nodes = cfg["nodes"]
        # Physical world state
        self.phys_pos  = {n[0]:[float(n[2]),float(n[3])] for n in nodes}
        self.phys_vel  = {n[0]:[n[4]*self.speed,n[5]*self.speed] for n in nodes}
        self.phys_bnc  = {n[0]:n[7] for n in nodes}
        self.phys_spd  = {n[0]:n[8] for n in nodes}
        self.phys_sinr = {n[0]:[] for n in nodes}
        self.phys_tp   = {n[0]:[] for n in nodes}
        self.phys_lat  = {n[0]:[] for n in nodes}
        self.phys_trail_x = {n[0]:[] for n in nodes}
        self.phys_trail_y = {n[0]:[] for n in nodes}

        # Digital Twin state (slightly delayed + independent noise)
        self.twin_pos  = {n[0]:[float(n[2]),float(n[3])] for n in nodes}
        self.twin_vel  = {n[0]:[n[4]*self.speed,n[5]*self.speed] for n in nodes}
        self.twin_sinr = {n[0]:[] for n in nodes}
        self.twin_tp   = {n[0]:[] for n in nodes}
        self.twin_lat  = {n[0]:[] for n in nodes}
        self.twin_trail_x = {n[0]:[] for n in nodes}
        self.twin_trail_y = {n[0]:[] for n in nodes}

        # Twin quality metrics
        self.sync_latency  = {n[0]:cfg["lat_ms"]*0.5 for n in nodes}   # ms
        self.fidelity      = {n[0]:100.0 for n in nodes}               # %
        self.prediction_err= {n[0]:0.0 for n in nodes}                 # m
        self.anomaly_flag  = {n[0]:False for n in nodes}

        self.total_phys_tp = []
        self.total_twin_tp = []
        self.handovers     = 0
        self.prev_cell     = {n[0]:None for n in nodes}
        self.t = 0.0; self.frame = 0
        self.inject_anomaly = False; self.anomaly_node = None; self.anomaly_timer = 0

        if self.ename not in self.heatmaps:
            print(f"  [HM] {self.ename}…",end="",flush=True)
            self.heatmaps[self.ename] = build_heatmap(cfg)
            print(" ✓")

    def step(self, dt=0.04):
        if self.paused: return
        cfg = self.cfg; W,H = cfg["area"]
        self.t += dt*self.speed; self.frame += 1
        tot_phys=0.0; tot_twin=0.0

        for node in cfg["nodes"]:
            nid = node[0]
            # ── Physical world movement ──────────────────────────────────────
            x,y   = self.phys_pos[nid]; vx,vy = self.phys_vel[nid]
            sp    = self.phys_spd[nid]*self.speed
            x+=vx*dt; y+=vy*dt
            if self.phys_bnc[nid]:
                if x<=0 or x>=W: vx*=-1; x=np.clip(x,0.1,W-0.1)
                if y<=0 or y>=H: vy*=-1; y=np.clip(y,0.1,H-0.1)
            else:
                if vx>0 and x>W+3: x=-3
                if vx<0 and x<-3:  x=W+3
                if vy>0 and y>H+3: y=-3
                if vy<0 and y<-3:  y=H+3
                x=np.clip(x,0.1,W-0.1); y=np.clip(y,0.1,H-0.1)
            self.phys_pos[nid]=[x,y]; self.phys_vel[nid]=[vx,vy]

            # Physical SINR
            p_sv = calc_sinr(x,y,cfg["gnbs"],cfg,self.rng,sp)
            p_tp = thz_tp(p_sv,cfg["bw_ghz"],cfg["env_t"])
            p_lt = lat_ms(p_sv,cfg)

            for h,v,mx in [(self.phys_sinr[nid],p_sv,HIST),
                            (self.phys_tp[nid],p_tp,HIST),
                            (self.phys_lat[nid],p_lt,HIST)]:
                h.append(v); (h.pop(0) if len(h)>mx else None)
            tot_phys += p_tp

            self.phys_trail_x[nid].append(x); self.phys_trail_y[nid].append(y)
            if len(self.phys_trail_x[nid])>70: self.phys_trail_x[nid].pop(0); self.phys_trail_y[nid].pop(0)

            # Handover
            gnbs=cfg["gnbs"]
            cell=int(np.argmin([np.hypot(x-g[0],y-g[1]) for g in gnbs]))
            if self.prev_cell[nid] is not None and self.prev_cell[nid]!=cell: self.handovers+=1
            self.prev_cell[nid]=cell

            # ── Digital Twin update ──────────────────────────────────────────
            # Twin follows physical with small lag + independent noise
            tx_,ty_ = self.twin_pos[nid]
            tvx,tvy = self.twin_vel[nid]

            if self.inject_anomaly and nid==self.anomaly_node:
                # Anomaly: twin position drifts away from reality
                self.anomaly_timer -= 1
                if self.anomaly_timer > 0:
                    tx_ += tvx*dt*1.8   # twin moves faster than reality
                    ty_ += tvy*dt*1.8
                    if self.phys_bnc[nid]:
                        if tx_<=0 or tx_>=W: tvx*=-1; tx_=np.clip(tx_,0.1,W-0.1)
                        if ty_<=0 or ty_>=H: tvy*=-1; ty_=np.clip(ty_,0.1,H-0.1)
                    self.anomaly_flag[nid] = True
                else:
                    self.inject_anomaly=False; self.anomaly_node=None; self.anomaly_flag[nid]=False
            else:
                # Normal twin: tracks physical with slight delay
                sync_factor = 0.12    # how fast twin catches up
                tx_ = tx_ + (x - tx_)*sync_factor + self.rng_t.normal(0,0.05)
                ty_ = ty_ + (y - ty_)*sync_factor + self.rng_t.normal(0,0.05)
                tx_=np.clip(tx_,0.1,W-0.1); ty_=np.clip(ty_,0.1,H-0.1)
                tvx,tvy = vx,vy
                self.anomaly_flag[nid] = False

            self.twin_pos[nid]=[tx_,ty_]; self.twin_vel[nid]=[tvx,tvy]

            # Twin SINR — computed at twin position
            t_sv = calc_sinr(tx_,ty_,cfg["gnbs"],cfg,self.rng_t,sp,noise_offset=self.rng_t.normal(0,0.5))
            t_tp = thz_tp(t_sv,cfg["bw_ghz"],cfg["env_t"])
            t_lt = lat_ms(t_sv,cfg)

            for h,v,mx in [(self.twin_sinr[nid],t_sv,HIST),
                            (self.twin_tp[nid],t_tp,HIST),
                            (self.twin_lat[nid],t_lt,HIST)]:
                h.append(v); (h.pop(0) if len(h)>mx else None)
            tot_twin += t_tp

            self.twin_trail_x[nid].append(tx_); self.twin_trail_y[nid].append(ty_)
            if len(self.twin_trail_x[nid])>70: self.twin_trail_x[nid].pop(0); self.twin_trail_y[nid].pop(0)

            # Fidelity = 100 - position error normalised to area diagonal
            pos_err = np.hypot(x-tx_, y-ty_)
            diag    = np.hypot(W,H)
            fid     = max(0.0, 100.0 - (pos_err/diag)*100.0*8)
            sinr_err= abs(p_sv-t_sv)
            fid     = max(0.0, fid - sinr_err*0.5)
            self.fidelity[nid]       = 0.92*self.fidelity[nid]+0.08*fid
            self.prediction_err[nid] = 0.9*self.prediction_err[nid]+0.1*pos_err
            sync_ms = pos_err/max(sp,0.01)*1000 if sp>0 else cfg["lat_ms"]*0.2
            self.sync_latency[nid]   = max(0.01,0.9*self.sync_latency[nid]+0.1*sync_ms)

        self.total_phys_tp.append(tot_phys)
        self.total_twin_tp.append(tot_twin)
        if len(self.total_phys_tp)>HIST: self.total_phys_tp.pop(0)
        if len(self.total_twin_tp)>HIST: self.total_twin_tp.pop(0)

    def inject(self):
        """Inject an anomaly into a random moving node."""
        cfg=self.cfg
        moving=[n[0] for n in cfg["nodes"] if n[4]!=0 or n[5]!=0]
        if moving:
            self.anomaly_node=np.random.choice(moving)
            self.anomaly_timer=60
            self.inject_anomaly=True
            print(f"\n  ⚠ ANOMALY injected → {self.anomaly_node}")

DT = DigitalTwin()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RADAR KPI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RL=["SINR","Rate","Latency","Coverage","Fidelity","Reliability"]

def avg_kpi(cfg,dt):
    nodes=cfg["nodes"]; W,H=cfg["area"]
    svs=[dt.phys_sinr[n[0]][-1] for n in nodes if dt.phys_sinr[n[0]]]
    tps=[dt.phys_tp[n[0]][-1]   for n in nodes if dt.phys_tp[n[0]]]
    lts=[dt.phys_lat[n[0]][-1]  for n in nodes if dt.phys_lat[n[0]]]
    fds=[dt.fidelity[n[0]]      for n in nodes]
    sv=np.mean(svs) if svs else 5.0; tp=np.mean(tps) if tps else 0.0
    lt=np.mean(lts) if lts else cfg["lat_ms"]; fd=np.mean(fds)
    x_,y_=np.mean([dt.phys_pos[n[0]][0] for n in nodes]),np.mean([dt.phys_pos[n[0]][1] for n in nodes])
    dm=min(np.hypot(x_-g[0],y_-g[1]) for g in cfg["gnbs"])
    cov=max(0,1-dm/(0.5*np.hypot(W,H)))
    return [min(1,max(0,(sv+5)/30)),
            min(1,tp/max(cfg["peak_gbps"],1)),
            min(1,max(0,1-lt/(cfg["lat_ms"]*4))),
            cov, fd/100.0, min(1,max(0,(sv+5)/30))]

def draw_radar(ax,vals,color,title=""):
    N=len(RL); angs=np.linspace(0,2*np.pi,N,endpoint=False)
    v=vals+[vals[0]]; a=angs.tolist()+[angs[0]]
    ax.set_facecolor(TBG); ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    for r in [0.25,0.5,0.75,1.0]: ax.plot(np.linspace(0,2*np.pi,200),[r]*200,color=GC,lw=0.5,ls=":")
    for ang in angs: ax.plot([ang,ang],[0,1],color=GC,lw=0.4)
    ax.plot(a,v,color=color,lw=2.0,zorder=5); ax.fill(a,v,color=color,alpha=0.20,zorder=4)
    for ang,val in zip(angs,vals): ax.scatter([ang],[val],c=color,s=35,zorder=6,edgecolors="white",lw=0.5)
    ax.set_xticks(angs); ax.set_xticklabels(RL,fontsize=6,color=TC)
    ax.set_yticks([0.5,1.0]); ax.set_yticklabels(["50%","100%"],fontsize=4.5,color=MT)
    ax.set_ylim(0,1.05); ax.spines["polar"].set_color(GC)
    if title: ax.set_title(title,fontsize=8,color=color,pad=8,fontweight="bold")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE LAYOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":8,
    "axes.facecolor":PBG,"figure.facecolor":BG,
    "axes.edgecolor":GC,"axes.labelcolor":TC,
    "xtick.color":MT,"ytick.color":MT,
    "text.color":TC,"axes.titlecolor":TC,
    "axes.grid":True,"grid.color":GC,"grid.linewidth":0.3,
    "axes.spines.top":False,"axes.spines.right":False,
})

fig = plt.figure(figsize=(24,14), facecolor=BG)
try: fig.canvas.manager.set_window_title("6G THz Digital Twin Simulator")
except: pass

# ── Outer 3-row grid ─────────────────────────────────────────────────────────
outer = gridspec.GridSpec(4,1,figure=fig,
    left=0.01,right=0.99,top=0.93,bottom=0.10,
    hspace=0.50,height_ratios=[3.2,2.0,2.0,0.0])

# Row 0: Physical | Twin | Right panel (KPI + radar)
gs0=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[0],wspace=0.30,width_ratios=[2,2,1.4])
ax_phys  = fig.add_subplot(gs0[0])       # physical world
ax_twin  = fig.add_subplot(gs0[1])       # digital twin replica
gs0r=gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs0[2],wspace=0.35,hspace=0.50)
ax_radar = fig.add_subplot(gs0r[0,0],projection="polar")
ax_kpi   = fig.add_subplot(gs0r[0,1]); ax_kpi.axis("off")
ax_sync  = fig.add_subplot(gs0r[1,0])
ax_fid   = fig.add_subplot(gs0r[1,1])

# Row 1: SINR comparison | TP comparison
gs1=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[1],wspace=0.38)
ax_sinr_cmp = fig.add_subplot(gs1[0])
ax_tp_cmp   = fig.add_subplot(gs1[1])
ax_lat_cmp  = fig.add_subplot(gs1[2])

# Row 2: SINR bars | Prediction | Anomaly | Heatmap diff
gs2=gridspec.GridSpecFromSubplotSpec(1,4,subplot_spec=outer[2],wspace=0.38)
ax_bars  = fig.add_subplot(gs2[0])
ax_pred  = fig.add_subplot(gs2[1])
ax_anom  = fig.add_subplot(gs2[2])
ax_hdiff = fig.add_subplot(gs2[3])

# ── Control strip ─────────────────────────────────────────────────────────────
gs_c=gridspec.GridSpec(1,6,figure=fig,left=0.01,right=0.99,bottom=0.01,top=0.09,wspace=0.25)
axr=fig.add_subplot(gs_c[0,0]); axsp=fig.add_subplot(gs_c[0,3])
axl=fig.add_subplot(gs_c[0,4]); axl.axis("off")
axi=fig.add_subplot(gs_c[0,5]); axi.axis("off")
for ax in [axr]: ax.set_facecolor(PBG)

# ── Title + DT banner ─────────────────────────────────────────────────────────
fig.text(0.5,0.970,"6G THz Digital Twin Simulator  ·  Physical World ↔ Virtual Replica",
    ha="center",fontsize=14,fontweight="bold",color=TC)
fig.text(0.5,0.953,
    "Real-time synchronisation  ·  Fidelity scoring  ·  Anomaly detection  ·  Prediction engine  ·  100 GHz – 1 THz",
    ha="center",fontsize=8,color=MT)

txt_hud=fig.text(0.99,0.970,"t=0.0s ▶",ha="right",fontsize=9,fontweight="bold",color=DT_GREEN)

# Left/Right labels for physical vs twin
fig.text(0.115,0.892,"⬛ PHYSICAL WORLD",fontsize=9,fontweight="bold",color=TC,ha="center")
fig.text(0.385,0.892,"🔷 DIGITAL TWIN",fontsize=9,fontweight="bold",color=DT_BLUE,ha="center")

# ── SYNC arrow (center ornament) ──────────────────────────────────────────────
txt_sync_arrow = fig.text(0.252,0.87,"◄── SYNC ──►",ha="center",fontsize=8,
    color=DT_GREEN,fontweight="bold",style="italic")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WIDGETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
axr.set_title("Use Case",fontsize=8,color=TC,pad=1)
radio=RadioButtons(axr,EN,activecolor=BLUE)
for l in radio.labels: l.set_color(TC); l.set_fontsize(7)
def on_env(lb): DT.ename=lb; DT.reset()
radio.on_clicked(on_env)

axsp.set_title("Speed",fontsize=8,color=TC,pad=1)
sl=Slider(axsp,"",0.1,5.0,valinit=1.0,color=BLUE,track_color=GC)
sl.label.set_color(TC); sl.valtext.set_color(BLUE)
sl.on_changed(lambda v: setattr(DT,"speed",v))

# Buttons
for pos,lbl,key in [
    ([0.35,0.025,0.055,0.030],"⏸ Pause","pause"),
    ([0.41,0.025,0.055,0.030],"↺ Reset","reset"),
    ([0.47,0.025,0.070,0.030],"⚡ Anomaly","anom"),
]:
    ba=fig.add_axes(pos); btn=Button(ba,lbl,color=PBG,hovercolor=GC)
    btn.label.set_color(TC); btn.label.set_fontsize(7.5)
    if key=="pause":   btn_pause=btn
    elif key=="reset": btn_reset=btn
    else:              btn_anom=btn

def on_pause(ev):
    DT.paused=not DT.paused
    btn_pause.label.set_text("▶ Resume" if DT.paused else "⏸ Pause")
    txt_hud.set_color(ORANGE if DT.paused else DT_GREEN)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(lambda ev: DT.reset())
btn_anom.on_clicked(lambda ev: DT.inject())

# Legend
axl.set_title("Node Legend",fontsize=8,color=TC,pad=1)
yl=0.97
for nt,st in list(NS.items())[:7]:
    axl.plot(0.04,yl,marker=st["m"],color=st["c"],markersize=6,transform=axl.transAxes,clip_on=False)
    axl.text(0.12,yl,st["l"],fontsize=6.5,color=TC,va="center",transform=axl.transAxes)
    yl-=0.135

# DT Legend
axi.set_title("Digital Twin Legend",fontsize=8,color=TC,pad=1)
dt_items=[
    ("●",TC,"Physical world node"),("●",DT_BLUE,"Digital twin replica"),
    ("—",DT_GREEN,"Synced (fidelity>90%)"),("—",DT_AMBER,"Drift (70-90%)"),
    ("—",DT_RED,"Anomaly / desync"),("→",CYAN,"SINR trail (physical)"),
    ("→",DT_BLUE,"SINR trail (twin)"),
]
yl=0.96
for sym,col,lbl in dt_items:
    axi.text(0.04,yl,sym,fontsize=8,color=col,transform=axi.transAxes,va="center",fontweight="bold")
    axi.text(0.15,yl,lbl,fontsize=6.5,color=TC,transform=axi.transAxes,va="center")
    yl-=0.13

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER: draw a 2D map (physical or twin)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
norm_s=Normalize(vmin=-10,vmax=28)
cmap_t=plt.get_cmap("tab20")

def draw_map(ax, cfg, pos_dict, vel_dict, sinr_hist_dict,
             trail_x_dict, trail_y_dict, fidelity_dict,
             anomaly_dict, title, is_twin, hm_data, frame_idx):
    W,H=cfg["area"]; nodes=cfg["nodes"]; ec=cfg["color"]
    bg=TBG if is_twin else PBG
    ax.cla(); ax.set_facecolor(bg)
    ax.set_xlim(0,W); ax.set_ylim(0,H); ax.set_aspect("equal","box")
    freq_s=f"{cfg['freq_hz']/1e9:.0f}GHz" if cfg["freq_hz"]<1e12 else f"{cfg['freq_hz']/1e12:.1f}THz"
    ax.set_title(title,fontsize=9,color=DT_BLUE if is_twin else TC,fontweight="bold",pad=3)
    ax.set_xlabel("x (m)",color=MT,fontsize=7); ax.set_ylabel("y (m)",color=MT,fontsize=7)

    # Blue tinted border for twin
    if is_twin:
        for sp in ax.spines.values():
            sp.set_edgecolor(DT_BLUE); sp.set_linewidth(1.8)

    # Heatmap
    if DT.show_hm and DT.ename in hm_data:
        XX,YY,hm=hm_data[DT.ename]
        ax.pcolormesh(XX,YY,hm,cmap=_SINR_CM,vmin=-10,vmax=28,shading="gouraud",
                      alpha=0.28 if is_twin else 0.32)

    # Buildings
    for bld in cfg["buildings"]:
        bx,by,bw_,bh_,blbl,mat,*_=bld
        col=ME.get(mat,"#253545"); fc=MC.get(mat,"#101820")
        ax.add_patch(mpatches.FancyBboxPatch((bx,by),bw_,bh_,boxstyle="round,pad=0.3",
                     lw=0.7,edgecolor=col,facecolor=fc,alpha=0.85))
        ax.text(bx+bw_/2,by+bh_/2,blbl,ha="center",va="center",fontsize=4.5,color=MT)

    # gNBs
    gnbs_xy=[(g[0],g[1]) for g in cfg["gnbs"]]
    for gx,gy,glbl,*_ in cfg["gnbs"]:
        rm=min(W,H)*0.25
        for r,a in [(rm,0.04),(rm*0.6,0.07),(rm*0.3,0.12)]:
            ax.add_patch(plt.Circle((gx,gy),r,color=DT_BLUE if is_twin else ec,alpha=a,lw=0))
        pr=(frame_idx*0.6)%(rm*1.3)+2
        ax.add_patch(plt.Circle((gx,gy),pr,color=DT_BLUE if is_twin else ec,
                     alpha=max(0,0.35-pr/(rm*1.5)),fill=False,lw=1.1))
        ax.plot(gx,gy,"^",color=DT_BLUE if is_twin else ec,ms=11,zorder=8,
                markeredgecolor="white",markeredgewidth=0.9)
        ax.text(gx,gy-H*0.04,glbl,ha="center",fontsize=6,
                color=DT_BLUE if is_twin else ec,fontweight="bold")

    # Nodes
    for ni,node in enumerate(nodes):
        nid=node[0]; x,y=pos_dict[nid]; hist=sinr_hist_dict[nid]
        sv=hist[-1] if hist else 12.0; st=NS.get(node[1],NS["sensor"])
        anom=anomaly_dict.get(nid,False)

        # Determine fidelity color
        fid=fidelity_dict.get(nid,100.0)
        if anom:          fc=DT_RED
        elif fid>=90:     fc=DT_GREEN
        elif fid>=70:     fc=DT_AMBER
        else:             fc=DT_RED

        # Trails
        if len(trail_x_dict[nid])>2:
            tx_arr=trail_x_dict[nid]; ty_arr=trail_y_dict[nid]
            pts=np.array([tx_arr,ty_arr]).T.reshape(-1,1,2)
            segs=np.concatenate([pts[:-1],pts[1:]],axis=1); nseg=len(segs)
            trail_cm=LinearSegmentedColormap.from_list("twin_trail",
                [DT_BLUE+"40",DT_BLUE] if is_twin else ["#00FFFF20",CYAN],N=64)
            lc=LineCollection(segs,cmap=trail_cm,norm=Normalize(0,nseg),lw=1.2,alpha=0.60)
            lc.set_array(np.arange(nseg)); ax.add_collection(lc)

        # Link to nearest gNB
        d_=[np.hypot(x-gx,y-gy) for gx,gy in gnbs_xy]
        bst=gnbs_xy[int(np.argmin(d_))]
        ax.plot([x,bst[0]],[y,bst[1]],color=fc,lw=0.65,alpha=0.55,zorder=2,
                ls="--" if is_twin else "-")

        # Node marker — twin is slightly transparent and outlined in DT_BLUE
        mc=DT_BLUE if is_twin else st["c"]
        ax.scatter(x,y,c=mc,marker=st["m"],s=st["s"],zorder=7,
                   edgecolors=DT_BLUE if is_twin else "white",linewidths=1.2 if is_twin else 0.7,
                   alpha=0.82 if is_twin else 1.0)

        # Fidelity ring on twin
        if is_twin:
            ring_col=fc; r_size=st["s"]*2.2
            ax.scatter(x,y,c="none",s=r_size,zorder=6,
                       edgecolors=ring_col,linewidths=1.5,alpha=0.60)

        # SINR dot
        ax.scatter(x+W*0.013,y+H*0.022,c=sinr_col(sv),s=20,zorder=9,edgecolors="none")

        # Anomaly warning icon
        if anom:
            ax.text(x,y+H*0.07,"⚠",ha="center",fontsize=9,color=DT_RED,zorder=12,
                    fontweight="bold")

        ax.text(x,y+H*0.045,nid,fontsize=5,ha="center",color=TC,zorder=10)

    # Frequency label
    ax.text(0.99,0.02,freq_s,transform=ax.transAxes,fontsize=7,
            color=DT_BLUE if is_twin else ec,ha="right",fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2",fc=bg,ec=GC,alpha=0.8))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DRAW FRAME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_frame(_fn):
    DT.step(dt=0.04)
    cfg   = DT.cfg; W,H = cfg["area"]; ec=cfg["color"]
    nodes = cfg["nodes"]; na=max(len(nodes)-1,1)
    freq_s=f"{cfg['freq_hz']/1e9:.0f} GHz" if cfg["freq_hz"]<1e12 else f"{cfg['freq_hz']/1e12:.1f} THz"

    # Sync status
    all_fid = np.mean([DT.fidelity[n[0]] for n in nodes])
    any_anom = any(DT.anomaly_flag[n[0]] for n in nodes)
    if any_anom:
        sync_col=DT_RED; sync_txt="⚠  ANOMALY DETECTED"
    elif all_fid>=90:
        sync_col=DT_GREEN; sync_txt=f"◄── SYNC  {all_fid:.1f}% ──►"
    elif all_fid>=70:
        sync_col=DT_AMBER; sync_txt=f"◄── DRIFT  {all_fid:.1f}% ──►"
    else:
        sync_col=DT_RED; sync_txt=f"◄── DESYNC  {all_fid:.1f}% ──►"
    txt_sync_arrow.set_text(sync_txt); txt_sync_arrow.set_color(sync_col)

    # ── PHYSICAL MAP ─────────────────────────────────────────────────────────
    draw_map(ax_phys,cfg,DT.phys_pos,DT.phys_vel,DT.phys_sinr,
             DT.phys_trail_x,DT.phys_trail_y,
             DT.fidelity,DT.anomaly_flag,
             f"Physical World  [{freq_s} · {cfg['bw_ghz']} GHz BW · {DT.ename}]",
             False,DT.heatmaps,DT.frame)

    # ── DIGITAL TWIN MAP ──────────────────────────────────────────────────────
    draw_map(ax_twin,cfg,DT.twin_pos,DT.twin_vel,DT.twin_sinr,
             DT.twin_trail_x,DT.twin_trail_y,
             DT.fidelity,DT.anomaly_flag,
             f"Digital Twin Replica  [Fidelity: {all_fid:.1f}%  ·  HO: {DT.handovers}]",
             True,DT.heatmaps,DT.frame)

    # ── KPI RADAR ─────────────────────────────────────────────────────────────
    ax_radar.cla()
    kpi_vals=avg_kpi(cfg,DT)
    draw_radar(ax_radar,kpi_vals,ec,title=f"Avg KPI\n({len(nodes)} nodes)")

    # ── KPI TEXT SCORECARD ────────────────────────────────────────────────────
    ax_kpi.cla(); ax_kpi.axis("off"); ax_kpi.set_facecolor(TBG)
    ax_kpi.set_title("Twin vs Real",fontsize=8,color=TC,pad=2)
    y0=0.92
    for svc,scc in SVC_C.items():
        sn=[n for n in nodes if n[6]==svc]
        if not sn: continue
        p_sv=[DT.phys_sinr[n[0]][-1] for n in sn if DT.phys_sinr[n[0]]]
        t_sv=[DT.twin_sinr[n[0]][-1] for n in sn if DT.twin_sinr[n[0]]]
        if not p_sv: continue
        p_m=np.mean(p_sv); t_m=np.mean(t_sv) if t_sv else p_m
        diff=abs(p_m-t_m); dcol=DT_GREEN if diff<1.5 else DT_AMBER if diff<3 else DT_RED
        ax_kpi.text(0.02,y0,svc,fontsize=7.5,color=scc,fontweight="bold",transform=ax_kpi.transAxes,va="center")
        ax_kpi.text(0.30,y0,f"R:{p_m:.0f}dB",fontsize=7,color=TC,transform=ax_kpi.transAxes,va="center")
        ax_kpi.text(0.60,y0,f"T:{t_m:.0f}dB",fontsize=7,color=DT_BLUE,transform=ax_kpi.transAxes,va="center")
        ax_kpi.text(0.88,y0,f"Δ{diff:.1f}",fontsize=7,color=dcol,transform=ax_kpi.transAxes,va="center",fontweight="bold")
        y0-=0.18

    # ── SYNC LATENCY BARS ────────────────────────────────────────────────────
    ax_sync.cla(); ax_sync.set_facecolor(TBG)
    ax_sync.set_title("Sync Latency (ms)",fontsize=8,pad=2)
    sync_vals=[DT.sync_latency[n[0]] for n in nodes]
    bar_cols=[DT_GREEN if v<=cfg["lat_ms"] else DT_AMBER if v<=cfg["lat_ms"]*2 else DT_RED
              for v in sync_vals]
    bars=ax_sync.bar(range(len(nodes)),sync_vals,color=bar_cols,edgecolor="none",alpha=0.85)
    ax_sync.axhline(cfg["lat_ms"],color=DT_RED,lw=0.9,ls="--",alpha=0.7)
    ax_sync.set_xticks(range(len(nodes)))
    ax_sync.set_xticklabels([n[0] for n in nodes],rotation=45,ha="right",fontsize=5)
    ax_sync.set_ylabel("ms",color=MT,fontsize=6)
    ax_sync.text(0.99,0.96,f"Target<{cfg['lat_ms']}ms",fontsize=6,color=DT_RED,
                 transform=ax_sync.transAxes,ha="right",va="top")

    # ── FIDELITY BARS ────────────────────────────────────────────────────────
    ax_fid.cla(); ax_fid.set_facecolor(TBG)
    ax_fid.set_title("Twin Fidelity (%)",fontsize=8,pad=2)
    fid_vals=[DT.fidelity[n[0]] for n in nodes]
    fid_cols=[DT_GREEN if v>=90 else DT_AMBER if v>=70 else DT_RED for v in fid_vals]
    ax_fid.bar(range(len(nodes)),fid_vals,color=fid_cols,edgecolor="none",alpha=0.85)
    ax_fid.axhline(90,color=DT_GREEN,lw=0.8,ls="--",alpha=0.6)
    ax_fid.axhline(70,color=DT_AMBER,lw=0.8,ls="--",alpha=0.6)
    ax_fid.set_ylim(0,105)
    ax_fid.set_xticks(range(len(nodes)))
    ax_fid.set_xticklabels([n[0] for n in nodes],rotation=45,ha="right",fontsize=5)
    ax_fid.set_ylabel("%",color=MT,fontsize=6)

    # ── SINR COMPARISON ──────────────────────────────────────────────────────
    ax_sinr_cmp.cla(); ax_sinr_cmp.set_facecolor(PBG)
    ax_sinr_cmp.set_title("SINR: Physical (solid) vs Twin (dashed)",fontsize=8.5,pad=3)
    ax_sinr_cmp.grid(color=GC,lw=0.3)
    ax_sinr_cmp.axhspan(22,35,alpha=0.07,color=GREEN); ax_sinr_cmp.axhspan(10,22,alpha=0.07,color=ORANGE)
    ax_sinr_cmp.axhspan(-10,10,alpha=0.07,color=RED)
    for ni,node in enumerate(nodes):
        nid=node[0]; col=cmap_t(ni/na)
        hp=DT.phys_sinr[nid]; ht=DT.twin_sinr[nid]
        if len(hp)>2: ax_sinr_cmp.plot(np.arange(len(hp))*0.04,hp,color=col,lw=1.1,alpha=0.90)
        if len(ht)>2: ax_sinr_cmp.plot(np.arange(len(ht))*0.04,ht,color=col,lw=0.8,ls="--",alpha=0.70)
    ax_sinr_cmp.axhline(22,color=GREEN,lw=0.6,ls="--",alpha=0.4); ax_sinr_cmp.axhline(10,color=RED,lw=0.6,ls="--",alpha=0.4)
    ax_sinr_cmp.set_ylim(-10,32); ax_sinr_cmp.set_xlabel("t (s)",color=MT); ax_sinr_cmp.set_ylabel("dB",color=MT)

    # ── TP COMPARISON ─────────────────────────────────────────────────────────
    ax_tp_cmp.cla(); ax_tp_cmp.set_facecolor(PBG)
    ax_tp_cmp.set_title("Throughput: Physical vs Twin (Gbps)",fontsize=8.5,pad=3)
    ax_tp_cmp.grid(color=GC,lw=0.3)
    if len(DT.total_phys_tp)>2:
        ta=np.arange(len(DT.total_phys_tp))*0.04
        tp_p=np.array(DT.total_phys_tp); tp_t=np.array(DT.total_twin_tp)
        ax_tp_cmp.plot(ta,tp_p,color=TC,lw=1.4,alpha=0.92,label="Physical")
        ax_tp_cmp.fill_between(ta,tp_p,alpha=0.08,color=TC)
        ax_tp_cmp.plot(ta,tp_t,color=DT_BLUE,lw=1.1,ls="--",alpha=0.85,label="Twin")
        ax_tp_cmp.fill_between(ta,tp_t,alpha=0.06,color=DT_BLUE)
        ax_tp_cmp.axhline(cfg["peak_gbps"],color=YELLOW,lw=0.8,ls=":",alpha=0.5)
    ax_tp_cmp.set_xlabel("t (s)",color=MT); ax_tp_cmp.set_ylabel("Gbps",color=MT)
    ax_tp_cmp.legend(fontsize=6,loc="upper left",framealpha=0.3,facecolor=PBG,edgecolor=GC,labelcolor=TC)

    # ── LATENCY COMPARISON ────────────────────────────────────────────────────
    ax_lat_cmp.cla(); ax_lat_cmp.set_facecolor(PBG)
    ax_lat_cmp.set_title("Latency: Physical vs Twin (ms)",fontsize=8.5,pad=3)
    ax_lat_cmp.grid(color=GC,lw=0.3)
    ax_lat_cmp.axhline(cfg["lat_ms"],color=DT_RED,lw=0.9,ls="--",alpha=0.65)
    for ni,node in enumerate(nodes):
        nid=node[0]; col=cmap_t(ni/na)
        hp=DT.phys_lat[nid]; ht=DT.twin_lat[nid]
        if len(hp)>2: ax_lat_cmp.plot(np.arange(len(hp))*0.04,hp,color=col,lw=1.0,alpha=0.88)
        if len(ht)>2: ax_lat_cmp.plot(np.arange(len(ht))*0.04,ht,color=col,lw=0.7,ls="--",alpha=0.65)
    ax_lat_cmp.set_xlabel("t (s)",color=MT); ax_lat_cmp.set_ylabel("ms",color=MT)
    ax_lat_cmp.text(0.98,0.96,f"Target {cfg['lat_ms']}ms",fontsize=6,color=DT_RED,
                    transform=ax_lat_cmp.transAxes,ha="right",va="top")

    # ── LIVE SINR BARS ────────────────────────────────────────────────────────
    ax_bars.cla(); ax_bars.set_facecolor(PBG)
    ax_bars.set_title("Live SINR  Real(█) / Twin(░)",fontsize=8.5,pad=3)
    ax_bars.set_xlim(-10,30)
    for i,node in enumerate(nodes):
        nid=node[0]; svc=node[6]
        p_sv=DT.phys_sinr[nid][-1] if DT.phys_sinr[nid] else 0
        t_sv=DT.twin_sinr[nid][-1] if DT.twin_sinr[nid] else 0
        ax_bars.barh(i+0.18,p_sv,height=0.32,color=sinr_col(p_sv),edgecolor="none",alpha=0.92)
        ax_bars.barh(i-0.18,t_sv,height=0.32,color=DT_BLUE,edgecolor="none",alpha=0.70)
        ax_bars.text(max(p_sv+0.2,0.3),i+0.18,f"{p_sv:.0f}",va="center",fontsize=5.5,color=TC)
        ax_bars.text(max(t_sv+0.2,0.3),i-0.18,f"{t_sv:.0f}",va="center",fontsize=5.5,color=DT_BLUE)
        ax_bars.text(-9.5,i,nid,va="center",fontsize=5.5,color=SVC_C.get(svc,MT),fontweight="bold")
    ax_bars.set_yticks([]); ax_bars.set_xlabel("SINR (dB)",color=MT)
    ax_bars.axvline(22,color=GREEN,lw=0.7,ls="--",alpha=0.5); ax_bars.axvline(10,color=RED,lw=0.7,ls="--",alpha=0.5)
    ax_bars.grid(axis="x",color=GC,lw=0.3)

    # ── PREDICTION ENGINE ────────────────────────────────────────────────────
    ax_pred.cla(); ax_pred.set_facecolor(PBG)
    ax_pred.set_title("Twin Prediction vs Reality",fontsize=8.5,pad=3)
    ax_pred.grid(color=GC,lw=0.3)
    ax_pred.set_xlabel("Node",color=MT); ax_pred.set_ylabel("Position Error (m)",color=MT)
    pred_vals=[DT.prediction_err[n[0]] for n in nodes]
    pred_cols=[DT_GREEN if v<0.5 else DT_AMBER if v<1.5 else DT_RED for v in pred_vals]
    ax_pred.bar(range(len(nodes)),pred_vals,color=pred_cols,alpha=0.85,edgecolor="none")
    ax_pred.axhline(0.5,color=DT_GREEN,lw=0.8,ls="--",alpha=0.6)
    ax_pred.axhline(1.5,color=DT_RED,  lw=0.8,ls="--",alpha=0.6)
    ax_pred.set_xticks(range(len(nodes)))
    ax_pred.set_xticklabels([n[0] for n in nodes],rotation=45,ha="right",fontsize=5.5)
    ax_pred.text(0.99,0.96,"< 0.5m Excellent",fontsize=5.5,color=DT_GREEN,
                 transform=ax_pred.transAxes,ha="right",va="top")

    # ── ANOMALY PANEL ────────────────────────────────────────────────────────
    ax_anom.cla(); ax_anom.set_facecolor(PBG); ax_anom.axis("off")
    ax_anom.set_title("Anomaly & Fault Detection",fontsize=8.5,pad=3)
    anodes=[n[0] for n in nodes if DT.anomaly_flag[n[0]]]
    drift_nodes=[n[0] for n in nodes if DT.fidelity[n[0]]<70 and not DT.anomaly_flag[n[0]]]
    y_=0.92
    # Overall status
    if anodes:
        box_c=DT_RED; stat_txt=f"ANOMALY ACTIVE"; stat_c=DT_RED
    elif drift_nodes:
        box_c=DT_AMBER; stat_txt="DRIFT WARNING"; stat_c=DT_AMBER
    else:
        box_c=DT_GREEN; stat_txt="ALL NOMINAL"; stat_c=DT_GREEN
    ax_anom.text(0.5,y_,stat_txt,ha="center",fontsize=11,color=stat_c,
                 transform=ax_anom.transAxes,fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4",fc=stat_c+"25",ec=stat_c,lw=1.5))
    y_-=0.18
    if anodes:
        ax_anom.text(0.5,y_,f"Anomaly nodes: {', '.join(anodes)}",ha="center",
                     fontsize=8,color=DT_RED,transform=ax_anom.transAxes)
        y_-=0.12
    if drift_nodes:
        ax_anom.text(0.5,y_,f"Drifting: {', '.join(drift_nodes)}",ha="center",
                     fontsize=8,color=DT_AMBER,transform=ax_anom.transAxes)
        y_-=0.12
    ax_anom.text(0.5,y_,f"Avg fidelity: {all_fid:.1f}%",ha="center",
                 fontsize=8,color=DT_GREEN if all_fid>=90 else DT_AMBER,transform=ax_anom.transAxes)
    y_-=0.12
    ax_anom.text(0.5,y_,f"Total HO: {DT.handovers}  t={DT.t:.1f}s",ha="center",
                 fontsize=8,color=MT,transform=ax_anom.transAxes)
    y_-=0.13
    ax_anom.text(0.5,y_,"Press ⚡ Anomaly to inject fault",ha="center",
                 fontsize=7.5,color=MT,style="italic",transform=ax_anom.transAxes)
    y_-=0.12
    ax_anom.text(0.5,y_,f"Env: {DT.ename}",ha="center",
                 fontsize=8,color=ec,fontweight="bold",transform=ax_anom.transAxes)

    # ── HEATMAP DIFFERENCE ───────────────────────────────────────────────────
    ax_hdiff.cla(); ax_hdiff.set_facecolor(PBG)
    ax_hdiff.set_title("SINR Diff: Real − Twin",fontsize=8.5,pad=3)
    if DT.ename in DT.heatmaps:
        XX,YY,hm=DT.heatmaps[DT.ename]
        # Add simulated drift to twin heatmap
        drift_noise=gaussian_filter(DT.rng_t.normal(0,1.5,hm.shape),sigma=2)
        hm_twin=hm+drift_noise
        diff=np.abs(hm-hm_twin)
        cf2=ax_hdiff.contourf(XX,YY,diff,levels=np.linspace(0,5,14),cmap=_DIFF_CM,alpha=0.92)
        ax_hdiff.contour(XX,YY,diff,levels=[1.0,2.5],colors="white",linewidths=0.5,alpha=0.3)
        for gx,gy,glbl,*_ in cfg["gnbs"]: ax_hdiff.plot(gx,gy,"^w",ms=8,zorder=5)
        for node in nodes:
            x,y=DT.phys_pos[node[0]]; tx_,ty_=DT.twin_pos[node[0]]
            ax_hdiff.plot([x,tx_],[y,ty_],color=DT_RED,lw=0.8,alpha=0.6,zorder=6)
            ax_hdiff.scatter(x,y,c=TC,s=30,zorder=7,edgecolors="none")
            ax_hdiff.scatter(tx_,ty_,c=DT_BLUE,s=25,zorder=7,edgecolors="none",marker="o")
        plt.colorbar(cf2,ax=ax_hdiff,fraction=0.046,pad=0.04,
                     label="SINR diff (dB)").ax.tick_params(labelsize=4.5,colors=MT)
    ax_hdiff.set_xlim(0,W); ax_hdiff.set_ylim(0,H); ax_hdiff.set_aspect("equal","box")
    ax_hdiff.set_xticks([]); ax_hdiff.set_yticks([])
    ax_hdiff.text(0.5,-0.06,"● physical  ○ twin  — gap = position error",
                  transform=ax_hdiff.transAxes,ha="center",fontsize=5.5,color=MT)

    # HUD
    if not DT.paused:
        all_sv=[DT.phys_sinr[n[0]][-1] for n in nodes if DT.phys_sinr[n[0]]]
        tp_cur=DT.total_phys_tp[-1] if DT.total_phys_tp else 0
        avg_sv=np.mean(all_sv) if all_sv else 0
        txt_hud.set_text(f"t={DT.t:.1f}s ▶  {DT.ename}  ·  "
                         f"SINR {avg_sv:.1f}dB  ·  TP {tp_cur:.0f}Gbps  ·  "
                         f"Fidelity {all_fid:.1f}%")
        txt_hud.set_color(DT_RED if any_anom else DT_GREEN if all_fid>=90 else DT_AMBER)
    return []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KEYBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def on_key(ev):
    k=ev.key
    if k==" ":        on_pause(None)
    elif k=="r":      DT.reset()
    elif k=="h":      DT.show_hm=not DT.show_hm
    elif k=="a":      DT.inject()
    elif k=="q":      print("\n  Closing."); plt.close("all"); sys.exit(0)
    elif k in ("+","="): sl.set_val(min(5.0,DT.speed+0.5))
    elif k=="-":      sl.set_val(max(0.1,DT.speed-0.5))
    elif k in [str(i) for i in range(1,7)]:
        idx=int(k)-1
        if idx<len(EN): radio.set_active(idx); on_env(EN[idx])
fig.canvas.mpl_connect("key_press_event",on_key)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STARTUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n"+"="*68)
print("  6G THz Digital Twin Simulator")
print("  Physical World ↔ Virtual Replica Synchronisation")
print("="*68)
print("\n  Use cases:")
for i,(k,v) in enumerate(ENV.items(),1):
    fs=f"{v['freq_hz']/1e9:.0f}GHz" if v["freq_hz"]<1e12 else f"{v['freq_hz']/1e12:.1f}THz"
    print(f"    {i}. {k:22s}  {fs:7s}  {v['bw_ghz']}GHz BW  Peak {v['peak_gbps']:5d}Gbps  <{v['lat_ms']}ms")
print("\n  Controls: SPACE pause | R reset | A inject anomaly | H heatmap")
print("           1-6 switch env | +/- speed | Q quit")
print("\n  Digital Twin concepts:")
print("    Fidelity    = how closely twin mirrors physical (0-100%)")
print("    Sync latency = time for twin to update from reality (ms)")
print("    Anomaly     = deliberate desync (fault simulation)")
print("    Prediction  = twin position offset from real (metres)")
print("    Heatmap diff= SINR field divergence between worlds")
print("\n  Pre-computing heatmaps...")
for en in EN:
    if en not in DT.heatmaps:
        print(f"    {en}...",end="",flush=True)
        DT.heatmaps[en]=build_heatmap(ENV[en])
        print(" ✓")
print("\n  Launching Digital Twin window...\n")

anim=FuncAnimation(fig,draw_frame,interval=60,blit=False,cache_frame_data=False)
plt.show()
