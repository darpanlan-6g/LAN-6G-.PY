"""
================================================================================
  3GPP 6G THz Network — Real-Life Use Cases  (Python / Matplotlib Live Sim)
================================================================================
  C++ / NS3 alignment:
    • Friis + THz absorption path loss  ←→  ns3::ThzSpectrumPropagationLoss
    • Multi-beam MIMO beamforming       ←→  ns3::ThzDirectionalAntenna
    • Handover trigger (SINR threshold) ←→  ns3::LteHandoverAlgorithm
    • Flow monitor (TX/RX/latency)      ←→  ns3::FlowMonitor
    • UdpClient / UdpServer traffic     ←→  ns3::UdpClientHelper
    • Random walk mobility              ←→  ns3::RandomWalk2dMobilityModel
    • Constant velocity                 ←→  ns3::ConstantVelocityMobilityModel
    • WaypointMobility (figure-8)       ←→  ns3::WaypointMobilityModel

  Real-Life Use Cases (6 environments):
    1. Holographic XR Surgery      — Operating room, 300 GHz, URLLC
    2. Autonomous Factory          — Smart factory floor, 140 GHz, URLLC+mMTC
    3. Smart City Intersection     — V2X crossroad, 300 GHz, URLLC
    4. 6G Terabit Backhaul         — Roof-top P2P links, 1 THz, eMBB
    5. Underwater/Tunnel Rescue    — Confined space, 100 GHz, URLLC
    6. Holographic Classroom       — EDU room, 300 GHz, eMBB

  Controls  :  SPACE pause | R reset | E export CSV | 1–6 switch env
               + / -  speed | H heatmap | T trails | L links
================================================================================
  Run       : python 6g_thz_live_sim.py
  Deps      : pip install matplotlib numpy scipy
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")           # fallback: "Qt5Agg"  "MacOSX"  "Agg"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, RadioButtons, Slider, CheckButtons
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import warnings, time, csv
warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COLOUR PALETTE  (GitHub-dark inspired)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG        = "#0D1117"
PANEL_BG  = "#161B22"
GRID_COL  = "#21262D"
TEXT_COL  = "#E6EDF3"
MUTED     = "#7D8590"
BLUE      = "#388BFD"
GREEN     = "#3FB950"
RED       = "#F85149"
ORANGE    = "#D29922"
PURPLE    = "#BC8CFF"
TEAL      = "#39D353"
CYAN      = "#39C5CF"
PINK      = "#FF6EB4"

SVC_COLOR = {"URLLC": RED, "eMBB": BLUE, "mMTC": GREEN, "XR": PURPLE, "V2X": ORANGE}

NODE_STYLE = {
    "surgeon"   : {"color": RED,    "marker": "*",  "sz": 160, "label": "Surgeon"},
    "robot"     : {"color": PURPLE, "marker": "h",  "sz": 120, "label": "Robot Arm"},
    "sensor"    : {"color": TEAL,   "marker": "+",  "sz":  80, "label": "IoT Sensor"},
    "agv"       : {"color": ORANGE, "marker": "D",  "sz": 110, "label": "AGV"},
    "car"       : {"color": BLUE,   "marker": "^",  "sz": 110, "label": "Car"},
    "drone"     : {"color": CYAN,   "marker": "v",  "sz": 100, "label": "Drone"},
    "backhaul"  : {"color": GREEN,  "marker": "s",  "sz": 120, "label": "Backhaul Node"},
    "rescuer"   : {"color": RED,    "marker": "P",  "sz": 100, "label": "Rescuer"},
    "holo_disp" : {"color": PINK,   "marker": "8",  "sz": 110, "label": "Holo Display"},
    "student"   : {"color": PURPLE, "marker": "o",  "sz":  80, "label": "Student"},
    "camera"    : {"color": TEAL,   "marker": "x",  "sz":  80, "label": "Camera"},
    "rsu"       : {"color": ORANGE, "marker": "H",  "sz": 100, "label": "RSU"},
}

MAT_COLOR = {"wall": "#1C2227", "glass": "#0D2030", "metal": "#2D1F10",
             "concrete": "#1E2228", "free": "#0D1117"}
MAT_EDGE  = {"wall": "#484F58", "glass": "#388BFD",
             "metal": "#D29922", "concrete": "#555D64", "free": "#21262D"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  THz PHYSICS  (aligned with NS3 ThzSpectrumPropagationLoss)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Molecular absorption coefficients k(f) at key THz bands [1/m]
# Source: Jornet & Akyildiz, "Channel Modeling … THz band", 2011
_THZ_ABS = {
    0.10e12: 0.40,   # 100 GHz
    0.14e12: 0.50,   # 140 GHz
    0.30e12: 1.20,   # 300 GHz
    1.00e12: 8.00,   # 1 THz
}

def thz_absorption_db(d_m, freq_hz):
    """
    THz molecular absorption loss  [dB]
    L_abs = 10*log10(e^(k*d))  =  k*d * 10/ln(10)
    Equivalent C++: ThzSpectrumPropagationLossModel::DoCalcRxPower
    """
    k = _THZ_ABS.get(freq_hz, 1.0)
    return k * max(d_m, 0.01) * 10 / np.log(10)

def friis_db(d_m, freq_hz):
    """
    Free-space Friis path loss [dB]
    PL = 20*log10(4*pi*d*f/c)
    Equivalent C++: FriisPropagationLossModel::DoCalcRxPower
    """
    c = 3e8
    return 20*np.log10(max(4*np.pi*max(d_m,0.01)*freq_hz/c, 1e-30))

def compute_sinr_thz(nx, ny, gnbs_cfg, env_cfg, rng):
    """
    Full THz SINR:
      Rx_power = Tx_power - Friis_PL - THz_absorption - pen_loss + beamforming_gain
      SINR = signal / (interference + noise)
    Equivalent C++ block:
      nrHelper->SetPathlossAttribute("Frequency", DoubleValue(freq));
      channel->AddPropagationLossModel(thzLossModel);
    """
    freq     = env_cfg["freq_hz"]
    tx       = env_cfg["tx_power_dbm"]
    pen      = env_cfg["pen_loss_avg"]
    bf_gain  = env_cfg["beamforming_gain_db"]   # massive MIMO BF gain
    noise    = env_cfg["noise_floor_dbm"]

    rxs = []
    for gx, gy, *_ in gnbs_cfg:
        d   = np.hypot(nx - gx, ny - gy)
        pl  = friis_db(d, freq) + thz_absorption_db(d, freq) + pen * 0.20
        rx  = tx - pl + bf_gain + rng.normal(0, 1.8)
        rxs.append(rx)

    rxs.sort(reverse=True)
    sig  = 10 ** (rxs[0] / 10)
    intf = sum(10 ** (p / 10) for p in rxs[1:]) if len(rxs) > 1 else 0.0
    nois = 10 ** (noise / 10)
    return 10 * np.log10(max(sig / (intf + nois), 1e-12))

def shannon_tp(sinr_db, bw_ghz):
    """
    Shannon capacity in Tbps (BW in GHz)
    Practical efficiency factor 0.65
    """
    return bw_ghz * np.log2(1 + 10 ** (sinr_db / 10)) * 0.65

def sinr_qcolor(v):
    if v >= 18: return GREEN
    if v >= 10: return ORANGE
    if v >=  3: return "#E67E22"
    return RED

def build_heatmap_thz(env_cfg, res=38):
    W, H = env_cfg["area"]
    xs = np.linspace(0, W, res)
    ys = np.linspace(0, H, res)
    XX, YY = np.meshgrid(xs, ys)
    rng = np.random.default_rng(0)
    G = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            G[i, j] = compute_sinr_thz(XX[i,j], YY[i,j],
                                        env_cfg["gnbs"], env_cfg, rng)
    return XX, YY, gaussian_filter(G, sigma=1.5)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  USE-CASE ENVIRONMENT DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Each node tuple:
#  (id, type, x0, y0, vx, vy, service, bounce, ns3_model)
#  ns3_model → comment showing equivalent NS3 mobility model

ENVIRONMENTS = {

    # ── 1. HOLOGRAPHIC XR SURGERY ──────────────────────────────────────────
    "XR Surgery": {
        "desc": "Holographic remote surgery at 300 GHz\n"
                "Latency < 1 ms  |  BW 100 Gbps  |  URLLC",
        "area": (12, 10),           # 12×10 m operating room
        "freq_hz": 300e9,
        "bw_ghz": 100,              # 100 GHz bandwidth
        "tx_power_dbm": 20,
        "pen_loss_avg": 8,
        "beamforming_gain_db": 30,  # 1024-element beamformer
        "noise_floor_dbm": -80,
        "color": RED,
        "latency_target_ms": 1.0,
        "buildings": [
            (0, 0, 12, 10, "OR Chamber",     "wall",  28),
            (1, 1,  4,  2, "Instrument Bay", "metal", 35),
            (7, 1,  4,  2, "Imaging Suite",  "metal", 35),
        ],
        "gnbs": [
            (6.0, 9.5, "THz-AP", 2.5),
        ],
        # C++ equiv: nrHelper->SetGnbPhyAttribute("TxPower", DoubleValue(20));
        "nodes": [
            ("Surg1",  "surgeon",    3.0, 5.0,  0.00,  0.00, "URLLC", False,
             "ns3::ConstantPositionMobilityModel"),
            ("Surg2",  "surgeon",    9.0, 5.0,  0.00,  0.00, "URLLC", False,
             "ns3::ConstantPositionMobilityModel"),
            ("RobotL", "robot",      4.5, 4.5,  0.01,  0.00, "XR",    True,
             "ns3::ConstantVelocityMobilityModel"),
            ("RobotR", "robot",      7.5, 4.5, -0.01,  0.00, "XR",    True,
             "ns3::ConstantVelocityMobilityModel"),
            ("Cam1",   "camera",     2.0, 8.5,  0.00,  0.00, "URLLC", False,
             "ns3::ConstantPositionMobilityModel"),
            ("Cam2",   "camera",    10.0, 8.5,  0.00,  0.00, "URLLC", False,
             "ns3::ConstantPositionMobilityModel"),
            ("Holo",   "holo_disp",  6.0, 7.5,  0.00,  0.00, "XR",    False,
             "ns3::ConstantPositionMobilityModel"),
        ],
    },

    # ── 2. AUTONOMOUS FACTORY ──────────────────────────────────────────────
    "Auto Factory": {
        "desc": "Industry 6.0 smart factory at 140 GHz\n"
                "AGVs + robot arms + 10k sensors  |  URLLC + mMTC",
        "area": (80, 60),
        "freq_hz": 140e9,
        "bw_ghz": 50,
        "tx_power_dbm": 30,
        "pen_loss_avg": 25,
        "beamforming_gain_db": 25,
        "noise_floor_dbm": -85,
        "color": ORANGE,
        "latency_target_ms": 2.0,
        "buildings": [
            (0,  0, 80, 60, "Factory Shell",   "metal",    32),
            (5,  5, 20, 15, "Assembly Line A", "metal",    30),
            (30, 5, 20, 15, "Assembly Line B", "metal",    30),
            (55, 5, 20, 15, "Assembly Line C", "metal",    30),
            (5, 40, 30, 15, "Warehouse",       "concrete", 22),
            (45,40, 30, 15, "Control Room",    "wall",     20),
        ],
        "gnbs": [
            (20, 30, "gNB-F0", 8),
            (60, 30, "gNB-F1", 8),
            (40, 10, "gNB-F2", 8),
        ],
        "nodes": [
            ("AGV1",  "agv",     10, 30,  0.80,  0.00, "URLLC", False,
             "ns3::ConstantVelocityMobilityModel"),
            ("AGV2",  "agv",     40, 30, -0.70,  0.00, "URLLC", False,
             "ns3::ConstantVelocityMobilityModel"),
            ("AGV3",  "agv",     65, 20,  0.00,  0.60, "URLLC", False,
             "ns3::ConstantVelocityMobilityModel"),
            ("Arm1",  "robot",   15, 12,  0.00,  0.00, "URLLC", False,
             "ns3::ConstantPositionMobilityModel"),
            ("Arm2",  "robot",   40, 12,  0.00,  0.00, "URLLC", False,
             "ns3::ConstantPositionMobilityModel"),
            ("Arm3",  "robot",   65, 12,  0.00,  0.00, "URLLC", False,
             "ns3::ConstantPositionMobilityModel"),
            ("S1",    "sensor",  10, 45,  0.00,  0.00, "mMTC",  False,
             "ns3::ConstantPositionMobilityModel"),
            ("S2",    "sensor",  30, 45,  0.00,  0.00, "mMTC",  False,
             "ns3::ConstantPositionMobilityModel"),
            ("S3",    "sensor",  55, 45,  0.00,  0.00, "mMTC",  False,
             "ns3::ConstantPositionMobilityModel"),
            ("S4",    "sensor",  70, 45,  0.00,  0.00, "mMTC",  False,
             "ns3::ConstantPositionMobilityModel"),
        ],
    },

    # ── 3. SMART CITY INTERSECTION ──────────────────────────────────────────
    "Smart Intersection": {
        "desc": "6G V2X city crossroad at 300 GHz\n"
                "Zero-accident autonomous driving  |  URLLC <0.5 ms",
        "area": (100, 100),
        "freq_hz": 300e9,
        "bw_ghz": 80,
        "tx_power_dbm": 38,
        "pen_loss_avg": 15,
        "beamforming_gain_db": 28,
        "noise_floor_dbm": -82,
        "color": CYAN,
        "latency_target_ms": 0.5,
        "buildings": [
            (0,  0, 38, 38, "Block NW", "concrete", 22),
            (62, 0, 38, 38, "Block NE", "concrete", 22),
            (0, 62, 38, 38, "Block SW", "concrete", 22),
            (62,62, 38, 38, "Block SE", "concrete", 22),
        ],
        "gnbs": [
            (50, 50, "V2X-gNB", 12),
            (20, 50, "RSU-W",    6),
            (80, 50, "RSU-E",    6),
            (50, 20, "RSU-N",    6),
            (50, 80, "RSU-S",    6),
        ],
        "nodes": [
            ("C1",  "car",  5,  50,  2.5,  0.0,  "V2X",   False,
             "ns3::ConstantVelocityMobilityModel"),
            ("C2",  "car", 95,  50, -2.2,  0.0,  "V2X",   False,
             "ns3::ConstantVelocityMobilityModel"),
            ("C3",  "car", 50,  5,   0.0,  2.3,  "V2X",   False,
             "ns3::ConstantVelocityMobilityModel"),
            ("C4",  "car", 50, 95,   0.0, -2.0,  "V2X",   False,
             "ns3::ConstantVelocityMobilityModel"),
            ("Dr1", "drone",30, 30,  0.15, 0.10,  "URLLC", True,
             "ns3::RandomWalk2dMobilityModel"),
            ("Dr2", "drone",70, 70, -0.12, 0.08,  "URLLC", True,
             "ns3::RandomWalk2dMobilityModel"),
            ("Ped1","student",48,48,  0.08, 0.05, "URLLC", True,
             "ns3::RandomWalk2dMobilityModel"),
            ("Cam1","camera", 38,38,  0.0,  0.0,  "URLLC", False,
             "ns3::ConstantPositionMobilityModel"),
            ("Cam2","camera", 62,62,  0.0,  0.0,  "URLLC", False,
             "ns3::ConstantPositionMobilityModel"),
        ],
    },

    # ── 4. TERABIT BACKHAUL ─────────────────────────────────────────────────
    "THz Backhaul": {
        "desc": "1 THz point-to-point rooftop backhaul\n"
                "Target: 1 Tbps per link  |  eMBB / Network infra",
        "area": (500, 100),         # 500 m rooftop-to-rooftop distance
        "freq_hz": 1.00e12,
        "bw_ghz": 300,              # 300 GHz bandwidth
        "tx_power_dbm": 45,
        "pen_loss_avg": 5,
        "beamforming_gain_db": 40,  # highly directional 4096-element
        "noise_floor_dbm": -75,
        "color": GREEN,
        "latency_target_ms": 0.1,
        "buildings": [
            (0,   35, 30, 30, "Building A", "concrete", 22),
            (235, 35, 30, 30, "Relay Node", "concrete", 22),
            (470, 35, 30, 30, "Building B", "concrete", 22),
        ],
        "gnbs": [
            ( 15, 65, "THz-TX", 15),
            (250, 65, "THz-Relay", 15),
            (485, 65, "THz-RX", 15),
        ],
        "nodes": [
            ("BH0", "backhaul",  15,  65,  0.0, 0.0, "eMBB", False,
             "ns3::ConstantPositionMobilityModel"),
            ("BH1", "backhaul", 250,  65,  0.0, 0.0, "eMBB", False,
             "ns3::ConstantPositionMobilityModel"),
            ("BH2", "backhaul", 485,  65,  0.0, 0.0, "eMBB", False,
             "ns3::ConstantPositionMobilityModel"),
            ("Dr1", "drone",     80,  55,  1.5, 0.08,"eMBB", False,
             "ns3::ConstantVelocityMobilityModel"),
            ("Dr2", "drone",    300,  70, -1.2, 0.05,"eMBB", False,
             "ns3::ConstantVelocityMobilityModel"),
            ("Dr3", "drone",    420,  60,  1.0,-0.06,"eMBB", False,
             "ns3::ConstantVelocityMobilityModel"),
        ],
    },

    # ── 5. TUNNEL / CONFINED RESCUE ────────────────────────────────────────
    "Tunnel Rescue": {
        "desc": "6G search-and-rescue in confined tunnel at 100 GHz\n"
                "Body-worn radios + drones  |  URLLC critical comms",
        "area": (150, 15),          # 150 m × 15 m tunnel
        "freq_hz": 100e9,
        "bw_ghz": 30,
        "tx_power_dbm": 27,
        "pen_loss_avg": 30,
        "beamforming_gain_db": 18,
        "noise_floor_dbm": -88,
        "color": ORANGE,
        "latency_target_ms": 5.0,
        "buildings": [
            (0,   0, 150, 15, "Tunnel Walls",  "metal", 32),
            (0,   0, 150,  2, "Floor",         "concrete", 20),
            (0,  13, 150,  2, "Ceiling",       "concrete", 20),
            (60,  2,  10, 11, "Debris Block",  "concrete", 28),
        ],
        "gnbs": [
            (  5, 7, "THz-R0",  2),
            ( 75, 7, "THz-R1",  2),
            (145, 7, "THz-R2",  2),
        ],
        "nodes": [
            ("Rs1", "rescuer",   8,  7,  0.50, 0.00, "URLLC", False,
             "ns3::ConstantVelocityMobilityModel"),
            ("Rs2", "rescuer",  25,  7,  0.45, 0.00, "URLLC", False,
             "ns3::ConstantVelocityMobilityModel"),
            ("Rs3", "rescuer",  45,  7,  0.40, 0.00, "URLLC", False,
             "ns3::ConstantVelocityMobilityModel"),
            ("Dr1", "drone",    15,  10, 0.60, 0.00, "URLLC", False,
             "ns3::ConstantVelocityMobilityModel"),
            ("Dr2", "drone",    35,  10, 0.55, 0.00, "URLLC", False,
             "ns3::ConstantVelocityMobilityModel"),
            ("S1",  "sensor",   70,  7,  0.00, 0.00, "mMTC",  False,
             "ns3::ConstantPositionMobilityModel"),
            ("S2",  "sensor",  110,  7,  0.00, 0.00, "mMTC",  False,
             "ns3::ConstantPositionMobilityModel"),
        ],
    },

    # ── 6. HOLOGRAPHIC CLASSROOM ────────────────────────────────────────────
    "Holo Classroom": {
        "desc": "Holographic tele-education at 300 GHz\n"
                "4K holo-displays per student  |  eMBB 10 Gbps/user",
        "area": (20, 15),
        "freq_hz": 300e9,
        "bw_ghz": 60,
        "tx_power_dbm": 22,
        "pen_loss_avg": 12,
        "beamforming_gain_db": 26,
        "noise_floor_dbm": -82,
        "color": PURPLE,
        "latency_target_ms": 3.0,
        "buildings": [
            (0,  0, 20, 15, "Classroom",         "wall",     20),
            (0,  0,  2, 15, "West Wall",         "concrete", 22),
            (18, 0,  2, 15, "East Wall",         "concrete", 22),
            (1, 11, 18,  3, "Presentation Wall", "wall",     20),
        ],
        "gnbs": [
            (10, 13.5, "THz-AP0", 2),
            ( 3,  5.5, "THz-AP1", 2),
            (17,  5.5, "THz-AP2", 2),
        ],
        "nodes": [
            ("St1", "student",  4,  3,  0.04,  0.03, "eMBB", True,
             "ns3::RandomWalk2dMobilityModel"),
            ("St2", "student",  8,  3, -0.03,  0.04, "eMBB", True,
             "ns3::RandomWalk2dMobilityModel"),
            ("St3", "student", 12,  3,  0.05, -0.02, "eMBB", True,
             "ns3::RandomWalk2dMobilityModel"),
            ("St4", "student", 16,  3, -0.04,  0.03, "eMBB", True,
             "ns3::RandomWalk2dMobilityModel"),
            ("St5", "student",  4,  7,  0.03,  0.04, "eMBB", True,
             "ns3::RandomWalk2dMobilityModel"),
            ("St6", "student",  8,  7, -0.04, -0.03, "eMBB", True,
             "ns3::RandomWalk2dMobilityModel"),
            ("St7", "student", 12,  7,  0.04,  0.02, "eMBB", True,
             "ns3::RandomWalk2dMobilityModel"),
            ("St8", "student", 16,  7, -0.03,  0.04, "eMBB", True,
             "ns3::RandomWalk2dMobilityModel"),
            ("Holo1","holo_disp", 5, 12, 0.0,  0.0,  "XR",   False,
             "ns3::ConstantPositionMobilityModel"),
            ("Holo2","holo_disp",10, 12, 0.0,  0.0,  "XR",   False,
             "ns3::ConstantPositionMobilityModel"),
            ("Holo3","holo_disp",15, 12, 0.0,  0.0,  "XR",   False,
             "ns3::ConstantPositionMobilityModel"),
        ],
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIMULATION STATE CLASS
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
        self.svc_filter  = {s: True for s in SVC_COLOR}
        self.heatmaps    = {}
        self.reset()

    @property
    def cfg(self):
        return ENVIRONMENTS[self.env_name]

    def reset(self):
        cfg = self.cfg
        self.rng       = np.random.default_rng(42)
        self.pos       = {n[0]: [float(n[2]), float(n[3])] for n in cfg["nodes"]}
        self.vel       = {n[0]: [n[4] * self.speed, n[5] * self.speed] for n in cfg["nodes"]}
        self.bounce    = {n[0]: n[7] for n in cfg["nodes"]}
        self.sinr_hist = {n[0]: [] for n in cfg["nodes"]}
        self.tp_hist   = {n[0]: [] for n in cfg["nodes"]}
        self.lat_hist  = {n[0]: [] for n in cfg["nodes"]}
        self.trail_x   = {n[0]: [] for n in cfg["nodes"]}
        self.trail_y   = {n[0]: [] for n in cfg["nodes"]}
        self.prev_cell = {n[0]: None for n in cfg["nodes"]}
        self.handovers = 0
        self.total_tp  = []
        self.t         = 0.0
        self.frame     = 0
        if self.env_name not in self.heatmaps:
            print(f"  [HM] {self.env_name}…", end="", flush=True)
            self.heatmaps[self.env_name] = build_heatmap_thz(cfg)
            print(" done")

    def step(self, dt=0.04):
        if self.paused:
            return
        cfg = self.cfg
        W, H = cfg["area"]
        self.t     += dt * self.speed
        self.frame += 1
        total_tp = 0.0

        for node in cfg["nodes"]:
            nid = node[0]
            x, y   = self.pos[nid]
            vx, vy = self.vel[nid]

            # ── Mobility ─────────────────────────────────────────────────────
            x += vx * dt
            y += vy * dt
            if self.bounce[nid]:
                if x <= 0 or x >= W: vx *= -1; x = np.clip(x, 0.1, W - 0.1)
                if y <= 0 or y >= H: vy *= -1; y = np.clip(y, 0.1, H - 0.1)
            else:
                if vx > 0 and x > W + 2: x = -2
                if vx < 0 and x < -2:    x = W + 2
                if vy > 0 and y > H + 2: y = -2
                if vy < 0 and y < -2:    y = H + 2
                x = np.clip(x, 0.1, W - 0.1)
                y = np.clip(y, 0.1, H - 0.1)
            self.pos[nid] = [x, y]
            self.vel[nid] = [vx, vy]

            # ── SINR / Throughput / Latency ──────────────────────────────────
            sv = compute_sinr_thz(x, y, cfg["gnbs"], cfg, self.rng)
            tp = shannon_tp(sv, cfg["bw_ghz"])  # Tbps
            # Latency model: inversely proportional to SINR quality
            latency_ms = max(0.05, cfg["latency_target_ms"] * 2 * np.exp(-sv / 15))

            for hist, val, maxlen in [
                (self.sinr_hist[nid], sv,         400),
                (self.tp_hist[nid],   tp * 1000,  400),   # store as Gbps
                (self.lat_hist[nid],  latency_ms, 400),
            ]:
                hist.append(val)
                if len(hist) > maxlen:
                    hist.pop(0)

            total_tp += tp

            # Trail
            self.trail_x[nid].append(x)
            self.trail_y[nid].append(y)
            if len(self.trail_x[nid]) > 60:
                self.trail_x[nid].pop(0)
                self.trail_y[nid].pop(0)

            # Handover
            gnbs   = cfg["gnbs"]
            dists  = [np.hypot(x - g[0], y - g[1]) for g in gnbs]
            cell   = int(np.argmin(dists))
            if self.prev_cell[nid] is not None and self.prev_cell[nid] != cell:
                self.handovers += 1
            self.prev_cell[nid] = cell

        self.total_tp.append(total_tp * 1000)  # Gbps
        if len(self.total_tp) > 400:
            self.total_tp.pop(0)

    def export_csv(self):
        fname = f"thz_export_{self.env_name.replace(' ','_')}_{int(time.time())}.csv"
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Node","Type","Service","X_m","Y_m",
                        "SINR_dB","Throughput_Gbps","Latency_ms","NS3_Mobility"])
            cfg = self.cfg
            for node in cfg["nodes"]:
                nid  = node[0]; svc = node[6]; ns3 = node[8]
                x, y = self.pos[nid]
                sv   = self.sinr_hist[nid][-1] if self.sinr_hist[nid] else 0
                tp   = self.tp_hist[nid][-1]   if self.tp_hist[nid]   else 0
                lat  = self.lat_hist[nid][-1]  if self.lat_hist[nid]  else 0
                w.writerow([nid, node[1], svc,
                            f"{x:.2f}", f"{y:.2f}",
                            f"{sv:.2f}", f"{tp:.1f}", f"{lat:.3f}", ns3])
        print(f"\n✅  Exported → {fname}")
        return fname

SIM = SimState()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE & AXES LAYOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.facecolor":   PANEL_BG, "figure.facecolor": BG,
    "axes.edgecolor":   GRID_COL, "axes.labelcolor":  TEXT_COL,
    "xtick.color":      MUTED,    "ytick.color":       MUTED,
    "text.color":       TEXT_COL, "axes.titlecolor":   TEXT_COL,
    "axes.grid":        True,     "grid.color":        GRID_COL,
    "grid.linewidth":   0.35,     "axes.spines.top":   False,
    "axes.spines.right":False,
})

fig = plt.figure(figsize=(24, 14), facecolor=BG)
try:
    fig.canvas.manager.set_window_title("6G THz Live Sim — Real-Life Use Cases")
except Exception:
    pass

# ── Layout ───────────────────────────────────────────────────────────────────
gs = gridspec.GridSpec(3, 5,
    figure=fig,
    left=0.01, right=0.99, top=0.91, bottom=0.27,
    hspace=0.58, wspace=0.42,
)

ax_topo  = fig.add_subplot(gs[0:2, 0:2])  # large topology
ax_sinr  = fig.add_subplot(gs[0,   2])    # SINR bars
ax_tp    = fig.add_subplot(gs[0,   3])    # TP timeline
ax_lat   = fig.add_subplot(gs[0,   4])    # latency timeline
ax_ts    = fig.add_subplot(gs[1,   2])    # SINR time series
ax_dist  = fig.add_subplot(gs[1,   3])    # SINR vs distance
ax_cdf   = fig.add_subplot(gs[1,   4])    # CDF
ax_agg   = fig.add_subplot(gs[2,   :])    # aggregate TP full width

# ── Control panel ────────────────────────────────────────────────────────────
gs_c = gridspec.GridSpec(1, 7,
    figure=fig,
    left=0.01, right=0.99, bottom=0.01, top=0.25,
    wspace=0.32,
)
ax_radio = fig.add_subplot(gs_c[0, 0])
ax_svc   = fig.add_subplot(gs_c[0, 1])
ax_disp  = fig.add_subplot(gs_c[0, 2])
ax_spd   = fig.add_subplot(gs_c[0, 3])
ax_btn   = fig.add_subplot(gs_c[0, 4])
ax_leg   = fig.add_subplot(gs_c[0, 5])
ax_inf   = fig.add_subplot(gs_c[0, 6])

for ax in [ax_btn, ax_leg, ax_inf]:
    ax.axis("off")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.967,
    "3GPP 6G THz Network  ·  Real-Life Use Cases  ·  Interactive Live Simulation",
    ha="center", fontsize=13, fontweight="bold", color=TEXT_COL)
fig.text(0.5, 0.951,
    "100 GHz – 1 THz  |  Up to 1 Tbps  |  Sub-ms URLLC  |  "
    "NS3-aligned physics: THz absorption + Friis PL + Massive MIMO BF",
    ha="center", fontsize=8, color=MUTED)

txt_status = fig.text(0.99, 0.967, "t=0.00s  ▶",
    ha="right", fontsize=9, fontweight="bold", color=GREEN)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WIDGETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENV_NAMES = list(ENVIRONMENTS.keys())

ax_radio.set_title("Use Case", fontsize=8, color=TEXT_COL, pad=2)
radio_env = RadioButtons(ax_radio, ENV_NAMES, activecolor=BLUE)
for lbl in radio_env.labels:
    lbl.set_color(TEXT_COL); lbl.set_fontsize(7.5)

def on_env(label):
    SIM.env_name = label
    SIM.reset()
radio_env.on_clicked(on_env)

ax_svc.set_title("Service", fontsize=8, color=TEXT_COL, pad=2)
chk_svc = CheckButtons(ax_svc, list(SVC_COLOR.keys()),
                        [True]*len(SVC_COLOR))
for lbl in chk_svc.labels:
    lbl.set_color(TEXT_COL); lbl.set_fontsize(7.5)

def on_svc(label):
    SIM.svc_filter[label] = not SIM.svc_filter[label]
chk_svc.on_clicked(on_svc)

ax_disp.set_title("Display", fontsize=8, color=TEXT_COL, pad=2)
chk_disp = CheckButtons(ax_disp, ["Heatmap","Trails","Links"],
                         [True, True, True])
for lbl in chk_disp.labels:
    lbl.set_color(TEXT_COL); lbl.set_fontsize(7.5)

def on_disp(label):
    if label == "Heatmap": SIM.show_hm     = not SIM.show_hm
    if label == "Trails":  SIM.show_trails = not SIM.show_trails
    if label == "Links":   SIM.show_links  = not SIM.show_links
chk_disp.on_clicked(on_disp)

ax_spd.set_title("Speed", fontsize=8, color=TEXT_COL, pad=2)
sl_spd = Slider(ax_spd, "", 0.1, 6.0, valinit=1.0,
                color=BLUE, track_color=GRID_COL)
sl_spd.label.set_color(TEXT_COL)
sl_spd.valtext.set_color(BLUE)
sl_spd.on_changed(lambda v: setattr(SIM, "speed", v))

# Buttons
for pos, label, cb_name in [
    ([0.625, 0.18, 0.065, 0.032], "⏸  Pause",  "pause"),
    ([0.625, 0.14, 0.065, 0.032], "↺  Reset",  "reset"),
    ([0.625, 0.10, 0.065, 0.032], "⬇  CSV",    "export"),
]:
    ax_b = fig.add_axes(pos)
    btn  = Button(ax_b, label, color=PANEL_BG, hovercolor=GRID_COL)
    btn.label.set_color(TEXT_COL); btn.label.set_fontsize(8)
    if cb_name == "pause":
        btn_pause = btn
    elif cb_name == "reset":
        btn_reset = btn
    else:
        btn_export = btn

def on_pause(ev):
    SIM.paused = not SIM.paused
    btn_pause.label.set_text("▶  Resume" if SIM.paused else "⏸  Pause")
    txt_status.set_color(ORANGE if SIM.paused else GREEN)
btn_pause.on_clicked(on_pause)
btn_reset.on_clicked(lambda ev: SIM.reset())
btn_export.on_clicked(lambda ev: SIM.export_csv())

# Legend
ax_leg.set_title("Node Types", fontsize=8, color=TEXT_COL, pad=2)
yl = 0.97
for nt, st in NODE_STYLE.items():
    ax_leg.plot(0.07, yl, marker=st["marker"], color=st["color"],
                markersize=7, transform=ax_leg.transAxes, clip_on=False)
    ax_leg.text(0.16, yl, st["label"], fontsize=7, color=TEXT_COL,
                va="center", transform=ax_leg.transAxes)
    yl -= 0.088

# RF Info panel
ax_inf.set_title("RF / NS3 Config", fontsize=8, color=TEXT_COL, pad=2)
rf_lines = [
    ("Freq range",  "100G – 1 THz"),
    ("Bandwidth",   "30 – 300 GHz"),
    ("BF gain",     "18 – 40 dBi"),
    ("PL model",    "Friis + THz abs"),
    ("NS3 module",  "ThzSpectrumLoss"),
    ("Mobility",    "Const/Walk/WP"),
    ("Flow mon.",   "FlowMonitorHelper"),
    ("App layer",   "UdpClientHelper"),
    ("SINR trig.",  "LteHandoverAlg"),
    ("Numerology",  "μ=4  240kHz SCS"),
]
for i, (k, v) in enumerate(rf_lines):
    yp = 0.96 - i * 0.096
    ax_inf.text(0.01, yp, k + ":", fontsize=7, color=MUTED,
                transform=ax_inf.transAxes, va="center")
    ax_inf.text(0.50, yp, v, fontsize=7, color=TEXT_COL, fontweight="bold",
                transform=ax_inf.transAxes, va="center")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DRAW FRAME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cmap_tab  = plt.get_cmap("tab20")
norm_sinr = Normalize(vmin=-5, vmax=25)

def draw_frame(_fn):
    SIM.step(dt=0.04)
    cfg       = SIM.cfg
    W, H      = cfg["area"]
    nodes_def = cfg["nodes"]
    active    = [n for n in nodes_def if SIM.svc_filter.get(n[6], True)]
    ec        = cfg["color"]
    freq_str  = f"{cfg['freq_hz']/1e9:.0f} GHz" if cfg["freq_hz"] < 1e12 else \
                f"{cfg['freq_hz']/1e12:.1f} THz"
    n_active  = max(len(active) - 1, 1)

    # ── TOPOLOGY ─────────────────────────────────────────────────────────────
    ax_topo.cla()
    ax_topo.set_facecolor(PANEL_BG)
    ax_topo.set_xlim(0, W); ax_topo.set_ylim(0, H)
    ax_topo.set_aspect("equal", adjustable="box")
    ax_topo.set_title(
        f"{SIM.env_name}   [{freq_str}  ·  {cfg['bw_ghz']} GHz BW]",
        fontsize=9, color=ec, fontweight="bold", pad=3)
    ax_topo.set_xlabel("x (m)", color=MUTED)
    ax_topo.set_ylabel("y (m)", color=MUTED)

    # Use-case description box
    ax_topo.text(0.01, 0.99, cfg["desc"], transform=ax_topo.transAxes,
                 fontsize=6, color=MUTED, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=GRID_COL, alpha=0.8))

    # Heatmap
    if SIM.show_hm and SIM.env_name in SIM.heatmaps:
        XX, YY, hm = SIM.heatmaps[SIM.env_name]
        ax_topo.pcolormesh(XX, YY, hm, cmap="plasma",
                           vmin=-10, vmax=25, shading="gouraud", alpha=0.35)

    # Buildings
    for bld in cfg["buildings"]:
        bx, by, bw, bh, blbl, mat, _ = bld
        rect = mpatches.FancyBboxPatch(
            (bx, by), bw, bh, boxstyle="round,pad=0.3",
            lw=0.7, edgecolor=MAT_EDGE.get(mat, "#484F58"),
            facecolor=MAT_COLOR.get(mat, "#1E2228"), alpha=0.80)
        ax_topo.add_patch(rect)
        ax_topo.text(bx + bw / 2, by + bh / 2, blbl,
                     ha="center", va="center", fontsize=5, color=MUTED)

    # gNB towers + pulse rings
    gnbs_xy = [(g[0], g[1]) for g in cfg["gnbs"]]
    for gx, gy, glbl, *_ in cfg["gnbs"]:
        rmax = min(W, H) * 0.30
        for r, a in [(rmax, 0.04), (rmax*0.6, 0.07), (rmax*0.3, 0.11)]:
            ax_topo.add_patch(plt.Circle((gx,gy), r, color=ec, alpha=a, lw=0))
        pr = (SIM.frame * 0.55) % (rmax * 1.2) + 2
        ax_topo.add_patch(plt.Circle((gx,gy), pr, color=ec,
                                     alpha=max(0, 0.35 - pr/(rmax*1.5)),
                                     fill=False, lw=0.9))
        ax_topo.plot(gx, gy, "^", color=ec, ms=10, zorder=8,
                     markeredgecolor="white", markeredgewidth=0.8)
        ax_topo.text(gx, gy - H * 0.04, glbl, ha="center",
                     fontsize=6, color=ec, fontweight="bold")

    # Nodes
    for ni, node in enumerate(active):
        nid, ntype = node[0], node[1]
        svc = node[6]
        x, y = SIM.pos[nid]
        hist = SIM.sinr_hist[nid]
        sv   = hist[-1] if hist else 12.0
        st   = NODE_STYLE.get(ntype, NODE_STYLE["sensor"])

        # Trail
        if SIM.show_trails and len(SIM.trail_x[nid]) > 2:
            pts = np.array([SIM.trail_x[nid], SIM.trail_y[nid]]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            n_seg = len(segs)
            lc = LineCollection(segs, cmap="plasma", norm=norm_sinr,
                                lw=1.1, alpha=0.55)
            lc.set_array(np.array(hist[-n_seg:]) if len(hist) >= n_seg
                         else np.full(n_seg, sv))
            ax_topo.add_collection(lc)

        # Link to nearest gNB
        if SIM.show_links:
            d_  = [np.hypot(x-gx, y-gy) for gx,gy in gnbs_xy]
            bst = gnbs_xy[int(np.argmin(d_))]
            ax_topo.plot([x, bst[0]], [y, bst[1]],
                         color=sinr_qcolor(sv), lw=0.55, alpha=0.50, zorder=2)

        # Marker
        ax_topo.scatter(x, y, c=st["color"], marker=st["marker"],
                        s=st["sz"], zorder=7,
                        edgecolors="white", linewidths=0.6)
        # SINR quality dot
        ax_topo.scatter(x + W*0.012, y + H*0.018, c=sinr_qcolor(sv),
                        s=16, zorder=9, edgecolors="none")
        ax_topo.text(x, y + H*0.04, nid, fontsize=5,
                     ha="center", color=TEXT_COL, zorder=10)

    # ── LIVE SINR BARS ────────────────────────────────────────────────────────
    ax_sinr.cla(); ax_sinr.set_facecolor(PANEL_BG)
    ax_sinr.set_title("Live SINR (dB)", fontsize=8, pad=3)
    ax_sinr.set_xlim(-5, 28)
    for i, node in enumerate(active):
        nid = node[0]; svc = node[6]
        sv  = SIM.sinr_hist[nid][-1] if SIM.sinr_hist[nid] else 0
        ax_sinr.barh(i, sv, color=sinr_qcolor(sv), height=0.65,
                     edgecolor="none", alpha=0.88)
        ax_sinr.text(max(sv+0.2, 0.3), i, f"{sv:.1f}",
                     va="center", fontsize=6, color=TEXT_COL)
        ax_sinr.text(-4.8, i, nid, va="center", fontsize=6,
                     color=SVC_COLOR.get(svc, MUTED), fontweight="bold")
    ax_sinr.set_yticks([])
    ax_sinr.axvline(18, color=GREEN, lw=0.7, ls="--", alpha=0.5)
    ax_sinr.axvline(10, color=RED,   lw=0.7, ls="--", alpha=0.5)
    ax_sinr.set_xlabel("SINR (dB)", color=MUTED)
    ax_sinr.grid(axis="x", color=GRID_COL, lw=0.3)

    # ── THROUGHPUT TIMELINE (Gbps) ───────────────────────────────────────────
    ax_tp.cla(); ax_tp.set_facecolor(PANEL_BG)
    ax_tp.set_title("Throughput (Gbps)", fontsize=8, pad=3)
    ax_tp.grid(color=GRID_COL, lw=0.3)
    for ni, node in enumerate(active):
        nid = node[0]
        h   = SIM.tp_hist[nid]
        if len(h) < 2: continue
        t_  = np.arange(len(h)) * 0.04
        ax_tp.plot(t_, h, color=cmap_tab(ni / n_active), lw=0.9,
                   alpha=0.85, label=nid)
    ax_tp.set_xlabel("t (s)", color=MUTED)
    ax_tp.set_ylabel("Gbps",  color=MUTED)
    ax_tp.legend(fontsize=5, ncol=2, loc="upper left",
                 framealpha=0.4, facecolor=PANEL_BG, edgecolor=GRID_COL,
                 labelcolor=TEXT_COL)

    # ── LATENCY TIMELINE (ms) ────────────────────────────────────────────────
    ax_lat.cla(); ax_lat.set_facecolor(PANEL_BG)
    ax_lat.set_title("Latency (ms)", fontsize=8, pad=3)
    ax_lat.grid(color=GRID_COL, lw=0.3)
    tgt = cfg["latency_target_ms"]
    ax_lat.axhline(tgt, color=RED, lw=0.8, ls="--", alpha=0.6)
    ax_lat.text(0.02, tgt * 1.05 if tgt > 0.5 else tgt + 0.05,
                f"Target {tgt} ms", fontsize=6, color=RED,
                transform=ax_lat.get_yaxis_transform())
    for ni, node in enumerate(active):
        nid = node[0]
        h   = SIM.lat_hist[nid]
        if len(h) < 2: continue
        t_ = np.arange(len(h)) * 0.04
        ax_lat.plot(t_, h, color=cmap_tab(ni / n_active), lw=0.9,
                    alpha=0.85, label=nid)
    ax_lat.set_xlabel("t (s)", color=MUTED)
    ax_lat.set_ylabel("ms",    color=MUTED)
    ax_lat.legend(fontsize=5, ncol=2, loc="upper right",
                  framealpha=0.4, facecolor=PANEL_BG, edgecolor=GRID_COL,
                  labelcolor=TEXT_COL)

    # ── SINR TIME SERIES ─────────────────────────────────────────────────────
    ax_ts.cla(); ax_ts.set_facecolor(PANEL_BG)
    ax_ts.set_title("SINR History", fontsize=8, pad=3)
    ax_ts.grid(color=GRID_COL, lw=0.3)
    ax_ts.axhspan(18, 30, alpha=0.06, color=GREEN)
    ax_ts.axhspan(10, 18, alpha=0.06, color=ORANGE)
    ax_ts.axhspan(-5, 10, alpha=0.06, color=RED)
    for ni, node in enumerate(active):
        nid = node[0]
        h   = SIM.sinr_hist[nid]
        if len(h) < 2: continue
        t_ = np.arange(len(h)) * 0.04
        ax_ts.plot(t_, h, color=cmap_tab(ni / n_active), lw=0.85, alpha=0.85)
    ax_ts.axhline(18, color=GREEN, lw=0.6, ls="--", alpha=0.4)
    ax_ts.axhline(10, color=RED,   lw=0.6, ls="--", alpha=0.4)
    ax_ts.set_ylim(-5, 30)
    ax_ts.set_xlabel("t (s)", color=MUTED)
    ax_ts.set_ylabel("SINR (dB)", color=MUTED)

    # ── SINR vs DISTANCE ─────────────────────────────────────────────────────
    ax_dist.cla(); ax_dist.set_facecolor(PANEL_BG)
    ax_dist.set_title("SINR vs Distance", fontsize=8, pad=3)
    ax_dist.grid(color=GRID_COL, lw=0.3)
    for ni, node in enumerate(active):
        nid, ntype = node[0], node[1]
        h = SIM.sinr_hist[nid]
        if not h: continue
        x, y = SIM.pos[nid]
        d_   = min(np.hypot(x-g[0], y-g[1]) for g in cfg["gnbs"])
        sv   = h[-1]
        st   = NODE_STYLE.get(ntype, NODE_STYLE["sensor"])
        ax_dist.scatter(d_, sv, c=st["color"], marker=st["marker"],
                        s=60, zorder=5, edgecolors="none", alpha=0.9)
        ax_dist.annotate(nid, (d_, sv), fontsize=5, color=MUTED,
                         textcoords="offset points", xytext=(3, 2))
    ax_dist.set_xlabel("Distance (m)", color=MUTED)
    ax_dist.set_ylabel("SINR (dB)",    color=MUTED)
    ax_dist.set_ylim(-5, 28)

    # ── CDF ──────────────────────────────────────────────────────────────────
    ax_cdf.cla(); ax_cdf.set_facecolor(PANEL_BG)
    ax_cdf.set_title("SINR CDF", fontsize=8, pad=3)
    ax_cdf.grid(color=GRID_COL, lw=0.3)
    for ni, node in enumerate(active):
        nid = node[0]
        h   = SIM.sinr_hist[nid]
        if len(h) < 5: continue
        s  = np.sort(h)
        cdf = np.arange(1, len(s)+1) / len(s)
        ax_cdf.plot(s, cdf, color=cmap_tab(ni / n_active), lw=0.9, alpha=0.85)
    ax_cdf.axvline(18, color=GREEN, lw=0.7, ls="--", alpha=0.5)
    ax_cdf.axvline(10, color=RED,   lw=0.7, ls="--", alpha=0.5)
    ax_cdf.set_xlim(-5, 28); ax_cdf.set_ylim(0, 1)
    ax_cdf.set_xlabel("SINR (dB)", color=MUTED)
    ax_cdf.set_ylabel("CDF",       color=MUTED)

    # ── AGGREGATE THROUGHPUT ─────────────────────────────────────────────────
    ax_agg.cla(); ax_agg.set_facecolor(PANEL_BG)
    ax_agg.set_title(
        f"System Aggregate Throughput (Gbps)  ·  {SIM.env_name}  "
        f"·  Handovers: {SIM.handovers}",
        fontsize=8, pad=3)
    ax_agg.grid(color=GRID_COL, lw=0.3)
    if len(SIM.total_tp) > 2:
        t_a = np.arange(len(SIM.total_tp)) * 0.04
        tp_a = np.array(SIM.total_tp)
        ax_agg.plot(t_a, tp_a, color=ec, lw=1.3, alpha=0.9, label="Total")
        ax_agg.fill_between(t_a, tp_a, alpha=0.10, color=ec)
        # Per-service lines
        for svc, sc in SVC_COLOR.items():
            svc_nodes = [n for n in nodes_def
                         if n[6]==svc and SIM.svc_filter.get(svc, True)]
            if not svc_nodes: continue
            svc_tp = []
            for i in range(len(SIM.total_tp)):
                row = sum(SIM.tp_hist[nd[0]][i]
                          for nd in svc_nodes
                          if i < len(SIM.tp_hist[nd[0]]))
                svc_tp.append(row)
            ax_agg.plot(t_a[:len(svc_tp)], svc_tp,
                        color=sc, lw=0.8, ls="--", alpha=0.7, label=svc)

    ax_agg.set_xlabel("Time (s)", color=MUTED)
    ax_agg.set_ylabel("Gbps",     color=MUTED)
    ax_agg.legend(fontsize=7, loc="upper left", ncol=4,
                  framealpha=0.4, facecolor=PANEL_BG,
                  edgecolor=GRID_COL, labelcolor=TEXT_COL)

    # ── Status ───────────────────────────────────────────────────────────────
    if not SIM.paused:
        txt_status.set_text(f"t={SIM.t:.1f}s  ▶  {SIM.env_name}")
        txt_status.set_color(GREEN)

    return []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KEYBOARD SHORTCUTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def on_key(ev):
    if ev.key == " ":          on_pause(None)
    elif ev.key == "r":        SIM.reset()
    elif ev.key == "e":        SIM.export_csv()
    elif ev.key in [str(i) for i in range(1, 7)]:
        idx = int(ev.key) - 1
        if idx < len(ENV_NAMES):
            radio_env.set_active(idx)
            on_env(ENV_NAMES[idx])
    elif ev.key in ("+","="):  sl_spd.set_val(min(6.0, SIM.speed + 0.5))
    elif ev.key == "-":        sl_spd.set_val(max(0.1, SIM.speed - 0.5))
    elif ev.key == "h":        on_disp("Heatmap")
    elif ev.key == "t":        on_disp("Trails")
    elif ev.key == "l":        on_disp("Links")

fig.canvas.mpl_connect("key_press_event", on_key)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STARTUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*65)
print("  3GPP 6G THz  ·  Real-Life Use Cases  ·  Live Simulation")
print("="*65)
print("\n  Use cases:")
for i, (k, v) in enumerate(ENVIRONMENTS.items(), 1):
    print(f"    {i}. {k:22s}  {v['freq_hz']/1e9:.0f} GHz  "
          f"{v['bw_ghz']} GHz BW  tgt<{v['latency_target_ms']} ms")
print("\n  Pre-computing THz heatmaps…")
for ename in ENV_NAMES:
    if ename not in SIM.heatmaps:
        print(f"    [{ename}]…", end="", flush=True)
        SIM.heatmaps[ename] = build_heatmap_thz(ENVIRONMENTS[ename])
        print(" done")
print("\n  Keyboard shortcuts:")
print("    SPACE / R / E     pause | reset | export CSV")
print("    1–6               switch use case")
print("    + / -             speed up / down")
print("    H / T / L         toggle heatmap / trails / links")
print("\n  Launching …\n")

anim = FuncAnimation(fig, draw_frame, interval=55,
                     blit=False, cache_frame_data=False)
plt.show()