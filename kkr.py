"""
=============================================================================
  3GPP 6G NR DIGITAL TWIN - FR3 V2X @ 24 GHz (PYTHON PORT)
  Translated from ns-3 C++
  Features: 5 gNBs, Figure-8 Mobility, FR3 Building Penetration, Handover
=============================================================================
"""

import matplotlib
import os

_DISPLAY = os.environ.get("DISPLAY", "")
if _DISPLAY:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import math
import collections
import random

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — SYSTEM CONFIGURATION (Ported from C++)
# ─────────────────────────────────────────────────────────────────────────────

class FR3_24GHz_Config:
    CARRIER_FREQ_HZ  = 24e9        # 24 GHz (FR3 Upper Limit)
    BANDWIDTH_HZ     = 200e6       # 200 MHz Bandwidth
    GNB_TX_POWER_DBM = 40.0        # 40 dBm
    GNB_ANT_GAIN_DBI = 22.0        # 22 dBi
    UE_ANT_GAIN_DBI  = 5.0         
    NOISE_FIGURE_DB  = 9.0           
    IMPLEMENTATION_LOSS_DB = 2.0     
    UDP_PACKET_BYTES = 1400        

    @staticmethod
    def thermal_noise_dbm():
        return -174 + 10*math.log10(FR3_24GHz_Config.BANDWIDTH_HZ) + FR3_24GHz_Config.NOISE_FIGURE_DB

    @staticmethod
    def shannon_capacity_mbps(snr_db):
        """Returns capacity in Mbps rather than Gbps due to lower FR3 bandwidth"""
        snr_lin = 10**(snr_db/10)
        return (FR3_24GHz_Config.BANDWIDTH_HZ * math.log2(1 + snr_lin)) / 1e6

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — ENVIRONMENT & CHANNEL MODEL (Ported from InitializeBuildings)
# ─────────────────────────────────────────────────────────────────────────────

BUILDINGS = [
    {"x_min": 10, "x_max": 20, "y_min": 10, "y_max": 25, "mat": "concrete", "loss": 22.0},
    {"x_min": 25, "x_max": 35, "y_min": 15, "y_max": 30, "mat": "glass",    "loss": 8.0},
    {"x_min": 40, "x_max": 50, "y_min": 20, "y_max": 35, "mat": "metal",    "loss": 32.0},
    {"x_min": 15, "x_max": 25, "y_min": 40, "y_max": 55, "mat": "concrete", "loss": 22.0},
    {"x_min": 55, "x_max": 70, "y_min": 45, "y_max": 60, "mat": "glass",    "loss": 8.0},
    {"x_min": 75, "x_max": 87, "y_min": 25, "y_max": 40, "mat": "concrete", "loss": 22.0},
    {"x_min": 85, "x_max": 93, "y_min": 60, "y_max": 68, "mat": "metal",    "loss": 32.0},
]

class FR3_Channel:
    @staticmethod
    def fspl_db(distance_m, freq_hz):
        if distance_m <= 0: distance_m = 0.1
        return 20*math.log10(distance_m) + 20*math.log10(freq_hz) - 147.55

    @staticmethod
    def get_building_penetration_loss(tx_x, tx_y, rx_x, rx_y):
        """Checks if UE is inside a building to apply specific material loss"""
        loss = 0
        for b in BUILDINGS:
            # Simple check: if RX is inside the building bounds
            if b["x_min"] <= rx_x <= b["x_max"] and b["y_min"] <= rx_y <= b["y_max"]:
                loss += b["loss"]
        return loss

    @staticmethod
    def compute_link(tx_power_dbm, tx_x, tx_y, rx_x, rx_y):
        d_los = max(math.hypot(rx_x - tx_x, rx_y - tx_y), 1.0)
        pl_los = FR3_Channel.fspl_db(d_los, FR3_24GHz_Config.CARRIER_FREQ_HZ)
        pen_loss = FR3_Channel.get_building_penetration_loss(tx_x, tx_y, rx_x, rx_y)
        shadow = random.gauss(0, 2.0)
        
        total_pl = pl_los + pen_loss + shadow

        rx_power_dbm = (tx_power_dbm + FR3_24GHz_Config.GNB_ANT_GAIN_DBI + 
                        FR3_24GHz_Config.UE_ANT_GAIN_DBI - total_pl - FR3_24GHz_Config.IMPLEMENTATION_LOSS_DB)

        snr_db = rx_power_dbm - FR3_24GHz_Config.thermal_noise_dbm()
        return {"snr_db": snr_db, "rx_power_dbm": rx_power_dbm, "d_los_m": d_los}

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — MOBILITY MODELS (Ported from C++ Waypoints & Velocity)
# ─────────────────────────────────────────────────────────────────────────────

class MobilityNode:
    def __init__(self, id, type_name, color, target_mbps):
        self.id = id
        self.name = type_name
        self.color = color
        self.target_mbps = target_mbps
        self.x, self.y = 0, 0
        self.connected_gnb = 0
        self.tput_history = collections.deque(maxlen=100)
        self.handover_count = 0

class LinearVehicle(MobilityNode):
    def __init__(self, id, name, color, y_start, speed_kmh, target_mbps):
        super().__init__(id, name, color, target_mbps)
        self.x = 0
        self.y = y_start
        self.vx = speed_kmh / 3.6  # Convert km/h to m/s

    def move(self, dt):
        self.x += self.vx * dt
        if self.x > 100: self.x = 0  # Loop around map
        if self.x < 0: self.x = 100

class Figure8Pedestrian(MobilityNode):
    def __init__(self, id, name, color, cx, cy, scale, target_mbps):
        super().__init__(id, name, color, target_mbps)
        self.cx, self.cy = cx, cy
        self.scale = scale
        self.t = random.uniform(0, 2*math.pi)

    def move(self, dt):
        self.t += 0.5 * dt
        if self.t > 2*math.pi: self.t = 0
        # Parametric lemniscate of Gerono (from your C++ code)
        denom = 1.0 + math.sin(self.t)**2
        self.x = self.cx + (self.scale * math.cos(self.t)) / denom
        self.y = self.cy + (self.scale * math.cos(self.t) * math.sin(self.t)) / denom

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — DIGITAL TWIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DigitalTwinFR3:
    DT = 0.1  # 100ms step
    
    def __init__(self):
        # Ported 5 gNB positions from C++
        self.gnbs = [
            {"id": 0, "x": 25.0, "y": 90.0, "tx_power": 40.0},
            {"id": 1, "x": 75.0, "y": 90.0, "tx_power": 40.0},
            {"id": 2, "x": 50.0, "y": 50.0, "tx_power": 40.0},
            {"id": 3, "x": 25.0, "y": 10.0, "tx_power": 40.0},
            {"id": 4, "x": 75.0, "y": 10.0, "tx_power": 40.0},
        ]
        
        # Select representative UEs to prevent visual clutter
        self.ues = [
            LinearVehicle(0, "eMBB Car", "#00f2ff", 30.0, 80.0, 150.0),    # eMBB Traffic
            LinearVehicle(1, "URLLC Ambulance", "#ff0000", 50.0, 120.0, 20.0), # URLLC Traffic
            Figure8Pedestrian(2, "mMTC Pedestrian 1", "#a29bfe", 30, 30, 15, 2.0), # mMTC
            Figure8Pedestrian(3, "mMTC Pedestrian 2", "#fd79a8", 70, 70, 15, 2.0), # mMTC
        ]
        
        self.sim_time, self.step = 0.0, 0
        self.paused = False
        self._setup_dashboard()

    def _setup_dashboard(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 9), facecolor="#060b14")
        self.fig.suptitle("3GPP 6G NR DIGITAL TWIN | FR3 V2X @ 24 GHz", color="#00f2ff", fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.2, left=0.05, right=0.97, top=0.9, bottom=0.08)

        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_bbu = self.fig.add_subplot(gs[0, 1], facecolor="#0a0f1e")
        self.ax_tput = self.fig.add_subplot(gs[1, :], facecolor="#0a0f1e")

        self.fig.canvas.mpl_connect("key_press_event", lambda e: setattr(self, 'paused', not self.paused) if e.key == ' ' else None)
        self._init_map_artists()

    def _init_map_artists(self):
        ax = self.ax_map
        ax.set_facecolor("#0d1b2a")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("100x100m Urban Grid (5 gNBs + Buildings)", color="white", fontsize=10)

        # Draw Buildings
        for b in BUILDINGS:
            color = "#636e72" if b["mat"] == "concrete" else "#74b9ff" if b["mat"] == "glass" else "#b2bec3"
            ax.add_patch(patches.Rectangle((b["x_min"], b["y_min"]), b["x_max"]-b["x_min"], b["y_max"]-b["y_min"], 
                                           fc=color, ec="black", alpha=0.6, zorder=2))

        # Draw gNBs
        for g in self.gnbs:
            ax.plot(g["x"], g["y"], '^', ms=10, color="lime", zorder=5)
            ax.add_patch(patches.Circle((g["x"], g["y"]), 30, color="lime", fill=False, linestyle="--", alpha=0.2))

        # Track UI Elements
        self.ue_markers, self.ue_lines = [], []
        self.tput_lines = []
        for ue in self.ues:
            m, = ax.plot([], [], "o", ms=10, color=ue.color, zorder=10, mec="white")
            l, = ax.plot([], [], "-", color="white", lw=1, alpha=0.5, zorder=4)
            self.ue_markers.append(m)
            self.ue_lines.append(l)
            
            tl, = self.ax_tput.plot([], [], "-", color=ue.color, lw=2, label=ue.name)
            self.tput_lines.append(tl)

        self.ax_tput.set_title("Live Application Throughput (Mbps)", color="#00f2ff", fontsize=10)
        self.ax_tput.legend(loc="upper left", facecolor="#0a0f1e", fontsize=8)
        self.ax_tput.set_ylim(0, 200)

    def step_simulation(self):
        self.sim_time += self.DT
        self.step += 1

        for ue in self.ues:
            ue.move(self.DT)
            
            # Find Best gNB (Handover Logic)
            best_snr, best_gnb = -1000, 0
            for g in self.gnbs:
                res = FR3_Channel.compute_link(g["tx_power"], g["x"], g["y"], ue.x, ue.y)
                if res["snr_db"] > best_snr:
                    best_snr = res["snr_db"]
                    best_gnb = g["id"]
            
            if ue.connected_gnb != best_gnb:
                ue.handover_count += 1
                ue.connected_gnb = best_gnb

            # Calculate Throughput based on Shannon Capacity
            cap = FR3_24GHz_Config.shannon_capacity_mbps(best_snr) if best_snr > -10 else 0
            actual_mbps = min(cap, ue.target_mbps)
            ue.tput_history.append(actual_mbps)

    def update_dashboard(self):
        t = self.sim_time

        # Update Map
        for i, ue in enumerate(self.ues):
            self.ue_markers[i].set_data([ue.x], [ue.y])
            g = self.gnbs[ue.connected_gnb]
            self.ue_lines[i].set_data([g["x"], ue.x], [g["y"], ue.y])

            # Update Tput Line
            if len(ue.tput_history) > 1:
                times = np.linspace(max(0, t - len(ue.tput_history) * self.DT), t, len(ue.tput_history))
                self.tput_lines[i].set_data(times, list(ue.tput_history))

        self.ax_tput.set_xlim(max(0, t - 10), t + 0.5)

        # Update Stats Panel
        self.ax_bbu.cla()
        self.ax_bbu.set_title("Network Flow Statistics", color="#f9ca24", fontsize=10)
        self.ax_bbu.axis("off")
        
        self.ax_bbu.text(0.05, 0.9, f"Simulation Time: {t:.1f} s", color="white", fontsize=10)
        
        y_offset = 0.75
        for ue in self.ues:
            self.ax_bbu.text(0.05, y_offset, f"{ue.name}:", color=ue.color, fontsize=9, fontweight="bold")
            self.ax_bbu.text(0.1, y_offset - 0.08, f"Target: {ue.target_mbps} Mbps | Actual: {ue.tput_history[-1]:.1f} Mbps", color="#dfe6e9", fontsize=8)
            self.ax_bbu.text(0.1, y_offset - 0.16, f"Handovers: {ue.handover_count} | Serving gNB: {ue.connected_gnb}", color="#b2bec3", fontsize=8)
            y_offset -= 0.25

    def run(self):
        plt.ion()
        print("\n[*] Starting 6G FR3 Simulation. Watch the handovers!")
        while plt.fignum_exists(self.fig.number):
            if not self.paused:
                self.step_simulation()
                if self.step % 2 == 0: self.update_dashboard()
            plt.pause(0.01)
        plt.ioff()

if __name__ == "__main__":
    twin = DigitalTwinFR3()
    twin.run()