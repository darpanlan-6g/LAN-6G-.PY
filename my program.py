"""
=============================================================================
  3GPP 6G NR DIGITAL TWIN - FR3 V2X @ 24 GHz
  Real-time Matplotlib Simulation translated from NS-3 C++ NetAnim Model
=============================================================================
Architecture:
  - Band: FR3 (24 GHz), 200 MHz Bandwidth, 40 dBm TX Power, Massive MIMO.
  - Environment: 100x100m Urban Grid with Concrete, Glass, and Metal buildings.
  - Mobility: 
      * Vehicles: Cars, Trucks, Emergency (Constant Velocity, wrapping).
      * Pedestrians: Random Walk & Figure-8 trajectories.
      * Infrastructure: gNBs (Macro cells) & RSUs.
  - Traffic Profiles:
      * URLLC: Ultra-low latency target (< 10ms).
      * eMBB: High throughput target (> 50 Mbps).
      * mMTC: Massive IoT packet delivery.

Dashboard Panels:
  1. Live V2X Map (Nodes, Buildings, and Active Links)
  2. Network KPIs & Flow Monitor (URLLC, eMBB, mMTC)
  3. Live SINR Tracker (dB)
  4. Application Throughput Timeline
=============================================================================
"""

import matplotlib
import os
import math
import random
import collections
import numpy as np

# Use appropriate backend
if os.environ.get("DISPLAY", ""):
    try: matplotlib.use("TkAgg")
    except: matplotlib.use("Agg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe

# =============================================================================
# 1. CONFIGURATION (Mapped from C++ FR3 parameters)
# =============================================================================
class FR3Config:
    FREQ_HZ       = 24.0e9    # 24 GHz
    BW_HZ         = 200.0e6   # 200 MHz channel bandwidth
    TX_POWER_DBM  = 40.0      # 10 Watts
    ANT_GAIN_DBI  = 22.0      # 32x32 MIMO
    UE_TX_DBM     = 23.0      # Standard UE TX power
    NOISE_FLOOR   = -90.0     # dBm
    NUMEROLOGY    = 2         # 60 kHz SCS

# =============================================================================
# 2. ENVIRONMENT & CHANNEL MODEL (Building Penetration + FSPL)
# =============================================================================
BUILDINGS = [
    # (x_min, x_max, y_min, y_max, material, loss_db, color)
    (10, 20, 10, 25, "concrete", 22.0, "#7f8c8d"),
    (25, 35, 15, 30, "glass",     8.0, "#3498db"),
    (40, 50, 20, 35, "metal",    32.0, "#95a5a6"),
    (15, 25, 40, 55, "concrete", 22.0, "#7f8c8d"),
    (55, 70, 45, 60, "glass",     8.0, "#3498db"),
    (75, 87, 25, 40, "concrete", 22.0, "#7f8c8d"),
    (85, 93, 60, 68, "metal",    32.0, "#95a5a6"),
    (20, 30, 70, 85, "concrete", 22.0, "#7f8c8d"),
]

def check_building_intersection(x1, y1, x2, y2):
    """Simple bounding box intersection for penetration loss."""
    total_loss = 0.0
    for (xmin, xmax, ymin, ymax, mat, loss, c) in BUILDINGS:
        # Fast AABB check with the line bounding box
        if max(x1, x2) < xmin or min(x1, x2) > xmax: continue
        if max(y1, y2) < ymin or min(y1, y2) > ymax: continue
        # If the ray passes near/through the building, add loss
        total_loss += loss
    return total_loss

def calculate_sinr(tx_x, tx_y, rx_x, rx_y):
    d = math.hypot(rx_x - tx_x, rx_y - tx_y)
    d = max(d, 1.0)
    
    # Free Space Path Loss (FSPL)
    fspl = 20 * math.log10(d) + 20 * math.log10(FR3Config.FREQ_HZ) - 147.55
    
    # Building Penetration
    pen_loss = check_building_intersection(tx_x, tx_y, rx_x, rx_y)
    
    rx_power = (FR3Config.TX_POWER_DBM + FR3Config.ANT_GAIN_DBI * 2 
                - fspl - pen_loss)
    
    sinr_db = rx_power - FR3Config.NOISE_FLOOR
    return sinr_db, rx_power

# =============================================================================
# 3. NETWORK NODES & MOBILITY
# =============================================================================
class Node:
    def __init__(self, id, type_name, x, y, color, traffic_type):
        self.id = id
        self.type_name = type_name
        self.x = x
        self.y = y
        self.color = color
        self.traffic_type = traffic_type
        
        self.sinr_history = collections.deque(maxlen=100)
        self.tput_history = collections.deque(maxlen=100)
        self.serving_gnb = None
        
        # QoS metrics
        self.rx_bytes = 0
        self.latency_ms = 0.0
        
    def get_shannon_tput(self, sinr_db):
        snr_lin = 10 ** (sinr_db / 10.0)
        # Assuming 200 MHz divided amongst active users; simplified
        bw_alloc = FR3Config.BW_HZ / 20  
        return (bw_alloc * math.log2(1 + snr_lin)) / 1e6 # Mbps

class MobileNode(Node):
    def __init__(self, id, type_name, x, y, vx, vy, color, traffic_type, move_type="linear"):
        super().__init__(id, type_name, x, y, color, traffic_type)
        self.vx = vx
        self.vy = vy
        self.move_type = move_type
        self.time = 0.0
        
    def move(self, dt):
        self.time += dt
        if self.move_type == "linear":
            self.x += self.vx * dt
            self.y += self.vy * dt
            if self.x > 100: self.x = 0
            if self.x < 0: self.x = 100
        elif self.move_type == "figure8":
            scale = 15.0
            t = self.time * 0.5 # speed factor
            denom = 1.0 + math.sin(t)**2
            self.x = 70 + (scale * math.cos(t)) / denom
            self.y = 70 + (scale * math.cos(t) * math.sin(t)) / denom
        elif self.move_type == "random":
            self.x += random.uniform(-1, 1) * dt
            self.y += random.uniform(-1, 1) * dt
            self.x = max(0, min(100, self.x))
            self.y = max(0, min(100, self.y))

# =============================================================================
# 4. DIGITAL TWIN DASHBOARD / MAIN LOOP
# =============================================================================
class FR3DigitalTwin:
    def __init__(self):
        self.dt = 0.1
        self.sim_time = 0.0
        self.paused = False
        
        self._setup_nodes()
        self._setup_dashboard()
        
    def _setup_nodes(self):
        # gNBs (Blue)
        self.gnbs = [
            Node(0, "gNB-0", 25, 90, "#0984e3", "infra"),
            Node(1, "gNB-1", 75, 90, "#0984e3", "infra"),
            Node(2, "gNB-2", 50, 50, "#0984e3", "infra"),
            Node(3, "gNB-3", 25, 10, "#0984e3", "infra"),
            Node(4, "gNB-4", 75, 10, "#0984e3", "infra")
        ]
        
        # RSUs (Yellow)
        self.rsus = [
            Node(0, "RSU-0", 30, 50, "#f1c40f", "infra"),
            Node(1, "RSU-1", 70, 50, "#f1c40f", "infra")
        ]
        
        self.ues = []
        # Cars (Green, URLLC) - 80 km/h = 22 m/s
        for i in range(7):
            self.ues.append(MobileNode(i, f"Car-{i}", 5, 30+i*5, 22, 0, "#00b894", "URLLC", "linear"))
            
        # Trucks (Orange, eMBB)
        self.ues.append(MobileNode(7, "Truck-1", 10, 70, 18, 0, "#e17055", "eMBB", "linear"))
        self.ues.append(MobileNode(8, "Truck-2", 90, 75, -16, 0, "#e17055", "eMBB", "linear"))
        
        # Emergency (Red, URLLC) - 120 km/h = 33 m/s
        self.ues.append(MobileNode(9, "Emergency", 20, 50, 33, 0, "#d63031", "URLLC", "linear"))
        
        # Pedestrians (Purple, mMTC/eMBB)
        for i in range(3):
            self.ues.append(MobileNode(10+i, f"Ped-{i}", random.uniform(20,80), random.uniform(20,80), 0, 0, "#9b59b6", "mMTC", "random"))
        for i in range(2):
            self.ues.append(MobileNode(13+i, f"Ped-F8-{i}", 70, 70, 0, 0, "#9b59b6", "eMBB", "figure8"))

    def _setup_dashboard(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 9), facecolor="#121212")
        self.fig.suptitle("3GPP 6G NR FR3 V2X @ 24 GHz DIGITAL TWIN", color="#00cec9", fontsize=14, fontweight="bold")
        
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Map
        self.ax_map = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_map.set_facecolor("#1e272e")
        self.ax_map.set_xlim(0, 100); self.ax_map.set_ylim(0, 100)
        self.ax_map.set_title("Live V2X Map (100x100m)", color="#dfe6e9", fontsize=10)
        
        # Draw Buildings
        for (xmin, xmax, ymin, ymax, mat, loss, c) in BUILDINGS:
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fc=c, alpha=0.7, ec="white", lw=0.5)
            self.ax_map.add_patch(rect)
            self.ax_map.text(xmin+(xmax-xmin)/2, ymin+(ymax-ymin)/2, mat.upper(), color="black", fontsize=5, ha="center", va="center", rotation=45)
            
        # UI Elements for map
        self.ue_scatters = [self.ax_map.plot([], [], "o", color=ue.color, ms=8)[0] for ue in self.ues]
        self.gnb_scatters = [self.ax_map.plot([g.x], [g.y], "^", color=g.color, ms=10)[0] for g in self.gnbs]
        self.rsu_scatters = [self.ax_map.plot([r.x], [r.y], "s", color=r.color, ms=7)[0] for r in self.rsus]
        self.link_lines = [self.ax_map.plot([], [], "--", color="white", alpha=0.3, lw=1)[0] for _ in self.ues]
        
        # Panel 2: Throughput Line Chart
        self.ax_tput = self.fig.add_subplot(gs[0, 2])
        self.ax_tput.set_facecolor("#1e272e")
        self.ax_tput.set_title("Live Application Throughput (Mbps)", color="#00cec9", fontsize=9)
        self.ax_tput.set_xlim(0, 10)
        self.ax_tput.set_ylim(0, 400)
        self.tput_lines = {}
        for ue in self.ues:
            if ue.traffic_type in ["URLLC", "eMBB"]:
                ln, = self.ax_tput.plot([], [], "-", color=ue.color, lw=1.5, alpha=0.8)
                self.tput_lines[ue.id] = ln

        # Panel 3: SINR Bar Chart
        self.ax_sinr = self.fig.add_subplot(gs[1, 2])
        self.ax_sinr.set_facecolor("#1e272e")
        self.ax_sinr.set_title("Instantaneous SINR (dB) by UE", color="#fdcb6e", fontsize=9)
        self.ax_sinr.set_ylim(-10, 50)
        self.sinr_bars = self.ax_sinr.bar([ue.type_name[:5] for ue in self.ues[:10]], [0]*10, color=[ue.color for ue in self.ues[:10]])
        self.ax_sinr.tick_params(axis='x', rotation=45, labelsize=7)
        self.ax_sinr.axhline(0, color="red", lw=1, ls="--")

        # Time Text
        self.time_text = self.ax_map.text(2, 95, "", color="white", fontsize=10, fontweight="bold")
        
        self.fig.canvas.mpl_connect("key_press_event", lambda e: setattr(self, 'paused', not self.paused) if e.key == ' ' else None)

    def step(self):
        if self.paused: return
        self.sim_time += self.dt
        self.time_text.set_text(f"Sim Time: {self.sim_time:.1f}s | FR3 @ 24GHz")
        
        # 1. Move UEs
        for ue in self.ues:
            ue.move(self.dt)
            
        # 2. Attach to best gNB & Calc SINR
        current_sinrs = []
        for i, ue in enumerate(self.ues):
            best_sinr = -999
            best_gnb = None
            
            for gnb in self.gnbs:
                sinr, _ = calculate_sinr(gnb.x, gnb.y, ue.x, ue.y)
                if sinr > best_sinr:
                    best_sinr = sinr
                    best_gnb = gnb
                    
            ue.serving_gnb = best_gnb
            ue.sinr_history.append(best_sinr)
            
            # Draw Links
            self.link_lines[i].set_data([ue.x, best_gnb.x], [ue.y, best_gnb.y])
            self.ue_scatters[i].set_data([ue.x], [ue.y])
            
            # Traffic Processing
            tput = ue.get_shannon_tput(best_sinr)
            if ue.traffic_type == "mMTC": tput *= 0.1  # small packets
            if best_sinr < 0: tput = 0  # Connection dropped
            ue.tput_history.append(tput)
            
            if i < 10: current_sinrs.append(best_sinr)
            
        # 3. Update Plots
        t_max = self.sim_time
        t_min = max(0, t_max - 10)
        self.ax_tput.set_xlim(t_min, t_max + 0.5)
        
        for ue in self.ues:
            if ue.id in self.tput_lines:
                x_data = np.linspace(max(0, self.sim_time - len(ue.tput_history)*self.dt), self.sim_time, len(ue.tput_history))
                self.tput_lines[ue.id].set_data(x_data, ue.tput_history)
                
        for bar, h in zip(self.sinr_bars, current_sinrs):
            bar.set_height(h)
            
        plt.pause(0.01)

    def run(self):
        plt.ion()
        print("Starting 3GPP 6G NR FR3 V2X Digital Twin...")
        print("Press [SPACE] to pause/unpause.")
        while plt.fignum_exists(self.fig.number):
            self.step()
        print("Simulation ended.")

if __name__ == "__main__":
    twin = FR3DigitalTwin()
    twin.run()