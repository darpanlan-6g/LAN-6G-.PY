"""
=============================================================================
  3GPP 6G NR DIGITAL TWIN — FR3 V2X @ 24 GHz
  Frequency: 24 GHz | Bandwidth: 200 MHz | FR3 Upper Limit
  Controls: 
    [Mouse] Move your "Human Traction" Pedestrian (Purple)
    [SPACE] Pause/Play
=============================================================================
"""

import math
import random
import collections
import numpy as np
import matplotlib

# Force interactive backend for live mouse controls
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — PHYSICS & FR3 SYSTEM CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class FR3Config:
    CARRIER_FREQ_HZ = 24.0e9      # 24 GHz FR3
    BANDWIDTH_HZ = 200e6          # 200 MHz Bandwidth
    GNB_ANT_GAIN_DBI = 22.0       # 32x32 MIMO
    UE_ANT_GAIN_DBI = 0.0         # Standard UE
    NOISE_FIGURE_DB = 9.0           
    IMPLEMENTATION_LOSS_DB = 2.0     
    HUMAN_BODY_LOSS_DB = 15.0     # Less severe than THz, but still impactful

    @staticmethod
    def thermal_noise_dbm():
        # -174 + 10*log10(BW) + NF
        return -174 + 10*math.log10(FR3Config.BANDWIDTH_HZ) + FR3Config.NOISE_FIGURE_DB

    @staticmethod
    def shannon_capacity_mbps(snr_db):
        if snr_db < -10: return 0
        return (FR3Config.BANDWIDTH_HZ * math.log2(1 + 10**(snr_db/10))) / 1e6

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Calculates if an object intersects the signal beam."""
    l2 = (x2 - x1)**2 + (y2 - y1)**2
    if l2 == 0: return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
    proj_x, proj_y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    return math.hypot(px - proj_x, py - proj_y)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — MOBILITY & AGENTS
# ─────────────────────────────────────────────────────────────────────────────

class Entity:
    def __init__(self, id_name, label, color, x, y, vx, vy, target_mbps=0):
        self.id, self.label, self.color = id_name, label, color
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.target_mbps = target_mbps
        self.connected_gnb = None
        self.tput_history = collections.deque([0]*100, maxlen=100)
        self.snr_history = collections.deque([-10]*100, maxlen=100)
        self.pen_loss = 0
        self.d_los = 1.0

    def move(self, dt, bounds):
        self.x += self.vx * dt
        self.y += self.vy * dt
        # Bounce off edges
        if self.x > bounds["x_max"] or self.x < bounds["x_min"]: self.vx *= -1
        if self.y > bounds["y_max"] or self.y < bounds["y_min"]: self.vy *= -1

class MousePedestrian(Entity):
    def __init__(self, id_name, label, color, target_mbps):
        super().__init__(id_name, label, color, 50, 50, 0, 0, target_mbps)
        self.tx, self.ty = 50, 50

    def move(self, dt, bounds):
        # Human traction logic: moves towards mouse cursor
        dist = math.hypot(self.tx - self.x, self.ty - self.y)
        if dist > 0.5:
            speed = 5.0 # Pedestrian sprint speed
            self.x += (self.tx - self.x) / dist * speed * dt
            self.y += (self.ty - self.y) / dist * speed * dt

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — SCENARIO ARCHITECTURE (MAPPED FROM NS-3)
# ─────────────────────────────────────────────────────────────────────────────

def build_fr3_scenario():
    return {
        "name": "3GPP 6G NR FR3 V2X @ 24 GHz", 
        "b": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "gnbs": [
            {"id": "G0", "x": 25, "y": 90, "tx": 40}, 
            {"id": "G1", "x": 75, "y": 90, "tx": 40},
            {"id": "G2", "x": 50, "y": 50, "tx": 40}, 
            {"id": "G3", "x": 25, "y": 10, "tx": 40},
            {"id": "G4", "x": 75, "y": 10, "tx": 40}
        ],
        "walls": [
            {"x": 10, "y": 10, "w": 10, "h": 15, "mat": "concrete", "loss": 22.0, "c": "#64748b"},
            {"x": 25, "y": 15, "w": 10, "h": 15, "mat": "glass", "loss": 8.0, "c": "#38bdf8"},
            {"x": 40, "y": 20, "w": 10, "h": 15, "mat": "metal", "loss": 32.0, "c": "#334155"},
            {"x": 75, "y": 25, "w": 12, "h": 15, "mat": "concrete", "loss": 22.0, "c": "#64748b"}
        ],
        "ues": [
            # Human Traction Node (Purple)
            MousePedestrian("U0", "YOU (Mouse)", "#a855f7", 50.0),
            
            # eMBB Cars (Green)
            Entity("C1", "Car 1", "#22c55e", 5, 30, 22, 0, 150.0),
            Entity("C2", "Car 2", "#22c55e", 5, 35, 25, 0, 150.0),
            
            # URLLC Emergency (Red)
            Entity("E1", "Emergency", "#ef4444", 20, 50, 33, 0, 20.0),
            
            # URLLC Truck (Orange)
            Entity("T1", "Truck 1", "#f97316", 90, 75, -18, 0, 50.0),
            
            # mMTC AI Pedestrian (Purple)
            Entity("P1", "AI Ped", "#a855f7", 30, 30, 1, 1.5, 5.0)
        ],
        "rsus": [
            {"id": "R1", "x": 30, "y": 50}, {"id": "R2", "x": 70, "y": 50}
        ]
    }

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class FR3DigitalTwin:
    DT = 0.05

    def __init__(self):
        self.sim_time, self.step, self.paused = 0.0, 0, False
        self.cfg = build_fr3_scenario()
        self.setup_dashboard()

    def setup_dashboard(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 9), facecolor="#020617")
        self.fig.suptitle("6G NR FR3 V2X DIGITAL TWIN (24 GHz) | Human Traction Active", color="#38bdf8", fontsize=16, fontweight="bold")
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3, left=0.05, right=0.97, top=0.90, bottom=0.08)

        self.ax_map = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_tput = self.fig.add_subplot(gs[0, 2], facecolor="#0f172a")
        self.ax_lb = self.fig.add_subplot(gs[1, 2], facecolor="#0f172a")

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse)
        
        self.init_map()

    def init_map(self):
        self.ax_map.set_facecolor("#0f172a")
        self.ax_map.set_xlim(0, 100); self.ax_map.set_ylim(0, 100)
        self.ax_map.set_xticks([]); self.ax_map.set_yticks([])

        # Draw Buildings
        for w in self.cfg["walls"]:
            self.ax_map.add_patch(patches.Rectangle((w["x"], w["y"]), w["w"], w["h"], fc=w["c"], alpha=0.8, zorder=2))
            self.ax_map.text(w["x"]+1, w["y"]+1, w["mat"], color="white", fontsize=7, zorder=3)

        # Draw gNBs (Blue)
        for g in self.cfg["gnbs"]:
            self.ax_map.plot(g["x"], g["y"], '^', ms=12, color="#3b82f6", zorder=10)
            
        # Draw RSUs (Yellow)
        for r in self.cfg["rsus"]:
            self.ax_map.plot(r["x"], r["y"], 's', ms=8, color="#eab308", zorder=9)

        # Init Dynamic UI Elements
        self.ue_ui = []
        for u in self.cfg["ues"]:
            m, = self.ax_map.plot([], [], 'o', ms=10, color=u.color, zorder=12, mec="white")
            l = self.ax_map.text(0, 0, u.label, color=u.color, fontsize=8, fontweight="bold", ha="center", zorder=13)
            ray, = self.ax_map.plot([], [], '--', color=u.color, lw=1.5, zorder=5)
            self.ue_ui.append({"m": m, "l": l, "ray": ray})

    def on_key(self, event):
        if event.key == " ": self.paused = not self.paused

    def on_mouse(self, event):
        # Update Target for Human Traction Pedestrian
        if event.inaxes == self.ax_map and event.xdata:
            for u in self.cfg["ues"]:
                if isinstance(u, MousePedestrian):
                    u.tx, u.ty = event.xdata, event.ydata

    def calculate_physics(self):
        for u in self.cfg["ues"]: u.move(self.DT, self.cfg["b"])

        for ue in self.cfg["ues"]:
            best_snr, best_g = -1000, self.cfg["gnbs"][0]
            ue.pen_loss = 0
            
            for g in self.cfg["gnbs"]:
                d = max(math.hypot(ue.x - g["x"], ue.y - g["y"]), 1.0)
                # 3GPP Free Space Path Loss for 24 GHz
                pl_los = 20*math.log10(d) + 20*math.log10(FR3Config.CARRIER_FREQ_HZ) - 147.55
                
                # Check Building Intersections
                pen = sum(w["loss"] for w in self.cfg["walls"] if point_to_segment_dist(w["x"]+w["w"]/2, w["y"]+w["h"]/2, g["x"], g["y"], ue.x, ue.y) < max(w["w"], w["h"])/2)
                
                rx_pwr = g["tx"] + FR3Config.GNB_ANT_GAIN_DBI + FR3Config.UE_ANT_GAIN_DBI - pl_los - pen - FR3Config.IMPLEMENTATION_LOSS_DB
                snr = rx_pwr - FR3Config.thermal_noise_dbm()
                
                if snr > best_snr:
                    best_snr, best_g = snr, g
                    ue.pen_loss, ue.d_los = pen, d

            ue.connected_gnb = best_g["id"]
            ue.snr_db = best_snr
            ue.serving_x, ue.serving_y = best_g["x"], best_g["y"]
            
            # Cap capacity based on target limits
            raw_tput = FR3Config.shannon_capacity_mbps(best_snr)
            ue.tput_history.append(min(raw_tput, ue.target_mbps))
            ue.snr_history.append(best_snr)

    def render(self):
        # Update Map UI
        for i, ue in enumerate(self.cfg["ues"]):
            ui = self.ue_ui[i]
            ui["m"].set_data([ue.x], [ue.y])
            ui["l"].set_position((ue.x, ue.y + 3))
            
            # Show blockage via ray styling
            ls = ":" if ue.pen_loss > 0 else "-"
            ui["ray"].set_data([ue.serving_x, ue.x], [ue.serving_y, ue.y])
            ui["ray"].set_linestyle(ls)

        # Update Charts sparingly to save frames
        if self.step % 4 == 0:
            # Throughput
            self.ax_tput.cla(); self.ax_tput.set_title("Live Throughput (Mbps)", color="#38bdf8", fontsize=10)
            for u in self.cfg["ues"]: 
                self.ax_tput.plot(u.tput_history, color=u.color, lw=2, label=f"{u.label} ({u.tput_history[-1]:.0f})")
            self.ax_tput.legend(loc="upper left", fontsize=8, facecolor="#0f172a", edgecolor="none", labelcolor="white")
            self.ax_tput.set_ylim(0, 160); self.ax_tput.set_facecolor("#0f172a"); self.ax_tput.set_xticks([])

            # Dynamic Link Budget for the Human Driven Pedestrian
            u0 = self.cfg["ues"][0]
            self.ax_lb.cla(); self.ax_lb.set_title(f"Link Budget: {u0.label}", color="#38bdf8", fontsize=10)
            fspl = 20*math.log10(u0.d_los) + 20*math.log10(FR3Config.CARRIER_FREQ_HZ) - 147.55
            vals = [self.cfg["gnbs"][0]["tx"], FR3Config.GNB_ANT_GAIN_DBI, -fspl, -u0.pen_loss, u0.snr_db+FR3Config.thermal_noise_dbm(), FR3Config.thermal_noise_dbm(), u0.snr_db]
            labels = ["Tx", "Gain", "-FSPL", "-Penetration", "Rx", "Noise Floor", "SNR"]
            colors = ["#10b981" if v>=0 else "#ef4444" for v in vals]
            
            bars = self.ax_lb.barh(labels, vals, color=colors)
            for b, v in zip(bars, vals): 
                self.ax_lb.text(v+(1 if v>=0 else -1), b.get_y()+0.3, f"{v:+.0f}", color="white", fontsize=8, ha="left" if v>=0 else "right")
            self.ax_lb.set_facecolor("#0f172a"); self.ax_lb.tick_params(colors="white", labelsize=8)

    def run(self):
        plt.ion()
        print("\n" + "="*70)
        print("  🚀 FR3 V2X DIGITAL TWIN STARTED")
        print("  Move your mouse over the map window to control the Purple Pedestrian!")
        print("="*70 + "\n")

        while plt.fignum_exists(self.fig.number):
            if not self.paused:
                self.sim_time += self.DT; self.step += 1
                self.calculate_physics()
                self.render()
            plt.pause(0.01)

if __name__ == "__main__":
    FR3DigitalTwin().run()