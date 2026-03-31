"""
=============================================================================
  6G THz DIGITAL TWIN — COMMUTE SCENARIO WITH SNR HEATMAP
  Scenario: Commuter moving Home → Street → Office
  Features: Multi-gNB Handover, Wall Penetration Physics, Contour Heatmap
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
import matplotlib.patheffects as pe
import numpy as np
import math
import collections
import random

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — SYSTEM CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class THz6GConfig:
    CARRIER_FREQ_HZ  = 300e9       
    BANDWIDTH_HZ     = 40e9        
    GNB_ANT_GAIN_DBI = 35          
    UE_ANT_GAIN_DBI  = 20          
    NOISE_FIGURE_DB  = 9           
    IMPLEMENTATION_LOSS_DB = 2     
    UDP_PACKET_BYTES = 9000        

    MCS_TABLE = [
        (-5,  "QPSK  R1/5",  0.40), (0,   "QPSK  R1/3",  0.66),
        (4,   "QPSK  R1/2",  1.00), (8,   "16QAM R1/2",  2.00),
        (12,  "64QAM R2/3",  4.00), (16,  "256QAM R3/4", 6.00),
        (20,  "1024QAM R4/5",8.00), (25,  "4096QAM R9/10",10.0),
    ]

    @staticmethod
    def thermal_noise_dbm():
        return -174 + 10*math.log10(THz6GConfig.BANDWIDTH_HZ) + THz6GConfig.NOISE_FIGURE_DB

    @staticmethod
    def select_mcs(snr_db):
        selected = THz6GConfig.MCS_TABLE[0]
        for row in THz6GConfig.MCS_TABLE:
            if snr_db >= row[0]: selected = row
        return selected[1], selected[2]

    @staticmethod
    def shannon_capacity_gbps(snr_db):
        snr_lin = 10**(snr_db/10)
        return THz6GConfig.BANDWIDTH_HZ * math.log2(1 + snr_lin) / 1e9

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — THz CHANNEL & HEATMAP PHYSICS
# ─────────────────────────────────────────────────────────────────────────────

class THz_Channel:
    ABSORPTION_COEFF_DB_PER_M = 0.0015  
    WALL_PENETRATION_LOSS_DB = 30.0  # Heavy loss for 300 GHz through exterior walls

    @staticmethod
    def fspl_db(distance_m, freq_hz):
        if distance_m <= 0: distance_m = 0.1
        return 20*math.log10(distance_m) + 20*math.log10(freq_hz) - 147.55

    @staticmethod
    def get_wall_loss(tx_x, rx_x):
        """Calculates if the signal crosses a building boundary (x=40 or x=160)."""
        loss = 0
        # If TX is in street (40-160) and RX is in Home (<40) or Office (>160)
        if 40 <= tx_x <= 160 and (rx_x < 40 or rx_x > 160):
            loss += THz_Channel.WALL_PENETRATION_LOSS_DB
        # If TX is in Office (>160) and RX is in street (<160)
        if tx_x > 160 and rx_x < 160:
            loss += THz_Channel.WALL_PENETRATION_LOSS_DB
        return loss

    @staticmethod
    def compute_link(tx_power_dbm, tx_x, tx_y, rx_x, rx_y):
        d_los = max(math.hypot(rx_x - tx_x, rx_y - tx_y), 1.0)
        pl_los = THz_Channel.fspl_db(d_los, THz6GConfig.CARRIER_FREQ_HZ)
        abs_los = THz_Channel.ABSORPTION_COEFF_DB_PER_M * d_los
        wall_loss = THz_Channel.get_wall_loss(tx_x, rx_x)
        shadow = random.gauss(0, 1.0)
        
        total_pl_los = pl_los + abs_los + wall_loss + shadow

        rx_power_dbm = (tx_power_dbm + THz6GConfig.GNB_ANT_GAIN_DBI + 
                        THz6GConfig.UE_ANT_GAIN_DBI - total_pl_los - THz6GConfig.IMPLEMENTATION_LOSS_DB)

        noise_floor = THz6GConfig.thermal_noise_dbm()
        snr_db = rx_power_dbm - noise_floor

        return {"snr_db": snr_db, "rx_power_dbm": rx_power_dbm, "d_los_m": d_los, "wall_loss": wall_loss}

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — CORE NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class BBU:
    def __init__(self, flows_config):
        self.flows = flows_config
        self.ecpri_used_gbps = 0

    def tick(self, dt_s, channel_capacity_per_flow):
        actual_per_flow = []
        total_ecpri = 0
        for i, flow in enumerate(self.flows):
            cap = channel_capacity_per_flow[i]
            actual = max(0, min(flow["target_gbps"], cap))
            pkts = int((actual * 1e9 * dt_s / 8) / THz6GConfig.UDP_PACKET_BYTES)
            flow["tx_packets"] += pkts
            
            overload = max(0, flow["target_gbps"] - cap)
            flow["queue_depth"] = min(5000, max(0, flow["queue_depth"] + int(overload * 1e5 * dt_s) - pkts // 2))
            
            total_ecpri += actual
            actual_per_flow.append(actual)
        self.ecpri_used_gbps = total_ecpri
        return actual_per_flow

class UserEquipment:
    def __init__(self, cfg):
        self.id = cfg['id']
        self.name = cfg['name']
        self.x, self.y = cfg['path'][0]
        self.path = cfg['path']
        self.waypoint_idx = 1
        self.speed = cfg['speed']
        self.color = cfg['color']
        self.target_gbps = cfg['target_gbps']
        self.connected_gnb_idx = 0
        
        self.rx_packets = 0
        self.lost_packets = 0
        self.tput_history = collections.deque(maxlen=200)

    def move_along_path(self, dt_s):
        if self.waypoint_idx < len(self.path):
            tx, ty = self.path[self.waypoint_idx]
            dist = math.hypot(tx - self.x, ty - self.y)
            if dist < 0.5:
                self.waypoint_idx += 1 # Reached waypoint, move to next
            else:
                self.x += (tx - self.x) / dist * self.speed * dt_s
                self.y += (ty - self.y) / dist * self.speed * dt_s

    def receive(self, actual_gbps, snr_db, tx_packets):
        per = max(0, min(1, 0.1 * (1 - snr_db / 15)))
        rx_pkts = int(tx_packets * (1 - per))
        self.rx_packets += rx_pkts
        self.lost_packets += tx_packets - rx_pkts
        self.tput_history.append(actual_gbps)
        return per

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — DIGITAL TWIN ENGINE WITH HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

class DigitalTwin:
    DT = 0.05
    
    def __init__(self):
        # Two gNBs: One Macro on the street, One Small-Cell inside the Office
        self.gnbs = [
            {"name": "Street Macro", "x": 100, "y": 80, "tx_power": 46},
            {"name": "Office Pico", "x": 180, "y": 30, "tx_power": 20}
        ]
        
        # Commuter starts at Home, walks to street, walks to office building
        self.ues = [UserEquipment({
            "id": 0, "name": "Commuter-UE", "speed": 6.0, "target_gbps": 50.0, "color": "#00f2ff",
            "path": [(20, 50), (45, 50), (45, 20), (165, 20), (180, 50)]
        })]
        
        self.bbu = BBU([{"id": 0, "name": "Commuter", "target_gbps": 50.0, "tx_packets": 0, "queue_depth": 0, "color": "#00f2ff"}])
        self.sim_time, self.step = 0.0, 0
        self.paused = False

        self._setup_dashboard()

    def _setup_dashboard(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(18, 10), facecolor="#060b14")
        self.fig.suptitle("6G THz DIGITAL TWIN | Commute Scenario with Live SNR Heatmap", color="#00f2ff", fontsize=15, fontweight="bold")
        gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3, left=0.05, right=0.97, top=0.92, bottom=0.08)

        self.ax_map = self.fig.add_subplot(gs[0, :2])
        self.ax_tput = self.fig.add_subplot(gs[1, :2], facecolor="#0a0f1e")
        self.ax_bbu = self.fig.add_subplot(gs[:, 2], facecolor="#0a0f1e")

        self.fig.canvas.mpl_connect("key_press_event", lambda e: setattr(self, 'paused', not self.paused) if e.key == ' ' else None)
        self._init_map_and_heatmap()

    def _init_map_and_heatmap(self):
        ax = self.ax_map
        ax.set_facecolor("#0d1b2a")
        ax.set_xlim(0, 200); ax.set_ylim(0, 100)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Urban Heatmap: Home (Left) → Street (Center) → Office (Right)", color="white", fontsize=10)

        # Draw Buildings Outline
        ax.add_patch(patches.Rectangle((0, 0), 40, 100, fc="none", ec="#ff6b6b", lw=2, linestyle="--", zorder=5))
        ax.text(5, 90, "HOME", color="#ff6b6b", fontweight="bold", zorder=5)
        
        ax.add_patch(patches.Rectangle((160, 0), 40, 100, fc="none", ec="#f9ca24", lw=2, linestyle="--", zorder=5))
        ax.text(165, 90, "OFFICE", color="#f9ca24", fontweight="bold", zorder=5)

        # GENERATE SNR HEATMAP (Pre-calculated for performance)
        print("[*] Generating 300GHz SNR Heatmap. Calculating wall penetrations...")
        X, Y = np.meshgrid(np.linspace(0, 200, 100), np.linspace(0, 100, 50))
        snr_map = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                best_snr = -100
                for gnb in self.gnbs:
                    res = THz_Channel.compute_link(gnb["tx_power"], gnb["x"], gnb["y"], X[i,j], Y[i,j])
                    best_snr = max(best_snr, res["snr_db"])
                snr_map[i,j] = best_snr

        # Plot Heatmap
        c = ax.contourf(X, Y, snr_map, levels=np.linspace(-20, 60, 30), cmap="inferno", alpha=0.6, zorder=1)
        plt.colorbar(c, ax=ax, label="SNR (dB)", fraction=0.046, pad=0.04).ax.tick_params(colors='white')

        # Draw gNBs
        for gnb in self.gnbs:
            ax.plot(gnb["x"], gnb["y"], '^', ms=12, color="lime", zorder=10)
            ax.text(gnb["x"], gnb["y"]+5, gnb["name"], color="lime", ha="center", fontsize=8, zorder=10)

        # UI Elements for UE
        self.ue_marker, = ax.plot([], [], "o", ms=12, color="#00f2ff", zorder=11, mec="white")
        self.link_line, = ax.plot([], [], "-", color="white", lw=1.5, zorder=10)
        
        self.tput_line, = self.ax_tput.plot([], [], "-", color="#00f2ff", lw=2)
        self.ax_tput.set_title("Live UDP Throughput (Gbps)", color="#00f2ff", fontsize=10)
        self.ax_tput.set_ylim(0, 60)

    def step_simulation(self):
        self.sim_time += self.DT
        self.step += 1
        ue = self.ues[0]
        ue.move_along_path(self.DT)

        # Handover Logic: Find best gNB
        best_snr = -1000
        best_gnb_idx = 0
        for i, gnb in enumerate(self.gnbs):
            res = THz_Channel.compute_link(gnb["tx_power"], gnb["x"], gnb["y"], ue.x, ue.y)
            if res["snr_db"] > best_snr:
                best_snr = res["snr_db"]
                best_gnb_idx = i
                
        ue.connected_gnb_idx = best_gnb_idx

        # Calculate Capacity & BBU processing
        cap = THz6GConfig.shannon_capacity_gbps(best_snr) if best_snr > -5 else 0
        mcs, _ = THz6GConfig.select_mcs(best_snr)
        
        actual_gbps = self.bbu.tick(self.DT, [cap])[0]
        tx_pkts = max(1, self.bbu.flows[0]["tx_packets"] // 100)
        ue.receive(actual_gbps, best_snr, tx_pkts)

        return best_snr, cap, mcs

    def update_dashboard(self, best_snr, cap, mcs):
        ue = self.ues[0]
        gnb = self.gnbs[ue.connected_gnb_idx]

        # Update Map
        self.ue_marker.set_data([ue.x], [ue.y])
        self.link_line.set_data([gnb["x"], ue.x], [gnb["y"], ue.y])

        # Update Tput Line
        t = self.sim_time
        self.ax_tput.set_xlim(max(0, t - 10), t + 0.5)
        if len(ue.tput_history) > 1:
            times = np.linspace(max(0, t - len(ue.tput_history) * self.DT), t, len(ue.tput_history))
            self.tput_line.set_data(times, list(ue.tput_history))

        # Update BBU/Stats Panel
        self.ax_bbu.cla()
        self.ax_bbu.set_title("Live Network Stats", color="#f9ca24", fontsize=10)
        self.ax_bbu.axis("off")
        
        stats = [
            ("Time", f"{t:.1f} s"),
            ("Location", "Home" if ue.x < 40 else "Office" if ue.x > 160 else "Street"),
            ("Connected gNB", gnb["name"]),
            ("Current SNR", f"{best_snr:.1f} dB"),
            ("MCS", mcs),
            ("Max Channel Cap", f"{cap:.1f} Gbps"),
            ("Actual UDP Tput", f"{ue.tput_history[-1] if ue.tput_history else 0:.1f} Gbps"),
            ("BBU Queue Depth", f"{self.bbu.flows[0]['queue_depth']} pkts")
        ]
        
        for i, (label, val) in enumerate(stats):
            self.ax_bbu.text(0.1, 0.9 - i*0.1, label + ":", color="#a29bfe", fontsize=10, transform=self.ax_bbu.transAxes)
            self.ax_bbu.text(0.6, 0.9 - i*0.1, val, color="white", fontsize=10, fontweight="bold", transform=self.ax_bbu.transAxes)

    def run(self):
        plt.ion()
        print("\n[*] Starting Commute Scenario. Watch the handover as the UE enters the Office!")
        while plt.fignum_exists(self.fig.number):
            if not self.paused:
                best_snr, cap, mcs = self.step_simulation()
                if self.step % 3 == 0: 
                    self.update_dashboard(best_snr, cap, mcs)
            plt.pause(0.01)
        plt.ioff()

if __name__ == "__main__":
    twin = DigitalTwin()
    twin.run()