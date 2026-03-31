"""
=============================================================================
  6G THz DIGITAL TWIN — MULTI-SCENARIO 8-PANEL DASHBOARD
  Frequency: 300 GHz | Bandwidth: 40 GHz | UDP Precision Tracking
  Scenarios: 1. Office  2. Urban Streets  3. Highway  4. Classroom
=============================================================================
"""

import os
import sys
import math
import time
import random
import collections
import numpy as np
import matplotlib

# Auto-detect display
_DISPLAY = os.environ.get("DISPLAY", "")
if _DISPLAY:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")
    print("[INFO] Headless environment detected. Plots will not render live.")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — SYSTEM CONFIGURATION (300 GHz THz Physics)
# ─────────────────────────────────────────────────────────────────────────────

class THz6GConfig:
    CARRIER_FREQ_HZ  = 300e9       # 300 GHz sub-THz carrier
    BANDWIDTH_HZ     = 40e9        # 40 GHz channel bandwidth
    NUMEROLOGY_MU    = 6           # μ=6 → SCS = 960 kHz
    GNB_ANT_GAIN_DBI = 35          # Massive MIMO beamforming gain
    UE_ANT_GAIN_DBI  = 20          # UE beamforming gain
    NOISE_FIGURE_DB  = 9           
    IMPLEMENTATION_LOSS_DB = 2     
    UDP_PACKET_BYTES = 9000        # Jumbo MTU for high-precision throughput

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
        if snr_db < -10: return 0
        snr_lin = 10**(snr_db/10)
        return THz6GConfig.BANDWIDTH_HZ * math.log2(1 + snr_lin) / 1e9

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — THz CHANNEL & PHYSICS
# ─────────────────────────────────────────────────────────────────────────────

class THz_Channel:
    WALL_REFLECTION_LOSS_DB = 12   
    ABSORPTION_COEFF_DB_PER_M = 0.0015  # Molecular absorption at 300GHz
    RAIN_ATTN_DB_PER_M = 0.01     

    @staticmethod
    def fspl_db(distance_m, freq_hz):
        distance_m = max(distance_m, 0.1)
        return 20*math.log10(distance_m) + 20*math.log10(freq_hz) - 147.55

    @staticmethod
    def compute_link(tx_power, tx_x, tx_y, rx_x, rx_y, rain=False, wall_y=None):
        d_los = max(math.hypot(rx_x - tx_x, rx_y - tx_y), 1.0)
        pl_los = THz_Channel.fspl_db(d_los, THz6GConfig.CARRIER_FREQ_HZ)
        abs_los = THz_Channel.ABSORPTION_COEFF_DB_PER_M * d_los
        rain_los = THz_Channel.RAIN_ATTN_DB_PER_M * d_los if rain else 0
        shadow = random.gauss(0, 2.0)  
        
        total_pl_los = pl_los + abs_los + rain_los + shadow
        rx_power_los = (tx_power + THz6GConfig.GNB_ANT_GAIN_DBI + 
                        THz6GConfig.UE_ANT_GAIN_DBI - total_pl_los - THz6GConfig.IMPLEMENTATION_LOSS_DB)

        # Process NLoS Wall Reflection if wall exists
        rx_combined_dbm = rx_power_los
        ref_point_x = rx_x
        d_ref = d_los

        if wall_y is not None:
            tx_img_y = 2 * wall_y - tx_y
            d_ref = max(math.hypot(rx_x - tx_x, (2*wall_y - tx_y) - tx_y) + math.hypot(rx_x - tx_x, rx_y - tx_y), 1.0)
            pl_ref = THz_Channel.fspl_db(d_ref, THz6GConfig.CARRIER_FREQ_HZ) + THz_Channel.WALL_REFLECTION_LOSS_DB
            abs_ref = THz_Channel.ABSORPTION_COEFF_DB_PER_M * d_ref
            total_pl_ref = pl_ref + abs_ref + rain_los
            
            rx_power_ref_dbm = (tx_power + THz6GConfig.GNB_ANT_GAIN_DBI + 
                                THz6GConfig.UE_ANT_GAIN_DBI - total_pl_ref - THz6GConfig.IMPLEMENTATION_LOSS_DB)

            p_rx_total_mw = (10**(rx_power_los/10) + 10**(rx_power_ref_dbm/10))
            rx_combined_dbm = 10*math.log10(p_rx_total_mw + 1e-30)

            if (2*wall_y - tx_y - tx_y) != 0:
                ref_point_x = tx_x + (rx_x - tx_x) * (wall_y - tx_y) / ((2*wall_y - tx_y) - tx_y)

        noise_floor = THz6GConfig.thermal_noise_dbm()
        snr_db = rx_combined_dbm - noise_floor

        return {
            "d_los_m": d_los, "pl_los_db": pl_los, "abs_loss_db": abs_los,
            "rain_loss_db": rain_los, "total_pl_db": total_pl_los,
            "rx_power_dbm": rx_combined_dbm, "noise_floor_dbm": noise_floor,
            "snr_db": snr_db, "ref_point_x": ref_point_x, "d_ref_m": d_ref
        }

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — NETWORK LOGIC & UEs
# ─────────────────────────────────────────────────────────────────────────────

class BBU:
    def __init__(self, ue_list):
        self.flows = []
        for ue in ue_list:
            self.flows.append({
                "id": ue['id'], "name": ue['name'], "target_gbps": ue['target_gbps'],
                "tx_packets": 0, "tx_bytes": 0, "queue_depth": 0, "color": ue['color']
            })
        self.ecpri_used_gbps = 0

    def tick(self, dt_s, capacity_list):
        actual_per_flow = []
        total_ecpri = 0
        for i, flow in enumerate(self.flows):
            cap = capacity_list[i]
            target = flow["target_gbps"]
            actual = max(0, min(target, cap))

            # High-precision byte tracking
            bytes_this_step = actual * 1e9 * dt_s / 8
            pkts = int(bytes_this_step / THz6GConfig.UDP_PACKET_BYTES)
            flow["tx_packets"] += pkts
            flow["tx_bytes"] += int(bytes_this_step)
            
            overload = max(0, target - cap)
            flow["queue_depth"] = min(10000, max(0, flow["queue_depth"] + int(overload * 1e6 * dt_s) - pkts // 2))

            total_ecpri += actual
            actual_per_flow.append(actual)

        self.ecpri_used_gbps = total_ecpri
        return actual_per_flow

class UserEquipment:
    def __init__(self, cfg):
        self.id = cfg['id']
        self.name = cfg['name']
        self.x, self.y = cfg['x'], cfg['y']
        self.vx, self.vy = cfg['vx'], cfg['vy']
        self.color = cfg['color']
        self.target_gbps = cfg['target_gbps']
        self.rx_packets = 0
        self.lost_packets = 0
        self.tput_history = collections.deque(maxlen=200)
        self.snr_history = collections.deque(maxlen=200)
        self.latency_history = collections.deque(maxlen=200)

    def move(self, dt_s, bounds):
        self.x += self.vx * dt_s
        self.y += self.vy * dt_s
        if self.x > bounds['x_max']: self.x = bounds['x_min']
        if self.x < bounds['x_min']: self.x = bounds['x_max']
        if self.y > bounds['y_max']: self.vy = -abs(self.vy)
        if self.y < bounds['y_min']: self.vy = abs(self.vy)

    def receive(self, actual_gbps, snr_db, dt_s, tx_packets):
        per = max(0, min(1, 0.1 * (1 - snr_db / 15)))
        rx_pkts = int(tx_packets * (1 - per))
        self.rx_packets += rx_pkts
        self.lost_packets += tx_packets - rx_pkts
        
        prop_delay_us = (math.hypot(self.x, self.y) / 3e8) * 1e6
        total_latency_us = prop_delay_us + 50 + random.gauss(0, 5)
        
        self.tput_history.append(actual_gbps)
        self.snr_history.append(snr_db)
        self.latency_history.append(max(0, total_latency_us))
        return per, total_latency_us

class FlowMonitor:
    def __init__(self, ue_list):
        self.ues = ue_list

    def get_per_flow_stats(self, flow_id):
        ue = self.ues[flow_id]
        total_tx = ue.rx_packets + ue.lost_packets
        return {
            "tx_packets": total_tx, "rx_packets": ue.rx_packets,
            "per_pct": (ue.lost_packets / total_tx * 100) if total_tx > 0 else 0,
            "rx_gbps": ue.tput_history[-1] if ue.tput_history else 0,
            "latency_us": ue.latency_history[-1] if ue.latency_history else 0,
        }

def generate_iq_cloud(snr_db, n_pts=200, qam_order=64):
    m = int(math.sqrt(qam_order))
    levels = np.linspace(-1, 1, m)
    pts = [(a, b) for a in levels for b in levels]
    noise_scale = 0.5 / max(1, 10**(snr_db/20))
    iq = []
    for _ in range(n_pts):
        base = pts[np.random.randint(len(pts))]
        iq.append([base[0] + np.random.normal(0, noise_scale), base[1] + np.random.normal(0, noise_scale)])
    return np.array(iq)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — SCENARIO MANAGER
# ─────────────────────────────────────────────────────────────────────────────

def get_scenario_config(scenario_name):
    scenarios = {
        "Office": {
            "title": "Indoor Office (Heavy THz Absorption)",
            "bounds": {"x_min": 0, "x_max": 40, "y_min": 0, "y_max": 25},
            "gnb": {"x": 20, "y": 12, "tx_power": 20}, # Low power indoor
            "wall_y": 20,
            "ues": [
                {"id": 0, "name": "Laptop (AR/VR)", "x": 5, "y": 5, "vx": 0, "vy": 0, "target_gbps": 50.0, "color": "#00f2ff"},
                {"id": 1, "name": "Smartphone", "x": 10, "y": 20, "vx": 1.0, "vy": 0.5, "target_gbps": 10.0, "color": "#ff6b6b"},
                {"id": 2, "name": "Smart Screen", "x": 35, "y": 15, "vx": 0, "vy": 0, "target_gbps": 80.0, "color": "#f9ca24"},
            ]
        },
        "Urban Streets": {
            "title": "Urban Smart-City (THz V2X + Pedestrians)",
            "bounds": {"x_min": -60, "x_max": 380, "y_min": -20, "y_max": 200},
            "gnb": {"x": 0, "y": 80, "tx_power": 46},
            "wall_y": 140,
            "ues": [
                {"id": 0, "name": "Car-UE (V2X)", "x": -40, "y": 20, "vx": 16.6, "vy": 0, "target_gbps": 100.0, "color": "#00f2ff"},
                {"id": 1, "name": "Pedestrian", "x": 80, "y": 35, "vx": 1.4, "vy": 0.5, "target_gbps": 20.0, "color": "#ff6b6b"},
                {"id": 2, "name": "IoT-Sensor", "x": 200, "y": 15, "vx": 0, "vy": 0, "target_gbps": 5.0, "color": "#f9ca24"},
            ]
        },
        "Highways": {
            "title": "High-Speed Highway (Long-Range THz Platoon)",
            "bounds": {"x_min": 0, "x_max": 1000, "y_min": 0, "y_max": 50},
            "gnb": {"x": 500, "y": 40, "tx_power": 55}, # Extreme EIRP for distance
            "wall_y": None, # No reflections on open highway
            "ues": [
                {"id": 0, "name": "Platoon Leader", "x": 50, "y": 15, "vx": 33.3, "vy": 0, "target_gbps": 120.0, "color": "#00f2ff"},
                {"id": 1, "name": "Platoon Follower", "x": 20, "y": 15, "vx": 33.3, "vy": 0, "target_gbps": 80.0, "color": "#ff6b6b"},
                {"id": 2, "name": "Opposite Traffic", "x": 950, "y": 35, "vx": -33.3, "vy": 0, "target_gbps": 50.0, "color": "#f9ca24"},
            ]
        },
        "Classrooms": {
            "title": "Dense Classroom (High Capacity Holograms)",
            "bounds": {"x_min": 0, "x_max": 15, "y_min": 0, "y_max": 15},
            "gnb": {"x": 7.5, "y": 7.5, "tx_power": 15},
            "wall_y": 15,
            "ues": [
                {"id": 0, "name": "Teacher Hologram", "x": 7.5, "y": 13, "vx": 0.2, "vy": 0, "target_gbps": 150.0, "color": "#00f2ff"},
                {"id": 1, "name": "Student Tablet A", "x": 3, "y": 4, "vx": 0, "vy": 0, "target_gbps": 10.0, "color": "#ff6b6b"},
                {"id": 2, "name": "Student Tablet B", "x": 12, "y": 8, "vx": 0, "vy": 0, "target_gbps": 10.0, "color": "#f9ca24"},
            ]
        }
    }
    return scenarios.get(scenario_name, scenarios["Urban Streets"])

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — DIGITAL TWIN ENGINE (8-Panel Dashboard)
# ─────────────────────────────────────────────────────────────────────────────

class DigitalTwin:
    DT = 0.02  # High precision 20ms steps

    def __init__(self, scenario_name):
        self.cfg = get_scenario_config(scenario_name)
        self.ues = [UserEquipment(u) for u in self.cfg["ues"]]
        self.bbu = BBU(self.cfg["ues"])
        self.monitor = FlowMonitor(self.ues)
        self.sim_time, self.step = 0.0, 0
        self.rain_mode, self.paused = False, False

        self._setup_dashboard()

    def _setup_dashboard(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(20, 11), facecolor="#060b14")
        self.fig.suptitle(f"6G THz DIGITAL TWIN | {self.cfg['title']} | 300 GHz", color="#00f2ff", fontsize=16, fontweight="bold")
        gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.4, left=0.05, right=0.97, top=0.92, bottom=0.06)

        # Init Panels
        self.ax_map = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_bbu = self.fig.add_subplot(gs[0, 2], facecolor="#0a0f1e")
        self.ax_spec = self.fig.add_subplot(gs[0, 3], facecolor="#0a0f1e")
        self.ax_tput = self.fig.add_subplot(gs[1, 2:4], facecolor="#0a0f1e")
        self.ax_snr = self.fig.add_subplot(gs[2, 0], facecolor="#0a0f1e")
        self.ax_udp = self.fig.add_subplot(gs[2, 1], facecolor="#0a0f1e")
        self.ax_iq = self.fig.add_subplot(gs[2, 2], facecolor="#0a0f1e")
        self.ax_lb = self.fig.add_subplot(gs[2, 3], facecolor="#0a0f1e")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._init_map_artists()

    def _init_map_artists(self):
        ax = self.ax_map
        b = self.cfg["bounds"]
        ax.set_facecolor("#0d1b2a")
        ax.set_xlim(b["x_min"], b["x_max"]); ax.set_ylim(b["y_min"], b["y_max"])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Dynamic Map ({b['x_max']}m x {b['y_max']}m)", color="white", fontsize=10, pad=5)

        # Draw Reflector Wall if exists
        if self.cfg["wall_y"]:
            ax.axhline(self.cfg["wall_y"], color="#a29bfe", lw=2, linestyle="-", alpha=0.6)

        # Draw gNB
        gx, gy = self.cfg["gnb"]["x"], self.cfg["gnb"]["y"]
        ax.plot(gx, gy, '^', ms=15, color="lime", zorder=10)
        ax.text(gx, gy + (b["y_max"]*0.05), f"gNB\n{self.cfg['gnb']['tx_power']}dBm", color="lime", ha="center", fontsize=8)

        self.ue_artists, self.ue_labels, self.beam_wedges, self.ray_lines, self.ref_lines = [], [], [], [], []
        self.tput_lines = []

        for ue in self.ues:
            m, = ax.plot([], [], "o", ms=12, color=ue.color, zorder=10, mec="white")
            l = ax.text(0, 0, ue.name, color=ue.color, fontsize=8, ha="center", va="bottom")
            w = patches.Wedge((gx, gy), b["x_max"]/2, 0, 1, color=ue.color, alpha=0.10)
            ax.add_patch(w)
            los, = ax.plot([], [], "-", color=ue.color, lw=1.5, alpha=0.75)
            ref, = ax.plot([], [], "--", color=ue.color, lw=0.8, alpha=0.45)
            
            self.ue_artists.append(m); self.ue_labels.append(l); self.beam_wedges.append(w)
            self.ray_lines.append(los); self.ref_lines.append(ref)
            
            tln, = self.ax_tput.plot([], [], "-", color=ue.color, lw=1.8, label=ue.name)
            self.tput_lines.append(tln)

        self.ax_tput.legend(fontsize=7, loc="upper left", facecolor="#0a0f1e")
        self.ax_tput.set_ylim(0, max([u.target_gbps for u in self.ues]) + 20)
        self.ax_tput.grid(True, color="#1a2a3a", linewidth=0.5)
        
        self.iq_scatter = self.ax_iq.scatter([], [], c=self.ues[0].color, s=6, alpha=0.7)

    def _on_key(self, event):
        if event.key == "r": self.rain_mode = not self.rain_mode
        elif event.key == " ": self.paused = not self.paused
        elif event.key == "q": plt.close("all")

    def step_simulation(self):
        self.sim_time += self.DT
        self.step += 1

        for ue in self.ues: ue.move(self.DT, self.cfg["bounds"])

        channel_results = []
        caps = []
        for ue in self.ues:
            cr = THz_Channel.compute_link(
                self.cfg["gnb"]["tx_power"], self.cfg["gnb"]["x"], self.cfg["gnb"]["y"], 
                ue.x, ue.y, rain=self.rain_mode, wall_y=self.cfg["wall_y"]
            )
            channel_results.append(cr)
            caps.append(THz6GConfig.shannon_capacity_gbps(cr["snr_db"]))

        actual_gbps = self.bbu.tick(self.DT, caps)

        for i, ue in enumerate(self.ues):
            tx_pkts = max(1, self.bbu.flows[i]["tx_packets"] // 100)
            ue.receive(actual_gbps[i], channel_results[i]["snr_db"], self.DT, tx_pkts)

        return channel_results

    def update_dashboard(self, crs):
        t = self.sim_time
        gx, gy = self.cfg["gnb"]["x"], self.cfg["gnb"]["y"]

        # MAP PANEL
        for i, ue in enumerate(self.ues):
            self.ue_artists[i].set_data([ue.x], [ue.y])
            self.ue_labels[i].set_position((ue.x, ue.y + (self.cfg["bounds"]["y_max"]*0.02)))
            
            angle = math.degrees(math.atan2(ue.y - gy, ue.x - gx))
            self.beam_wedges[i].set_theta1(angle - 4); self.beam_wedges[i].set_theta2(angle + 4)
            self.ray_lines[i].set_data([gx, ue.x], [gy, ue.y])
            
            if self.cfg["wall_y"]:
                self.ref_lines[i].set_data([gx, crs[i]["ref_point_x"], ue.x], [gy, self.cfg["wall_y"], ue.y])

        # BBU PANEL
        self.ax_bbu.cla(); self.ax_bbu.set_title("BBU — Flow Scheduler", color="#f9ca24", fontsize=9)
        queues = [f["queue_depth"] for f in self.bbu.flows]
        bars = self.ax_bbu.barh([f["name"] for f in self.bbu.flows], queues, color=[f["color"] for f in self.bbu.flows], height=0.5)
        for bar, ue in zip(bars, self.ues): 
            self.ax_bbu.text(bar.get_width() + 50, bar.get_y() + 0.25, f"{ue.tput_history[-1] if ue.tput_history else 0:.1f} Gbps", color="white", fontsize=7.5)
        self.ax_bbu.set_xlim(0, max(max(queues)+500, 5000)); self.ax_bbu.set_facecolor("#0a0f1e")

        # SPECTRUM PANEL
        self.ax_spec.cla(); self.ax_spec.set_title("THz Spectrum (40 GHz BW)", color="#a29bfe", fontsize=9)
        freqs = np.linspace(300e9 - 20e9, 300e9 + 20e9, 500) / 1e9
        psd_dbm = np.full_like(freqs, self.cfg["gnb"]["tx_power"] - 60)
        noise_floor_arr = np.full_like(freqs, THz6GConfig.thermal_noise_dbm())
        self.ax_spec.fill_between(freqs, noise_floor_arr, psd_dbm, color="#a29bfe", alpha=0.5)
        self.ax_spec.plot(freqs, psd_dbm, color="#a29bfe", lw=0.8)
        self.ax_spec.axhline(THz6GConfig.thermal_noise_dbm(), color="red", lw=0.7, linestyle="--")
        self.ax_spec.set_facecolor("#0a0f1e"); self.ax_spec.tick_params(colors="white", labelsize=6.5)

        # TPUT PANEL
        self.ax_tput.set_xlim(max(0, t - 8), t + 0.1)
        for i, ue in enumerate(self.ues):
            if len(ue.tput_history) > 1:
                times = np.linspace(max(0, t - len(ue.tput_history) * self.DT), t, len(ue.tput_history))
                self.tput_lines[i].set_data(times, list(ue.tput_history))

        # SNR PANEL
        self.ax_snr.cla(); self.ax_snr.set_title("Live SNR", color="#55efc4", fontsize=9)
        snrs = [c["snr_db"] for c in crs]
        bars = self.ax_snr.bar([u.name[:8] for u in self.ues], snrs, color=[u.color for u in self.ues], width=0.5)
        for b, v, m in zip(bars, snrs, [THz6GConfig.select_mcs(s)[0] for s in snrs]):
            self.ax_snr.text(b.get_x() + 0.25, max(0, v) + 0.5, f"{v:.1f}dB\n{m}", ha="center", color="white", fontsize=6.5)
        self.ax_snr.set_ylim(-10, 60); self.ax_snr.set_facecolor("#0a0f1e"); self.ax_snr.tick_params(colors="white", labelsize=7)

        # FLOW MONITOR PANEL
        self.ax_udp.cla(); self.ax_udp.set_title("UDP Flow Monitor", color="#fd79a8", fontsize=9); self.ax_udp.axis("off")
        rows = [[self.ues[i].name[:10], f"{self.monitor.get_per_flow_stats(i)['tx_packets']:,}", f"{self.monitor.get_per_flow_stats(i)['rx_packets']:,}", f"{self.monitor.get_per_flow_stats(i)['per_pct']:.2f}"] for i in range(len(self.ues))]
        tbl = self.ax_udp.table(cellText=rows, colLabels=["Flow", "Tx Pkts", "Rx Pkts", "PER%"], loc="center")
        tbl.set_fontsize(7.5)
        for c in tbl.get_celld().values(): c.set_text_props(color="white"); c.set_facecolor("#0a0f1e")
        self.ax_udp.set_facecolor("#0a0f1e")

        # IQ CONSTELLATION PANEL (Tracks UE 0)
        c0 = crs[0]
        qam = 4096 if c0["snr_db"] > 25 else 1024 if c0["snr_db"] > 20 else 256 if c0["snr_db"] > 16 else 64 if c0["snr_db"] > 12 else 4
        self.iq_scatter.set_offsets(generate_iq_cloud(c0["snr_db"], n_pts=300, qam_order=qam))
        self.ax_iq.set_title(f"IQ ({self.ues[0].name} | {qam}-QAM)", color="#74b9ff", fontsize=9)
        self.ax_iq.set_xlim(-1.5, 1.5); self.ax_iq.set_ylim(-1.5, 1.5); self.ax_iq.set_xticks([]); self.ax_iq.set_yticks([])

        # LINK BUDGET PANEL (Tracks UE 0)
        self.ax_lb.cla(); self.ax_lb.set_title(f"Link Budget ({self.ues[0].name})", color="#e17055", fontsize=9)
        labels = ["Tx", "gNB Gain", "UE Gain", "−FSPL", "−Mol. Abs", "−Rain", "−Impl.", "= Rx", "Noise", "= SNR"]
        vals = [self.cfg["gnb"]["tx_power"], THz6GConfig.GNB_ANT_GAIN_DBI, THz6GConfig.UE_ANT_GAIN_DBI, -c0["pl_los_db"], -c0["abs_loss_db"], -c0["rain_loss_db"] if self.rain_mode else -2.0, -THz6GConfig.IMPLEMENTATION_LOSS_DB, c0["rx_power_dbm"], c0["noise_floor_dbm"], c0["snr_db"]]
        colors = ["#e17055" if j in [7, 9] else "#d63031" if v < 0 else "#00b894" for j, v in enumerate(vals)]
        bars = self.ax_lb.barh(labels, vals, color=colors, height=0.6)
        for b, v in zip(bars, vals): self.ax_lb.text(v + (0.5 if v >= 0 else -0.5), b.get_y() + 0.3, f"{v:+.1f}", va="center", color="white", fontsize=6.5, ha="left" if v >= 0 else "right")
        self.ax_lb.set_facecolor("#0a0f1e"); self.ax_lb.tick_params(colors="white", labelsize=6.5)

    def run(self):
        plt.ion()
        print("\n" + "="*70)
        print(f"  6G THz DIGITAL TWIN STARTED — Scenario: {self.cfg['title']}")
        print("  CONTROLS:  [SPACE] Pause   [R] Toggle Rain   [Q] Quit")
        print("="*70 + "\n")

        while plt.fignum_exists(self.fig.number):
            if not self.paused:
                crs = self.step_simulation()
                # Update visual dashboard every 2 ticks for smooth precision mapping
                if self.step % 2 == 0: 
                    self.update_dashboard(crs)
            plt.pause(0.001)

        plt.ioff()
        print("\n[*] Simulation ended.")

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # CHOOSE SCENARIO: "Office", "Urban Streets", "Highways", or "Classrooms"
    ACTIVE_SCENARIO = "Urban Streets" 
    
    twin = DigitalTwin(ACTIVE_SCENARIO)
    twin.run()