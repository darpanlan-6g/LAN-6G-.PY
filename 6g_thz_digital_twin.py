"""
=============================================================================
  6G THz DIGITAL TWIN — BBU → RU → THz CHANNEL → UE
  High-Speed UDP Data Transfer | Real-Life Urban Smart City Scenario
  Based on 3GPP Release 19/20 | IMT-2030 6G Vision
=============================================================================

ARCHITECTURE:
  [BBU] ──eCPRI─► [Radio Unit / gNB @300GHz] ──THz Channel──► [UE]
    - UDP/IP packet engine          - 1024-elem Massive MIMO     - 64-elem phased array
    - Proportional-Fair scheduler   - OFDM (μ=6, 40GHz BW)      - OFDM demodulator
    - 100+ Gbps data generation     - Dynamic beamforming        - CQI feedback → gNB
    - Per-flow QoS tracking         - MCS adaptation             - UDP stats (Rx/PER)

SCENARIO:
  Smart City intersection — gNB mounted on rooftop (80m height).
  3 UEs: Car (60 km/h), Pedestrian (5 km/h), Fixed IoT Sensor.
  Environment: buildings, reflections (NLoS), rain attenuation.

DASHBOARD (8 panels):
  1. Urban Scene Map          5. SNR → MCS Heatmap
  2. BBU Packet Scheduler     6. UDP Flow Monitor (Tx/Rx/PER)
  3. THz Spectrum View        7. IQ Constellation
  4. Throughput Timeline      8. Link Budget Waterfall
=============================================================================
"""

import matplotlib
import os, sys

# Auto-detect display — use TkAgg when $DISPLAY is set, else Agg
_DISPLAY = os.environ.get("DISPLAY", "")
if _DISPLAY:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    # Headless / SSH environment — use non-interactive Agg backend
    matplotlib.use("Agg")
    print("[INFO] No display found — using Agg backend (non-interactive).")
    print("[INFO] Set $DISPLAY or run with: DISPLAY=:0 python3 6g_thz_digital_twin.py")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np
import math
import time
import collections
import random
import sys

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — SYSTEM CONFIGURATION (3GPP IMT-2030 / 6G Targets)
# ─────────────────────────────────────────────────────────────────────────────

class THz6GConfig:
    """Central configuration for all 6G THz physical layer parameters."""

    # ── Frequency & Bandwidth ──────────────────────────────────────────────
    CARRIER_FREQ_HZ  = 300e9       # 300 GHz sub-THz carrier
    BANDWIDTH_HZ     = 40e9        # 40 GHz channel bandwidth
    NUM_SUBCARRIERS  = 3276        # OFDM subcarriers (μ=6)

    # ── Numerology (IMT-2030 extended NR) ─────────────────────────────────
    NUMEROLOGY_MU    = 6           # μ=6 → SCS = 960 kHz
    SCS_KHZ          = 15 * (2**6) # 960 kHz subcarrier spacing
    SYMBOLS_PER_SLOT = 14          # Normal cyclic prefix

    # ── Antenna Configuration ──────────────────────────────────────────────
    GNB_TX_POWER_DBM = 46          # 40 W EIRP-capable gNB
    GNB_ANT_ELEMENTS = 1024        # 32×32 planar array
    GNB_ANT_GAIN_DBI = 35          # Massive MIMO beamforming gain

    UE_ANT_ELEMENTS  = 64          # 8×8 UE phased array
    UE_ANT_GAIN_DBI  = 20          # UE beamforming gain

    # ── Receiver ──────────────────────────────────────────────────────────
    NOISE_FIGURE_DB  = 9           # Receiver noise figure
    IMPLEMENTATION_LOSS_DB = 2     # Hardware imperfections

    # ── UDP Traffic ───────────────────────────────────────────────────────
    UDP_PACKET_BYTES = 9000        # Jumbo MTU frames
    UDP_FLOWS        = 3           # One per UE
    TARGET_RATE_GBPS = [100.0, 20.0, 5.0]  # Per-flow target (Car, Ped, IoT)

    # ── MCS Table (SNR threshold → modulation order, code rate, efficiency)
    # [SNR_min_dB, label, bits_per_symbol]
    MCS_TABLE = [
        (-5,  "QPSK  R1/5",  0.40),
        (0,   "QPSK  R1/3",  0.66),
        (4,   "QPSK  R1/2",  1.00),
        (8,   "16QAM R1/2",  2.00),
        (12,  "64QAM R2/3",  4.00),
        (16,  "256QAM R3/4", 6.00),
        (20,  "1024QAM R4/5",8.00),
        (25,  "4096QAM R9/10",10.0),
    ]

    @staticmethod
    def thermal_noise_dbm():
        """Thermal noise floor = -174 + 10log10(BW) + NF  [dBm]"""
        return -174 + 10*math.log10(THz6GConfig.BANDWIDTH_HZ) + THz6GConfig.NOISE_FIGURE_DB

    @staticmethod
    def select_mcs(snr_db):
        """Returns (label, spectral_efficiency_bps_per_hz) for given SNR."""
        selected = THz6GConfig.MCS_TABLE[0]
        for row in THz6GConfig.MCS_TABLE:
            if snr_db >= row[0]:
                selected = row
        return selected[1], selected[2]

    @staticmethod
    def shannon_capacity_gbps(snr_db, bw_hz=None):
        bw = bw_hz or THz6GConfig.BANDWIDTH_HZ
        snr_lin = 10**(snr_db/10)
        return bw * math.log2(1 + snr_lin) / 1e9

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — THz CHANNEL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class THz_Channel:
    """
    Realistic THz channel model combining:
      - Free-Space Path Loss (FSPL)
      - Molecular Absorption (water vapor + O₂ at 300 GHz)
      - Rain & Atmospheric attenuation
      - Large-scale shadowing (log-normal)
      - Multi-path: LoS + 2 specular reflections (walls)
    """
    WALL_REFLECTION_LOSS_DB = 12   # Single-bounce wall reflection loss @ 300 GHz
    ABSORPTION_COEFF_DB_PER_M = 0.0015  # ≈ 1.5 dB/km at 300 GHz (moderate humidity)
    RAIN_ATTN_DB_PER_M = 0.01     # ≈ 10 dB/km in heavy rain

    @staticmethod
    def fspl_db(distance_m, freq_hz):
        """Free-Space Path Loss in dB."""
        if distance_m <= 0:
            distance_m = 0.1
        return 20*math.log10(distance_m) + 20*math.log10(freq_hz) - 147.55

    @staticmethod
    def molecular_absorption_db(distance_m):
        """Molecular absorption loss over distance."""
        return THz_Channel.ABSORPTION_COEFF_DB_PER_M * distance_m

    @staticmethod
    def compute_link(tx_x, tx_y, rx_x, rx_y, rain=False, wall_y=140):
        """
        Returns dict with all path components and total received power.
        Geometry: 2D (x, y)
        """
        # --- LoS distance ---
        d_los = math.hypot(rx_x - tx_x, rx_y - tx_y)
        d_los = max(d_los, 1.0)

        # --- LoS path loss ---
        pl_los = THz_Channel.fspl_db(d_los, THz6GConfig.CARRIER_FREQ_HZ)
        abs_los = THz_Channel.molecular_absorption_db(d_los)
        rain_los = THz_Channel.RAIN_ATTN_DB_PER_M * d_los if rain else 0
        shadow = random.gauss(0, 2.0)  # 2 dB std log-normal shadowing
        total_pl_los = pl_los + abs_los + rain_los + shadow

        # --- Received power (LoS) ---
        rx_power_los = (THz6GConfig.GNB_TX_POWER_DBM
                        + THz6GConfig.GNB_ANT_GAIN_DBI
                        + THz6GConfig.UE_ANT_GAIN_DBI
                        - total_pl_los
                        - THz6GConfig.IMPLEMENTATION_LOSS_DB)

        # --- NLoS via wall reflection (image source method) ---
        # Image of TX reflected in wall at wall_y
        tx_img_y = 2 * wall_y - tx_y
        d_ref = math.hypot(rx_x - tx_x, (2*wall_y - tx_y) - tx_y) + math.hypot(rx_x - tx_x, rx_y - tx_y)
        d_ref = max(d_ref, 1.0)
        pl_ref = THz_Channel.fspl_db(d_ref, THz6GConfig.CARRIER_FREQ_HZ) + THz_Channel.WALL_REFLECTION_LOSS_DB
        abs_ref = THz_Channel.molecular_absorption_db(d_ref)
        total_pl_ref = pl_ref + abs_ref + (THz_Channel.RAIN_ATTN_DB_PER_M * d_ref if rain else 0)
        rx_power_ref_dbm = (THz6GConfig.GNB_TX_POWER_DBM
                            + THz6GConfig.GNB_ANT_GAIN_DBI
                            + THz6GConfig.UE_ANT_GAIN_DBI
                            - total_pl_ref
                            - THz6GConfig.IMPLEMENTATION_LOSS_DB)

        # --- Combine LoS + NLoS (power domain) ---
        noise_floor = THz6GConfig.thermal_noise_dbm()
        p_rx_total_mw = (10**(rx_power_los/10) + 10**(rx_power_ref_dbm/10))
        rx_combined_dbm = 10*math.log10(p_rx_total_mw + 1e-30)

        # Reflection point x
        if (2*wall_y - tx_y - tx_y) != 0:
            ref_point_x = tx_x + (rx_x - tx_x) * (wall_y - tx_y) / ((2*wall_y - tx_y) - tx_y)
        else:
            ref_point_x = (tx_x + rx_x) / 2

        snr_db = rx_combined_dbm - noise_floor

        return {
            "d_los_m"        : d_los,
            "pl_los_db"      : pl_los,
            "abs_loss_db"    : abs_los,
            "rain_loss_db"   : rain_los,
            "total_pl_db"    : total_pl_los,
            "rx_power_dbm"   : rx_combined_dbm,
            "noise_floor_dbm": noise_floor,
            "snr_db"         : snr_db,
            "ref_point_x"    : ref_point_x,
            "d_ref_m"        : d_ref,
        }

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — BBU (BASEBAND UNIT)
# ─────────────────────────────────────────────────────────────────────────────

class BBU:
    """
    Baseband Unit — models the core network / DU side.
    Responsibilities:
      - Generates UDP/IP packets at configured Gbps rate per flow
      - Proportional-Fair scheduler assigns PRBs to flows
      - Tracks eCPRI fronthaul utilization
      - Exposes per-flow stats for FlowMonitor
    """
    def __init__(self):
        self.flows = []
        for i in range(THz6GConfig.UDP_FLOWS):
            self.flows.append({
                "id"           : i,
                "name"         : ["Car-UE", "Pedestrian-UE", "IoT-Sensor"][i],
                "target_gbps"  : THz6GConfig.TARGET_RATE_GBPS[i],
                "tx_packets"   : 0,
                "tx_bytes"     : 0,
                "queue_depth"  : 0,      # Simulated buffer depth (packets)
                "color"        : ["#00f2ff", "#ff6b6b", "#f9ca24"][i],
            })
        self.ecpri_used_gbps = 0
        self.total_tx_gbps_history = collections.deque(maxlen=200)

    def tick(self, dt_s, channel_capacity_per_flow):
        """
        Called every simulation step.
        dt_s: time delta in seconds
        channel_capacity_per_flow: list of Gbps available per flow
        Returns list of actual_gbps delivered per flow.
        """
        actual_per_flow = []
        total_ecpri = 0
        for i, flow in enumerate(self.flows):
            cap = channel_capacity_per_flow[i]
            target = flow["target_gbps"]
            actual = min(target, cap)  # Can only send as fast as channel allows
            if cap <= 0:
                actual = 0

            # Generate packets
            bytes_this_step = actual * 1e9 * dt_s / 8
            pkts = int(bytes_this_step / THz6GConfig.UDP_PACKET_BYTES)
            flow["tx_packets"] += pkts
            flow["tx_bytes"]   += int(bytes_this_step)

            # Queue dynamics: accumulate if channel can't drain fast enough
            overload = max(0, target - cap)
            flow["queue_depth"] = min(10000, flow["queue_depth"] + int(overload * 1e6 * dt_s))
            flow["queue_depth"] = max(0,  flow["queue_depth"] - pkts // 2)

            total_ecpri += actual
            actual_per_flow.append(actual)

        self.ecpri_used_gbps = total_ecpri
        return actual_per_flow

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — RADIO UNIT
# ─────────────────────────────────────────────────────────────────────────────

class RadioUnit:
    """
    Radio Unit — THz air interface layer.
    Models: OFDM numerology, beamforming, MCS selection, SINR→capacity mapping.
    """
    def __init__(self):
        self.current_mcs_labels = ["QPSK R1/2"] * THz6GConfig.UDP_FLOWS
        self.current_se = [1.0] * THz6GConfig.UDP_FLOWS  # spectral efficiency bps/Hz
        self.beam_angles_deg = [0.0] * THz6GConfig.UDP_FLOWS

    def process(self, channel_results):
        """
        channel_results: list of dicts from THz_Channel.compute_link for each UE
        Returns list of capacity_gbps per UE.
        """
        capacities = []
        for i, cr in enumerate(channel_results):
            snr = cr["snr_db"]
            mcs_label, se = THz6GConfig.select_mcs(snr)
            self.current_mcs_labels[i] = mcs_label
            self.current_se[i] = se
            # Capacity = BW × spectral_efficiency (capped by Shannon)
            shannon_cap = THz6GConfig.shannon_capacity_gbps(snr)
            mcs_cap = THz6GConfig.BANDWIDTH_HZ * se / 1e9
            capacities.append(min(shannon_cap, mcs_cap))
        return capacities

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — USER EQUIPMENT
# ─────────────────────────────────────────────────────────────────────────────

class UserEquipment:
    """Mobile UE — models receiver chain and reports CQI/ACK back."""
    def __init__(self, ue_id, name, x, y, vx, vy, color):
        self.id   = ue_id
        self.name = name
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy       # velocity m/s
        self.color = color

        self.rx_packets   = 0
        self.rx_bytes     = 0
        self.lost_packets = 0
        self.tput_history = collections.deque(maxlen=200)
        self.snr_history  = collections.deque(maxlen=200)
        self.latency_history = collections.deque(maxlen=200)

    def move(self, dt_s, x_min=-50, x_max=400):
        self.x += self.vx * dt_s
        self.y += self.vy * dt_s
        # Bounce within map
        if self.x > x_max: self.x = x_min
        if self.x < x_min: self.x = x_max
        if self.y > 90:  self.vy = -abs(self.vy)
        if self.y < 10:  self.vy =  abs(self.vy)

    def receive(self, actual_gbps, snr_db, dt_s, tx_packets):
        """Simulate packet reception and loss."""
        # Packet Error Rate model: PER ≈ 0 for SNR > threshold, else rises
        per = max(0, min(1, 0.1 * (1 - snr_db / 15)))
        rx_pkts = int(tx_packets * (1 - per))
        lost    = tx_packets - rx_pkts

        bytes_rx = rx_pkts * THz6GConfig.UDP_PACKET_BYTES
        self.rx_packets   += rx_pkts
        self.rx_bytes     += bytes_rx
        self.lost_packets += lost

        # Latency: propagation + processing (μs range at THz)
        prop_delay_us = (math.hypot(self.x, self.y) / 3e8) * 1e6  # speed of light
        proc_delay_us = 50  # baseband processing budget
        total_latency_us = prop_delay_us + proc_delay_us + random.gauss(0, 5)

        self.tput_history.append(actual_gbps)
        self.snr_history.append(snr_db)
        self.latency_history.append(max(0, total_latency_us))

        return per, total_latency_us

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — FLOW MONITOR (Network Analytics)
# ─────────────────────────────────────────────────────────────────────────────

class FlowMonitor:
    """Track and summarise per-flow network KPIs."""
    def __init__(self, ue_list):
        self.ues = ue_list
        self.time_history = collections.deque(maxlen=200)

    def snapshot(self, t, actual_gbps_list, per_list, latency_list):
        self.time_history.append(t)

    def get_per_flow_stats(self, flow_id):
        ue = self.ues[flow_id]
        total_tx = ue.rx_packets + ue.lost_packets
        per = (ue.lost_packets / total_tx * 100) if total_tx > 0 else 0
        return {
            "tx_packets": total_tx,
            "rx_packets": ue.rx_packets,
            "per_pct"   : per,
            "rx_gbps"   : list(ue.tput_history)[-1] if ue.tput_history else 0,
            "latency_us": list(ue.latency_history)[-1] if ue.latency_history else 0,
        }

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — IQ CONSTELLATION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_iq_cloud(snr_db, n_pts=200, qam_order=64):
    """
    Generate noisy IQ constellation points for a given QAM order.
    Noise spread is inversely proportional to SNR.
    """
    m = int(math.sqrt(qam_order))
    levels = np.linspace(-1, 1, m)
    pts = [(a, b) for a in levels for b in levels]

    noise_scale = 0.5 / max(1, 10**(snr_db/20))
    iq = []
    for _ in range(n_pts):
        base = pts[np.random.randint(len(pts))]
        iq.append([base[0] + np.random.normal(0, noise_scale),
                   base[1] + np.random.normal(0, noise_scale)])
    return np.array(iq)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 — DIGITAL TWIN ENGINE (Main Loop + Dashboard)
# ─────────────────────────────────────────────────────────────────────────────

class DigitalTwin:
    """Orchestrates the 6G simulation and renders the live 8-panel dashboard."""

    GNB_X, GNB_Y = 0, 80       # gNB position (rooftop)
    WALL_Y        = 140         # Building wall for reflections
    DT            = 0.02        # Simulation time step (20ms)

    def __init__(self):
        # ── UEs ──────────────────────────────────────────────────────────
        # (id, name, x, y, vx [m/s], vy, color)
        UE_SPEED_CAR = 60/3.6      # 60 km/h → m/s
        UE_SPEED_PED = 5/3.6       # 5 km/h pedestrian
        self.ues = [
            UserEquipment(0, "Car-UE",       -40, 20,  UE_SPEED_CAR, 0,    "#00f2ff"),
            UserEquipment(1, "Pedestrian-UE",  80, 35,  UE_SPEED_PED, 0.5, "#ff6b6b"),
            UserEquipment(2, "IoT-Sensor",    200, 15,  0,            0,    "#f9ca24"),
        ]

        self.bbu     = BBU()
        self.ru      = RadioUnit()
        self.channel = THz_Channel()
        self.monitor = FlowMonitor(self.ues)

        self.sim_time  = 0.0
        self.step      = 0
        self.rain_mode = False   # toggle with 'r' key
        self.paused    = False

        self._setup_dashboard()

    # ── Dashboard Layout ─────────────────────────────────────────────────────

    def _setup_dashboard(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(20, 11), facecolor="#060b14")
        self.fig.suptitle(
            "6G THz DIGITAL TWIN  |  BBU → 300 GHz RU → UE  |  Urban Smart-City",
            color="#00f2ff", fontsize=15, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=4, foreground="#000000")]
        )

        gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.4,
                               left=0.05, right=0.97, top=0.92, bottom=0.06)

        # Panel 1 — Urban Scene Map (top-left, spans 2 cols × 2 rows)
        self.ax_map = self.fig.add_subplot(gs[0:2, 0:2])
        self._init_map()

        # Panel 2 — BBU Packet Scheduler (top-right, row 0)
        self.ax_bbu = self.fig.add_subplot(gs[0, 2])
        self.ax_bbu.set_facecolor("#0a0f1e")
        self.ax_bbu.set_title("BBU — UDP Packet Scheduler", color="#f9ca24", fontsize=9, pad=4)

        # Panel 3 — THz Spectrum View (top-right, row 0, col 3)
        self.ax_spec = self.fig.add_subplot(gs[0, 3])
        self.ax_spec.set_facecolor("#0a0f1e")
        self.ax_spec.set_title("THz Spectrum (40 GHz BW)", color="#a29bfe", fontsize=9, pad=4)

        # Panel 4 — Throughput Timeline (middle-right, row 1, cols 2-3)
        self.ax_tput = self.fig.add_subplot(gs[1, 2:4])
        self.ax_tput.set_facecolor("#0a0f1e")
        self.ax_tput.set_title("Throughput Timeline (Gbps)", color="#00f2ff", fontsize=9, pad=4)
        self.ax_tput.set_ylabel("Gbps", color="white", fontsize=8)
        self.ax_tput.grid(True, color="#1a2a3a", linewidth=0.5)

        # Panel 5 — SNR / MCS Heatmap (bottom row, col 0)
        self.ax_snr = self.fig.add_subplot(gs[2, 0])
        self.ax_snr.set_facecolor("#0a0f1e")
        self.ax_snr.set_title("Live SNR & MCS", color="#55efc4", fontsize=9, pad=4)

        # Panel 6 — UDP Flow Monitor (bottom row, col 1)
        self.ax_udp = self.fig.add_subplot(gs[2, 1])
        self.ax_udp.set_facecolor("#0a0f1e")
        self.ax_udp.set_title("UDP Flow Monitor", color="#fd79a8", fontsize=9, pad=4)

        # Panel 7 — IQ Constellation (bottom row, col 2)
        self.ax_iq = self.fig.add_subplot(gs[2, 2])
        self.ax_iq.set_facecolor("#0a0f1e")
        self.ax_iq.set_title("IQ Constellation (Car-UE)", color="#74b9ff", fontsize=9, pad=4)
        self.ax_iq.set_xlim(-1.5, 1.5); self.ax_iq.set_ylim(-1.5, 1.5)
        self.ax_iq.set_xticks([]); self.ax_iq.set_yticks([])

        # Panel 8 — Link Budget Waterfall (bottom row, col 3)
        self.ax_lb = self.fig.add_subplot(gs[2, 3])
        self.ax_lb.set_facecolor("#0a0f1e")
        self.ax_lb.set_title("Link Budget (Car-UE)", color="#e17055", fontsize=9, pad=4)

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # ── Animated objects ─────────────────────────────────────────────
        self.ue_artists   = []
        self.ue_labels    = []
        self.beam_wedges  = []
        self.ray_los_lines= []
        self.ray_ref_lines= []

        for ue in self.ues:
            marker, = self.ax_map.plot([], [], "o", ms=12, color=ue.color,
                                       zorder=10, mec="white", mew=1.5)
            label = self.ax_map.text(0, 0, ue.name, color=ue.color,
                                     fontsize=8, fontweight="bold", zorder=11,
                                     ha="center", va="bottom")
            wedge = patches.Wedge(
                (self.GNB_X, self.GNB_Y), 350, 0, 1,
                color=ue.color, alpha=0.10, zorder=2
            )
            self.ax_map.add_patch(wedge)
            los_line, = self.ax_map.plot([], [], "-", color=ue.color,
                                          lw=1.5, alpha=0.75, zorder=3)
            ref_line, = self.ax_map.plot([], [], "--", color=ue.color,
                                          lw=0.8, alpha=0.45, zorder=3)
            self.ue_artists.append(marker)
            self.ue_labels.append(label)
            self.beam_wedges.append(wedge)
            self.ray_los_lines.append(los_line)
            self.ray_ref_lines.append(ref_line)

        # Throughput lines per flow
        self.tput_lines = []
        for ue in self.ues:
            ln, = self.ax_tput.plot([], [], "-", color=ue.color, lw=1.8, label=ue.name)
            self.tput_lines.append(ln)
        self.ax_tput.legend(fontsize=7, loc="upper left",
                            facecolor="#0a0f1e", edgecolor="none")
        self.ax_tput.set_ylim(0, 130)

        # IQ scatter
        self.iq_scatter = self.ax_iq.scatter([], [], c="#74b9ff", s=6, alpha=0.7)

    def _init_map(self):
        ax = self.ax_map
        ax.set_facecolor("#0d1b2a")
        ax.set_xlim(-60, 380); ax.set_ylim(-20, 200)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Urban Smart-City Scene  |  300 GHz THz Rays",
                     color="white", fontsize=10, pad=5)

        # Road
        ax.fill_between([-60, 380], -5, 5, color="#2d3436", zorder=0)
        ax.fill_between([-60, 380], 14, 26, color="#2d3436", zorder=0)
        ax.axhline(0, color="#f9ca24", lw=0.5, linestyle="--", alpha=0.5, zorder=1)
        ax.axhline(20, color="#f9ca24", lw=0.5, linestyle="--", alpha=0.5, zorder=1)
        ax.text(-55, 10, "ROAD", color="#636e72", fontsize=7, rotation=90)

        # Buildings (urban canyon)
        buildings = [(10, 130, 60, 50), (100, 135, 70, 45), (200, 128, 80, 52),
                     (310, 132, 60, 48)]
        for bx, by, bw, bh in buildings:
            ax.add_patch(patches.Rectangle((bx, by), bw, bh,
                                           fc="#1e3799", ec="#4a69bd",
                                           lw=0.8, zorder=1))
            # Windows
            for wr in range(3):
                for wc in range(4):
                    ax.add_patch(patches.Rectangle(
                        (bx + 6 + wc*14, by + 8 + wr*14), 8, 8,
                        fc="#f9ca24", alpha=0.6, zorder=2))

        # Wall (reflector)
        ax.axhline(self.WALL_Y, color="#a29bfe", lw=2, linestyle="-",
                   alpha=0.6, label="Reflector Wall")
        ax.text(5, 142, "Building Wall (THz Reflector)", color="#a29bfe", fontsize=7)

        # gNB Tower
        self._draw_gnb(ax, self.GNB_X, self.GNB_Y)

        # Legend
        ax.legend(loc="upper right", fontsize=7, facecolor="#0d1b2a", edgecolor="#636e72")

    def _draw_gnb(self, ax, x, y):
        """Draw detailed gNB tower with 1024-element MIMO panel."""
        # Pole
        ax.add_patch(patches.Rectangle((x-1.5, y), 3, 25, fc="#636e72", zorder=5))
        # Base
        ax.add_patch(patches.Rectangle((x-10, y-8), 20, 8, fc="#2d3436", zorder=5))
        # MIMO Panel
        panel = patches.FancyBboxPatch((x-7, y+18), 14, 9,
                                       boxstyle="round,pad=0.5",
                                       fc="#dfe6e9", ec="#b2bec3", zorder=6)
        ax.add_patch(panel)
        # Antenna dots (32×32 visualised as 6×6)
        for r in range(4):
            for c in range(5):
                ax.add_patch(patches.Circle(
                    (x-5.5 + c*2.5, y+20 + r*2), 0.5, fc="#d63031", zorder=7))
        ax.text(x, y+30, f"gNB\n300 GHz\n1024-MIMO",
                color="lime", ha="center", fontsize=7,
                fontweight="bold", zorder=8)

    def _on_key(self, event):
        if event.key == "r":
            self.rain_mode = not self.rain_mode
        elif event.key == " ":
            self.paused = not self.paused
        elif event.key == "q":
            plt.close("all")

    # ── Simulation Step ───────────────────────────────────────────────────────

    def step_simulation(self):
        """Run one simulation tick."""
        self.sim_time += self.DT
        self.step     += 1

        # 1. Move UEs
        for ue in self.ues:
            ue.move(self.DT)

        # 2. Channel computation per UE
        channel_results = []
        for ue in self.ues:
            cr = THz_Channel.compute_link(
                self.GNB_X, self.GNB_Y, ue.x, ue.y,
                rain=self.rain_mode, wall_y=self.WALL_Y
            )
            channel_results.append(cr)

        # 3. RU: MCS & capacity
        capacities_gbps = self.ru.process(channel_results)

        # 4. BBU: schedule flows
        actual_gbps = self.bbu.tick(self.DT, capacities_gbps)

        # 5. UE: receive + generate stats
        per_list     = []
        latency_list = []
        for i, ue in enumerate(self.ues):
            bbu_flow  = self.bbu.flows[i]
            tx_pkts   = bbu_flow["tx_packets"]
            per, lat  = ue.receive(actual_gbps[i], channel_results[i]["snr_db"],
                                   self.DT, max(1, bbu_flow["tx_packets"] // 100))
            per_list.append(per)
            latency_list.append(lat)

        # 6. Flow Monitor snapshot
        self.monitor.snapshot(self.sim_time, actual_gbps, per_list, latency_list)

        return channel_results, actual_gbps, per_list

    # ── Dashboard Update ──────────────────────────────────────────────────────

    def update_dashboard(self, channel_results, actual_gbps, per_list):
        t = self.sim_time

        # =========================================================
        # PANEL 1 — URBAN MAP (beam wedges + rays)
        # =========================================================
        for i, ue in enumerate(self.ues):
            self.ue_artists[i].set_data([ue.x], [ue.y])
            self.ue_labels[i].set_position((ue.x, ue.y + 5))

            # Beam wedge from gNB toward UE
            angle_deg = math.degrees(
                math.atan2(ue.y - self.GNB_Y, ue.x - self.GNB_X))
            w = self.beam_wedges[i]
            w.set_center((self.GNB_X, self.GNB_Y))
            w.set_theta1(angle_deg - 4)
            w.set_theta2(angle_deg + 4)

            # LoS ray
            self.ray_los_lines[i].set_data(
                [self.GNB_X, ue.x], [self.GNB_Y, ue.y])

            # Reflected ray
            rpx = channel_results[i]["ref_point_x"]
            self.ray_ref_lines[i].set_data(
                [self.GNB_X, rpx, ue.x],
                [self.GNB_Y, self.WALL_Y, ue.y]
            )

        # Rain indicator
        if hasattr(self, "_rain_txt"):
            self._rain_txt.remove()
        rain_str = "🌧 RAIN: ON  (press R)" if self.rain_mode else "press R: toggle rain"
        self._rain_txt = self.ax_map.text(
            200, -15, rain_str,
            color="#74b9ff" if self.rain_mode else "#636e72",
            fontsize=8, ha="center")

        # Time / step counter
        if hasattr(self, "_time_txt"):
            self._time_txt.remove()
        self._time_txt = self.ax_map.text(
            -55, 190, f"T={t:.2f}s  Step={self.step}",
            color="white", fontsize=8)

        # =========================================================
        # PANEL 2 — BBU SCHEDULER (bar chart of queue depths)
        # =========================================================
        self.ax_bbu.cla()
        self.ax_bbu.set_facecolor("#0a0f1e")
        self.ax_bbu.set_title("BBU — UDP Scheduler", color="#f9ca24", fontsize=9, pad=4)
        flows  = self.bbu.flows
        names  = [f["name"] for f in flows]
        queues = [f["queue_depth"] for f in flows]
        colors = [f["color"] for f in flows]
        bars = self.ax_bbu.barh(names, queues, color=colors, height=0.5)
        for bar, ue in zip(bars, self.ues):
            gbps = ue.tput_history[-1] if ue.tput_history else 0
            self.ax_bbu.text(
                bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f"{gbps:.1f} Gbps",
                va="center", color="white", fontsize=7.5)
        self.ax_bbu.set_xlabel("Queue Depth (pkts)", color="#636e72", fontsize=7)
        self.ax_bbu.tick_params(colors="white", labelsize=7.5)
        self.ax_bbu.set_xlim(0, max(max(queues)+500, 5000))
        self.ax_bbu.grid(True, axis="x", color="#1a2a3a", linewidth=0.5)
        # eCPRI header
        self.ax_bbu.text(
            0.5, 1.02,
            f"eCPRI Fronthaul: {self.bbu.ecpri_used_gbps:.1f} Gbps",
            transform=self.ax_bbu.transAxes,
            ha="center", color="#dfe6e9", fontsize=7.5
        )

        # =========================================================
        # PANEL 3 — THz Spectrum View
        # =========================================================
        self.ax_spec.cla()
        self.ax_spec.set_facecolor("#0a0f1e")
        self.ax_spec.set_title("THz Spectrum (40 GHz BW)", color="#a29bfe", fontsize=9, pad=4)
        bw  = THz6GConfig.BANDWIDTH_HZ
        fc  = THz6GConfig.CARRIER_FREQ_HZ
        freqs = np.linspace(fc - bw/2, fc + bw/2, 500) / 1e9  # GHz
        # Flat PSD with notches (molecular absorption)
        psd_dbm = np.full_like(freqs, THz6GConfig.GNB_TX_POWER_DBM - 60)
        # Absorption notches around 325 GHz
        notch_mask = (freqs > 316) & (freqs < 326)
        psd_dbm[notch_mask] -= 12 + 3 * np.random.rand(notch_mask.sum())
        noise_floor_arr = np.full_like(freqs, THz6GConfig.thermal_noise_dbm())
        self.ax_spec.fill_between(freqs, noise_floor_arr, psd_dbm,
                                   color="#a29bfe", alpha=0.5)
        self.ax_spec.plot(freqs, psd_dbm, color="#a29bfe", lw=0.8)
        self.ax_spec.axhline(THz6GConfig.thermal_noise_dbm(),
                              color="red", lw=0.7, linestyle="--", label="Noise floor")
        self.ax_spec.set_xlabel("Freq (GHz)", color="#636e72", fontsize=7)
        self.ax_spec.set_ylabel("PSD (dBm/Hz)", color="#636e72", fontsize=7)
        self.ax_spec.tick_params(colors="white", labelsize=6.5)
        self.ax_spec.legend(fontsize=6, facecolor="#0a0f1e", edgecolor="none")

        # =========================================================
        # PANEL 4 — Throughput Timeline
        # =========================================================
        t_min = max(0, t - 8)
        self.ax_tput.set_xlim(t_min, t + 0.1)
        for i, ue in enumerate(self.ues):
            if len(ue.tput_history) > 1:
                times = np.linspace(
                    max(0, t - len(ue.tput_history) * self.DT),
                    t, len(ue.tput_history))
                self.tput_lines[i].set_data(times, list(ue.tput_history))
        self.ax_tput.tick_params(colors="white", labelsize=7)
        # Shade rain period
        if self.rain_mode:
            self.ax_tput.axvspan(t_min, t, alpha=0.1, color="#74b9ff")

        # =========================================================
        # PANEL 5 — SNR & MCS per UE (live bar)
        # =========================================================
        self.ax_snr.cla()
        self.ax_snr.set_facecolor("#0a0f1e")
        self.ax_snr.set_title("Live SNR & MCS", color="#55efc4", fontsize=9, pad=4)
        snr_vals = [cr["snr_db"] for cr in channel_results]
        snr_bars = self.ax_snr.bar(
            [ue.name for ue in self.ues], snr_vals,
            color=[ue.color for ue in self.ues], width=0.5)
        for bar, val, mcs in zip(snr_bars, snr_vals, self.ru.current_mcs_labels):
            self.ax_snr.text(
                bar.get_x() + bar.get_width()/2,
                max(0, val) + 0.5,
                f"{val:.1f}dB\n{mcs}",
                ha="center", color="white", fontsize=6.5)
        self.ax_snr.set_ylabel("SNR (dB)", color="#636e72", fontsize=7)
        self.ax_snr.tick_params(colors="white", labelsize=7)
        self.ax_snr.axhline(0, color="red", lw=0.5, linestyle="--")
        self.ax_snr.set_ylim(-10, 60)
        self.ax_snr.grid(True, axis="y", color="#1a2a3a", linewidth=0.5)

        # =========================================================
        # PANEL 6 — UDP Flow Monitor Table
        # =========================================================
        self.ax_udp.cla()
        self.ax_udp.set_facecolor("#0a0f1e")
        self.ax_udp.set_title("UDP Flow Monitor", color="#fd79a8", fontsize=9, pad=4)
        self.ax_udp.axis("off")
        col_labels = ["Flow", "Tx Pkts", "Rx Pkts", "PER%", "Lat μs"]
        rows = []
        for i, ue in enumerate(self.ues):
            stats = self.monitor.get_per_flow_stats(i)
            rows.append([
                ue.name[:10],
                f"{stats['tx_packets']:,}",
                f"{stats['rx_packets']:,}",
                f"{stats['per_pct']:.2f}",
                f"{stats['latency_us']:.1f}",
            ])
        tbl = self.ax_udp.table(
            cellText=rows, colLabels=col_labels,
            loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("#0a0f1e" if r > 0 else "#1a1a2e")
            cell.set_edgecolor("#2d3436")
            cell.set_text_props(color="white")

        # =========================================================
        # PANEL 7 — IQ Constellation (Car-UE)
        # =========================================================
        snr_car = channel_results[0]["snr_db"]
        # Choose QAM order based on SNR
        qam = 4
        if snr_car > 25: qam = 4096
        elif snr_car > 20: qam = 1024
        elif snr_car > 16: qam = 256
        elif snr_car > 12: qam = 64
        elif snr_car > 8:  qam = 16
        m = int(math.sqrt(qam))
        # Clamp m to reasonable sqrt
        while m*m != qam and m > 2: m -= 1
        qam = m*m
        iq_pts = generate_iq_cloud(snr_car, n_pts=300, qam_order=max(4, qam))
        self.iq_scatter.set_offsets(iq_pts)
        self.ax_iq.set_title(
            f"IQ Constellation (Car-UE | {qam}-QAM)",
            color="#74b9ff", fontsize=9, pad=4)
        self.ax_iq.tick_params(colors="white", labelsize=7)

        # =========================================================
        # PANEL 8 — Link Budget Waterfall (Car-UE)
        # =========================================================
        self.ax_lb.cla()
        self.ax_lb.set_facecolor("#0a0f1e")
        self.ax_lb.set_title("Link Budget (Car-UE)", color="#e17055", fontsize=9, pad=4)
        cr0 = channel_results[0]
        labels = [
            "Tx Power",
            "gNB Ant Gain",
            "UE Ant Gain",
            "−FSPL",
            "−Mol. Abs",
            f"{'−Rain' if self.rain_mode else '−Shadow'}",
            "−Impl. Loss",
            "= Rx Power",
            "Noise Floor",
            "= SNR",
        ]
        rain_sh_loss = cr0["rain_loss_db"] if self.rain_mode else 2.0
        values = [
            THz6GConfig.GNB_TX_POWER_DBM,
            THz6GConfig.GNB_ANT_GAIN_DBI,
            THz6GConfig.UE_ANT_GAIN_DBI,
            -cr0["pl_los_db"],
            -cr0["abs_loss_db"],
            -rain_sh_loss,
            -THz6GConfig.IMPLEMENTATION_LOSS_DB,
            cr0["rx_power_dbm"],
            cr0["noise_floor_dbm"],
            cr0["snr_db"],
        ]
        sep_idx = [7, 8]  # draw a line before index 7, 8
        bar_colors = []
        for j, v in enumerate(values):
            if j in sep_idx:
                bar_colors.append("#e17055")
            elif v < 0:
                bar_colors.append("#d63031")
            else:
                bar_colors.append("#00b894")

        bars = self.ax_lb.barh(labels, values, color=bar_colors, height=0.6)
        for bar, v in zip(bars, values):
            self.ax_lb.text(
                v + (0.5 if v >= 0 else -0.5),
                bar.get_y() + bar.get_height()/2,
                f"{v:+.1f}", va="center",
                color="white", fontsize=6.5,
                ha="left" if v >= 0 else "right")
        self.ax_lb.axvline(0, color="white", lw=0.5)
        self.ax_lb.set_xlabel("dB / dBm", color="#636e72", fontsize=7)
        self.ax_lb.tick_params(colors="white", labelsize=6.5)
        self.ax_lb.grid(True, axis="x", color="#1a2a3a", linewidth=0.5)

    # ── Main Run Loop ─────────────────────────────────────────────────────────

    def run(self):
        plt.ion()
        print("\n" + "="*70)
        print("  6G THz DIGITAL TWIN STARTED")
        print(f"  Carrier: {THz6GConfig.CARRIER_FREQ_HZ/1e9:.0f} GHz"
              f" | BW: {THz6GConfig.BANDWIDTH_HZ/1e9:.0f} GHz"
              f" | Numerology μ={THz6GConfig.NUMEROLOGY_MU}")
        print(f"  gNB: 1024-element MIMO @ {THz6GConfig.GNB_TX_POWER_DBM} dBm")
        print(f"  UEs: Car (60 km/h), Pedestrian (5 km/h), IoT Sensor (fixed)")
        print(f"  Noise Floor: {THz6GConfig.thermal_noise_dbm():.1f} dBm")
        print()
        print("  CONTROLS:  [SPACE] Pause   [R] Toggle Rain   [Q] Quit")
        print("="*70 + "\n")

        # Validation block
        d_test = 10.0
        pl_test = THz_Channel.fspl_db(d_test, THz6GConfig.CARRIER_FREQ_HZ)
        cap_test = THz6GConfig.shannon_capacity_gbps(40)
        print(f"  VALIDATION:")
        print(f"    FSPL @10m, 300GHz  = {pl_test:.1f} dB  (expected ~92 dB)")
        print(f"    Shannon Cap @40dB  = {cap_test:.1f} Gbps  (expected >100 Gbps)")
        print(f"    Noise Floor         = {THz6GConfig.thermal_noise_dbm():.1f} dBm\n")

        while plt.fignum_exists(self.fig.number):
            if not self.paused:
                channel_results, actual_gbps, per_list = self.step_simulation()
                self.update_dashboard(channel_results, actual_gbps, per_list)

                # Console status every 50 steps
                if self.step % 50 == 0:
                    cr0 = channel_results[0]
                    print(f"  T={self.sim_time:.1f}s | Car-UE: d={cr0['d_los_m']:.1f}m"
                          f" SNR={cr0['snr_db']:.1f}dB"
                          f" Cap={THz6GConfig.shannon_capacity_gbps(cr0['snr_db']):.1f}Gbps"
                          f" Rain={'ON' if self.rain_mode else 'OFF'}"
                          f" MCS={self.ru.current_mcs_labels[0]}")

            plt.pause(0.001)

        plt.ioff()
        print("\n[*] Simulation ended.")
        self._print_final_stats()

    def _print_final_stats(self):
        print("\n" + "="*60)
        print("  FINAL FLOW MONITOR STATISTICS")
        print("="*60)
        for i, ue in enumerate(self.ues):
            stats = self.monitor.get_per_flow_stats(i)
            print(f"\n  Flow {i} — {ue.name}")
            print(f"    TX Packets : {stats['tx_packets']:>12,}")
            print(f"    RX Packets : {stats['rx_packets']:>12,}")
            print(f"    Packet Loss: {stats['per_pct']:>11.3f} %")
            print(f"    Last SNR   : {ue.snr_history[-1] if ue.snr_history else 0:>10.1f} dB")
            print(f"    Last Tput  : {ue.tput_history[-1] if ue.tput_history else 0:>10.2f} Gbps")
        print("="*60)

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    twin = DigitalTwin()
    twin.run()
