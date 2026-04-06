# 3GPP 6G THz Network — SINR Radar Charts + Multi-Layer Heatmaps (Live Simulation)

> **Interactive Python simulation of 6G Terahertz wireless networks with real-time SINR radar charts, polar heatmaps, per-service overlays, KPI scoring, and live animation across 6 real-world use cases.**

---

## Table of Contents

1. [What This Code Does](#what-this-code-does)
2. [Quick Start](#quick-start)
3. [Physics Engine](#physics-engine)
4. [Use Cases (Environments)](#use-cases-environments)
5. [Visualization Panels (12 total)](#visualization-panels-12-total)
6. [Simulation Architecture](#simulation-architecture)
7. [Code Structure Walkthrough](#code-structure-walkthrough)
8. [Interactive Controls](#interactive-controls)
9. [Key Parameters](#key-parameters)
10. [Extending the Code](#extending-the-code)
11. [Dependencies](#dependencies)
12. [Glossary](#glossary)

---

## What This Code Does

This is a **single-file Python live simulation** of a 6G Terahertz (THz) wireless network. It opens an animated window showing nodes (surgeons, robots, cars, drones, factory AGVs, etc.) moving around real-world environments while computing realistic radio signal quality (SINR) in real time.

Every frame (~60 ms), the simulator:

1. **Moves all nodes** according to their velocity and mobility model (bouncing, wrap-around, or stationary).
2. **Computes SINR** for each node using a THz physics model (Friis path loss + molecular absorption + beamforming + shadowing).
3. **Derives throughput and latency** from the SINR using Shannon capacity.
4. **Refreshes 12 visualization panels** — topology map, heatmaps, polar SINR, KPI radar, time-series, CDF, histograms, and per-service overlays.
5. **Lets you switch environments, toggle layers, and export data** using keyboard shortcuts and on-screen widgets.

---

## Quick Start

### Install dependencies

```bash
pip install matplotlib numpy scipy
```

### Run the simulation

```bash
python 6g_thz_radar_heatmap.py
```

The first launch pre-computes SINR heatmaps for all 6 environments (takes ~10–30 seconds). These are cached in memory for the rest of the session.

---

## Physics Engine

The code models the THz radio channel using four physical effects:

### 1. Friis Free-Space Path Loss

```python
def friis_db(d, f):
    c = 3e8
    return 20 * log10(4 * pi * d * f / c)
```

This is the standard free-space propagation formula. At THz frequencies (100 GHz–1 THz), even short distances of 10–50 metres produce significant path loss because the wavelength is tiny (0.3 mm–3 mm).

### 2. THz Molecular Absorption

```python
_THZ_ABS = {0.10e12: 0.40, 0.14e12: 0.50, 0.30e12: 1.20, 1.00e12: 8.00}

def thz_abs_db(d, f):
    k = _THZ_ABS.get(f, 1.0)
    return k * d * 10 / log(10)
```

Water molecules and oxygen in the air absorb THz energy. The absorption coefficient `k` (in dB/m) varies by frequency — 140 GHz is a "window" with low absorption, while 1 THz has much higher absorption. This is why THz links are short-range.

### 3. SINR Calculation

```python
def compute_sinr(nx, ny, gnbs, cfg, rng):
    for each gNB:
        path_loss = friis_db(d, f) + thz_abs_db(d, f) + pen_loss * 0.20
        Rx_power  = Tx_power - path_loss + BF_gain + shadowing_noise
    
    signal      = strongest gNB (linear power)
    interference = sum of all other gNBs (linear power)
    noise       = thermal noise floor
    
    SINR = signal / (interference + noise)   [converted to dB]
```

The result is a Signal-to-Interference-plus-Noise Ratio in dB — higher is better. A value above 18 dB is "excellent"; below 10 dB is marginal.

### 4. Shannon Capacity (Throughput)

```python
def shannon_tp(sinr_db, bw_ghz):
    return bw_ghz * log2(1 + 10^(sinr_db/10)) * 0.65   # Gbps
```

This converts SINR into a realistic throughput estimate. The factor `0.65` is a practical THz hardware efficiency (accounts for overhead, coding, and hardware impairments). Bandwidths of 30–300 GHz are used depending on the environment.

### 5. Latency Model

```python
latency = max(0.05, latency_target * 2 * exp(-sinr / 15))   # ms
```

Higher SINR → lower latency. This exponential model reflects that good signal quality reduces the need for retransmissions, which are the main contributor to wireless latency.

---

## Use Cases (Environments)

Six real-world 6G deployment scenarios are defined in the `ENVIRONMENTS` dictionary.

| # | Name | Frequency | Bandwidth | Peak Target | Latency Target | Key Nodes |
|---|------|-----------|-----------|-------------|----------------|-----------|
| 1 | **XR Surgery** | 300 GHz | 100 GHz | ~800 Gbps | < 1 ms | Surgeons, robots, holo-displays |
| 2 | **Auto Factory** | 140 GHz | 50 GHz | ~200 Gbps | < 2 ms | AGVs, robot arms, IoT sensors |
| 3 | **Smart Intersection** | 300 GHz | 80 GHz | ~400 Gbps | < 0.5 ms | Cars, drones, pedestrians, RSUs |
| 4 | **THz Backhaul** | 1 THz | 300 GHz | ~1.8 Tbps | < 0.1 ms | Rooftop nodes, relay, drones |
| 5 | **Tunnel Rescue** | 100 GHz | 30 GHz | ~80 Gbps | < 5 ms | Rescuers, drones, debris sensors |
| 6 | **Holo Classroom** | 300 GHz | 60 GHz | ~300 Gbps | < 3 ms | Students, holo-displays, teacher |

Each environment definition contains:

```python
"XR Surgery": {
    "desc":                  # short description shown on plot
    "area": (12, 10),        # room/space size in metres (W × H)
    "freq_hz": 300e9,        # carrier frequency in Hz
    "bw_ghz": 100,           # channel bandwidth in GHz
    "tx_power_dbm": 20,      # transmitter power
    "pen_loss_avg": 8,       # average wall/material penetration loss (dB)
    "beamforming_gain_db": 30, # massive MIMO beamforming gain (dBi)
    "noise_floor_dbm": -80,  # thermal noise floor
    "latency_target_ms": 1.0,# 3GPP KPI target
    "buildings": [...],      # list of physical obstacles (x,y,w,h,label,material,loss)
    "gnbs": [...],           # gNB/AP locations (x, y, label, height)
    "nodes": [...],          # UE definitions (id, type, x, y, vx, vy, service, bounce)
}
```

---

## Visualization Panels (12 total)

The 26×16 inch figure is divided into rows:

### Row 0 — Main Signal Analysis

| Panel | What it shows |
|-------|--------------|
| **Network Topology** | Moving nodes with SINR-colored motion trails, pulsing gNB coverage rings, link lines to nearest gNB, SINR quality dots |
| **SINR Heatmap + Contours** | Filled contour map of signal quality across the entire area; white dB labels; node dots with current SINR value |
| **Polar SINR Map** | Angular SINR field around the primary gNB, shown in polar coordinates — reveals beam directionality |
| **KPI Radar Chart** | 6-axis spider chart for the first active node: SINR score, Throughput, Latency, Coverage, BF Gain, Reliability |

### Row 1 — Time Dynamics

| Panel | What it shows |
|-------|--------------|
| **Live SINR Bars** | Horizontal bar chart per node — color-coded Good/Fair/Poor, updates every frame |
| **Throughput Timeline** | Per-node throughput in Gbps, scrolling history |
| **Latency Timeline** | Per-node latency in ms with target threshold line |
| **SINR History** | Rolling SINR time series with quality band shading (green/orange/red) |
| **SINR CDF** | Cumulative distribution function of SINR — shows what fraction of time each node exceeds a given quality level |

### Row 2 — Per-Service Heatmaps

Five side-by-side heatmaps, one per service type: **URLLC, eMBB, mMTC, XR, V2X**. Each panel shows only the nodes belonging to that service overlaid on the SINR field. Panel borders are color-coded to match the service color.

### Row 3 — System-Level Metrics

| Panel | What it shows |
|-------|--------------|
| **Aggregate Throughput** | Total system throughput + per-service dashed overlays, with handover counter |
| **SINR Histogram** | Rolling stacked bar histogram of all SINR samples, color-coded by service, with mean/median lines |

---

## Simulation Architecture

```
SimState (class)
│
├── env_name          ← which of the 6 environments is active
├── pos / vel         ← current position and velocity of every node (dict by node ID)
├── bounce            ← True = bounce off walls, False = wrap-around (highways)
│
├── sinr_hist[nid]    ← rolling list of SINR values (last 400 samples)
├── tp_hist[nid]      ← rolling list of throughput values
├── lat_hist[nid]     ← rolling list of latency values
├── trail_x/y[nid]    ← last 60 position samples for drawing trails
│
├── heatmaps          ← cached {env_name: (XX, YY, SINR_grid)} dict
├── total_tp          ← rolling total system throughput
├── handovers         ← count of cell changes detected
│
└── step(dt)          ← advances one time step:
    ├── move each node by velocity × dt
    ├── apply boundary conditions
    ├── compute SINR at new position
    ├── compute throughput and latency
    ├── update histories and trails
    └── detect handovers (nearest gNB changed?)
```

### Frame loop (FuncAnimation)

```
draw_frame() is called every 60 ms:
    SIM.step(dt=0.04)          ← physics update
    
    for each of 12 panels:
        ax.cla()               ← clear the axes
        [draw new content]     ← render fresh data
    
    update status label
    return []
```

### Heatmap computation

Heatmaps are pre-computed once per environment at startup (res=45 points per axis = 2025 SINR evaluations). They use `gaussian_filter` for smooth rendering and are cached so the topology and heatmap panels never recompute them during live animation.

### Radar chart

The `node_radar_values()` function converts raw physics values into 0–1 normalized scores for 6 KPI axes. For example, SINR is mapped as `(sinr + 5) / 30`, so a −5 dB signal scores 0 and a 25 dB signal scores 1. The `draw_radar()` function then draws the polar spider chart on any `projection="polar"` axes.

---

## Code Structure Walkthrough

```
6g_thz_radar_heatmap.py
│
├── PALETTE                    # color constants for dark theme
├── SVC_COLOR                  # service type → color mapping
├── NODE_STYLE                 # node type → marker/color/size
├── MAT_COLOR / MAT_EDGE       # building material → colors
├── _SINR_CMAP                 # custom red→yellow→green colormap
│
├── PHYSICS FUNCTIONS
│   ├── thz_abs_db(d, f)       # THz molecular absorption loss
│   ├── friis_db(d, f)         # Free-space path loss
│   ├── compute_sinr(...)      # Full SINR computation
│   ├── shannon_tp(...)        # Shannon capacity → Gbps
│   ├── sinr_qcolor(v)         # SINR value → color string
│   └── build_heatmap(cfg)     # Pre-compute SINR grid
│
├── ENVIRONMENTS dict          # 6 use-case definitions
│
├── SimState class             # All simulation state and step()
│
├── RADAR FUNCTIONS
│   ├── node_radar_values()    # Compute 6 KPI scores 0–1
│   ├── draw_radar()           # Draw spider chart on polar axes
│   └── draw_polar_sinr()      # Draw polar SINR map
│
├── FIGURE SETUP               # Create 26×16 inch figure with GridSpec layout
│
├── WIDGETS                    # RadioButtons, CheckButtons, Slider, Buttons
│   ├── radio_env              # Use case selector
│   ├── chk_svc                # Service type filter
│   ├── chk_disp               # Display options
│   └── sl_spd                 # Speed slider
│
├── draw_frame(_fn)            # Main animation callback (12 panels)
│
├── on_key(ev)                 # Keyboard event handler
│
└── STARTUP                    # Pre-compute heatmaps → launch FuncAnimation
```

---

## Interactive Controls

| Key / Widget | Action |
|---|---|
| `SPACE` | Pause / Resume simulation |
| `R` | Reset current environment (restart positions) |
| `E` | Export current SINR/TP/latency snapshot to CSV file |
| `1` – `6` | Switch between the 6 use-case environments |
| `+` / `=` | Speed up simulation |
| `-` | Slow down simulation |
| `H` | Toggle SINR heatmap overlay on topology |
| `T` | Toggle node motion trails |
| `L` | Toggle link lines to nearest gNB |
| **Use Case radio** | Click to switch environment |
| **Service checkboxes** | Show/hide nodes by service type (URLLC, eMBB, mMTC, XR, V2X) |
| **Display checkboxes** | Toggle Heatmap / Trails / Links |
| **Speed slider** | Drag to set simulation speed (0.1× – 6×) |
| **Pause button** | Same as SPACE |
| **Reset button** | Same as R |
| **Export CSV button** | Same as E |

---

## Key Parameters

| Parameter | Location | Effect |
|---|---|---|
| `res=45` in `build_heatmap()` | Physics | Heatmap resolution — increase for sharper maps, decrease for faster startup |
| `interval=60` in `FuncAnimation` | Animation | ms between frames — decrease for smoother animation, increase for lower CPU |
| `dt=0.04` in `step()` | Physics | Simulation time step in seconds |
| `_POLAR_SKIP = 4` | Rendering | Recompute polar SINR every N frames (expensive) |
| `400` in history lists | SimState | Length of rolling history in samples |
| `60` in trail lists | SimState | Length of motion trail in positions |
| `η = 0.65` in `shannon_tp` | Physics | THz practical efficiency factor |

---

## Extending the Code

### Add a new environment

Copy any existing entry in the `ENVIRONMENTS` dict and modify:

```python
"My Scenario": {
    "desc": "My description",
    "area": (50, 40),          # metres
    "freq_hz": 200e9,          # 200 GHz
    "bw_ghz": 80,
    "tx_power_dbm": 30,
    "pen_loss_avg": 15,
    "beamforming_gain_db": 28,
    "noise_floor_dbm": -83,
    "color": "#FF6347",
    "latency_target_ms": 1.0,
    "buildings": [(x, y, w, h, "Label", "material", pen_loss_db), ...],
    "gnbs": [(x, y, "Label", height_m), ...],
    "nodes": [(id, type, x, y, vx, vy, service, bounce), ...],
}
```

### Add a new node type

Add an entry to `NODE_STYLE`:

```python
"uav": {"color": "#FF4500", "marker": "D", "sz": 120, "label": "UAV"},
```

Then reference `"uav"` as the type in node tuples.

### Upgrade the physics model

Replace `compute_sinr()` with a more detailed model. The function signature `(nx, ny, gnbs, cfg, rng) → float (dB)` must be preserved. You can add Rician/Rayleigh fading, O2I penetration loss, or 3GPP UMi/UMa path loss models inside.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `matplotlib` | ≥ 3.7 | All plotting, animation, widgets |
| `numpy` | ≥ 1.24 | Array math, physics calculations |
| `scipy` | ≥ 1.10 | `gaussian_filter` for heatmap smoothing |

Install with:

```bash
pip install matplotlib numpy scipy
```

No GPU or special hardware required. The simulation runs entirely on CPU.

---

## Glossary

| Term | Definition |
|---|---|
| **SINR** | Signal-to-Interference-plus-Noise Ratio — primary measure of wireless link quality, in dB |
| **THz** | Terahertz — frequencies from 100 GHz to 10 THz, used in next-generation 6G networks |
| **gNB** | Next-Generation Node B — the 5G/6G base station that serves user equipment |
| **BF gain** | Beamforming gain — signal amplification from directing antenna energy toward a specific node |
| **Friis PL** | Free-space path loss formula: energy spreads out as it travels, proportional to distance² |
| **Molecular absorption** | Water and oxygen molecules absorb THz radiation, causing additional path loss |
| **URLLC** | Ultra-Reliable Low-Latency Communication — 6G service for safety-critical applications |
| **eMBB** | Enhanced Mobile Broadband — high-throughput service for video, AR/VR |
| **mMTC** | Massive Machine-Type Communication — low-power IoT sensor connectivity |
| **XR** | Extended Reality — holographic, AR, and VR applications |
| **V2X** | Vehicle-to-Everything — communication between vehicles, infrastructure, and pedestrians |
| **CDF** | Cumulative Distribution Function — shows what percentage of time a value is below a threshold |
| **AGV** | Automated Guided Vehicle — self-driving factory robot |
| **RSU** | Road Side Unit — roadside infrastructure node supporting V2X |
| **KPI** | Key Performance Indicator — metric used to assess whether a system meets its target |
| **Shannon capacity** | Theoretical maximum data rate: `BW × log₂(1 + SINR)` |
| **NLOS** | Non-Line-of-Sight — signal path obstructed by buildings or obstacles |
| **η (eta)** | Spectral efficiency factor — fraction of Shannon capacity achieved in practice |

---

## File Summary

```
6g_thz_radar_heatmap.py   ← The complete simulation (single file, ~700 lines)
README.md                 ← This document
```

When the simulation runs it may also create:

```
thz_sinr_<env>_<timestamp>.csv   ← Exported SINR/TP/latency snapshot (E key)
```

---

*Built with Python · matplotlib · NumPy · SciPy*
*Physics models aligned with 3GPP TR 38.901, IEEE 802.15.3d, and ITU-R P.676*
