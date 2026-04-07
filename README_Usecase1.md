# 3GPP 6G THz Network — Real-Life Use Cases · Interactive Live Simulation

> **A single-file Python live simulation of 6G Terahertz wireless networks across six real-world scenarios, with NS3-aligned physics, real-time animation, and 8 interactive visualization panels.**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [How the Code Works — Big Picture](#how-the-code-works--big-picture)
4. [Physics Engine (NS3-Aligned)](#physics-engine-ns3-aligned)
5. [Six Real-Life Use Cases](#six-real-life-use-cases)
6. [Node Definition Format](#node-definition-format)
7. [Visualization Panels](#visualization-panels)
8. [Simulation State Machine](#simulation-state-machine)
9. [Mobility Models](#mobility-models)machine)
9. [Mobility Models](#mobility-models)
10. [Code Structure Walkthrough](#code-structure-walkthrough)
11. [Interactive Controls](#interactive-controls)
12. [CSV Export](#csv-export)
13. [Key Parameters to Tune](#key-parameters-to-tune)
14. [NS3 / C++ Mapping](#ns3--c-mapping)
15. [How to Add a New Use Case](#how-to-add-a-new-use-case)
16. [Dependencies](#dependencies)
17. [Glossary](#glossary)

---

## Overview

This code simulates **6G Terahertz (THz) wireless communication** for six real-world deployment scenarios in an animated Python window. Every ~55 ms it:

- Moves all nodes (vehicles, robots, drones, people) according to their mobility model
- Computes signal quality (SINR) at each node using a realistic THz physics model
- Derives throughput in Gbps and latency in ms from the SINR
- Refreshes 8 live visualization panels in real time

The physics models are directly aligned with **NS3 network simulator** modules, making this Python code useful for understanding, prototyping, or teaching before writing full NS3 C++ simulations.

---

## Quick Start

### Install dependencies

```bash
pip install matplotlib numpy scipy
```

### Run

```bash
python 6g_thz_live_sim.py
```

On first launch it pre-computes SINR heatmaps for all 6 environments (~10–30 s depending on CPU). These are cached in memory for the session.

---

## How the Code Works — Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    STARTUP  (once)                          │
│  • Build SINR heatmaps for all 6 environments              │
│  • Create matplotlib figure with 8 axes + control widgets  │
│  • Register keyboard/button callbacks                       │
└─────────────────────┬───────────────────────────────────────┘
                      │  FuncAnimation calls draw_frame()
                      ▼  every 55 ms
┌─────────────────────────────────────────────────────────────┐
│               draw_frame()  — one animation tick            │
│                                                             │
│  1. SIM.step(dt=0.04)       ← physics + movement update    │
│     ├── Move each node by velocity × dt                    │
│     ├── Apply boundary (bounce or wrap-around)             │
│     ├── Compute SINR  (Friis + THz absorption + BF)        │
│     ├── Compute throughput (Shannon capacity)              │
│     ├── Compute latency (exponential SINR model)           │
│     ├── Append to rolling histories (last 400 samples)     │
│     └── Detect handovers (nearest gNB changed?)            │
│                                                             │
│  2. Redraw all 8 panels with fresh data                    │
│  3. Update status bar text                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Physics Engine (NS3-Aligned)

The code implements four physical models. Each has a direct NS3 C++ equivalent.

### 1. Friis Free-Space Path Loss

```python
def friis_db(d_m, freq_hz):
    c = 3e8
    return 20 * log10(4 * pi * d * f / c)
```

**What it models:** Energy spreading out spherically as a radio wave travels through free space. At 300 GHz with a 12.5 mm wavelength, even 5 metres of distance causes ~70 dB of path loss.

**NS3 equivalent:** `ns3::FriisPropagationLossModel::DoCalcRxPower`

### 2. THz Molecular Absorption

```python
_THZ_ABS = {
    0.10e12: 0.40,   # 100 GHz  — low absorption window
    0.14e12: 0.50,   # 140 GHz  — preferred industrial window
    0.30e12: 1.20,   # 300 GHz  — indoor preferred window
    1.00e12: 8.00,   # 1 THz    — higher absorption
}

def thz_absorption_db(d_m, freq_hz):
    k = _THZ_ABS.get(freq_hz, 1.0)     # absorption coefficient [dB/m]
    return k * d * 10 / log(10)
```

**What it models:** Water vapour and oxygen molecules in air absorb THz energy. The absorption coefficient `k` (in dB/m) varies strongly with frequency — frequencies between absorption peaks are called "transmission windows". This is why 140 GHz and 300 GHz are chosen for indoor use.

**NS3 equivalent:** `ns3::ThzSpectrumPropagationLossModel::DoCalcRxPower`
**Source:** Jornet & Akyildiz, "Channel Modeling and Capacity Analysis for EM Nanonetworks in the THz Band", IEEE Trans. Wireless Commun. 2011

### 3. Full SINR Computation

```python
def compute_sinr_thz(nx, ny, gnbs_cfg, env_cfg, rng):
    for each gNB:
        distance d = sqrt((nx-gx)^2 + (ny-gy)^2)
        path_loss  = friis_db(d,f) + thz_absorption_db(d,f) + pen_loss*0.20
        Rx_power   = Tx_power - path_loss + BF_gain + Gaussian_noise(σ=1.8dB)

    # SINR = strongest signal / (all other signals + thermal noise)
    signal      = 10^(best_Rx / 10)             [linear power]
    interference = sum(10^(other_Rx / 10))
    noise        = 10^(noise_floor / 10)

    SINR = signal / (interference + noise)       [convert back to dB]
```

The `σ = 1.8 dB` Gaussian noise term models **log-normal shadowing** — random variation caused by obstacles and reflections not explicitly modelled (equivalent to NS3 `NakagamiPropagationLossModel`).

The `pen_loss * 0.20` term applies 20% of the wall penetration loss as an average indoor attenuation.

### 4. Shannon Capacity → Throughput

```python
def shannon_tp(sinr_db, bw_ghz):
    return bw_ghz * log2(1 + 10^(sinr_db/10)) * 0.65   # Gbps
```

**What it models:** Maximum theoretical data rate for a channel with given bandwidth and signal quality. The factor `0.65` is a practical THz efficiency that accounts for:
- Coding overhead (~10%)
- Guard intervals and control signalling (~8%)
- THz hardware impairments — phase noise, non-linearity (~17%)

### 5. Latency Model

```python
latency_ms = max(0.05, latency_target * 2 * exp(-sinr / 15))
```

**What it models:** As SINR drops, more HARQ (Hybrid Automatic Repeat reQuest) retransmissions are needed, which increase latency exponentially. At high SINR the latency approaches its floor of 0.05 ms.

---

## Six Real-Life Use Cases

| # | Environment | Frequency | Bandwidth | Latency Target | Key Nodes |
|---|-------------|-----------|-----------|---------------|-----------|
| 1 | **XR Surgery** | 300 GHz | 100 GHz | < 1 ms | Surgeons, robots, holo-displays, cameras |
| 2 | **Auto Factory** | 140 GHz | 50 GHz | < 2 ms | AGVs, robot arms, IoT sensors |
| 3 | **Smart Intersection** | 300 GHz | 80 GHz | < 0.5 ms | Cars, drones, pedestrians, RSUs |
| 4 | **THz Backhaul** | 1 THz | 300 GHz | < 0.1 ms | Rooftop nodes, relay, drones |
| 5 | **Tunnel Rescue** | 100 GHz | 30 GHz | < 5 ms | Rescuers, drones, sensors in debris |
| 6 | **Holo Classroom** | 300 GHz | 60 GHz | < 3 ms | Students, holo-displays |

### Why these frequencies?

| Frequency | Reason for choice |
|-----------|------------------|
| 100 GHz | Low absorption, good penetration through debris — ideal for tunnel/confined space |
| 140 GHz | Best THz transmission window — low absorption, good BW — ideal for industrial |
| 300 GHz | Strong transmission window, high bandwidth — ideal for indoor room-scale |
| 1 THz | Extreme bandwidth (300 GHz) for backhaul, but higher absorption limits range to rooftop P2P |

---

## Node Definition Format

Each node in an environment's `"nodes"` list is a tuple:

```python
(id, type, x0, y0, vx, vy, service, bounce, ns3_model)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier shown on plot (`"Surg1"`, `"AGV2"`, etc.) |
| `type` | string | Visual style key — must match a `NODE_STYLE` entry |
| `x0`, `y0` | float | Initial position in metres |
| `vx`, `vy` | float | Velocity in metres/second |
| `service` | string | 3GPP service class: `URLLC`, `eMBB`, `mMTC`, `XR`, or `V2X` |
| `bounce` | bool | `True` = reflects off walls; `False` = wraps around (highway/open) |
| `ns3_model` | string | NS3 mobility class name (documentation only, not executed) |

**Example:**
```python
("AGV1", "agv", 10, 30, 0.80, 0.00, "URLLC", False,
 "ns3::ConstantVelocityMobilityModel")
```
→ AGV starting at (10, 30) m, moving at 0.8 m/s east, URLLC service, wraps around, uses constant velocity in NS3.

---

## Visualization Panels

The 24×14 inch figure contains 8 panels:

### Main Area (top 2 rows, spanning 3×2 columns)

| Panel | Description |
|-------|-------------|
| **Network Topology** | The main 2D map. Shows buildings (colored by material), gNB towers with pulsing coverage rings, moving nodes with SINR-colored motion trails, link lines to nearest gNB, and a SINR quality dot per node. The background heatmap shows pre-computed signal coverage. |

### Right Column (top rows)

| Panel | Description |
|-------|-------------|
| **Live SINR Bars** | Horizontal bar per node, color-coded: Green (≥18 dB excellent), Orange (10–18 dB good), Red (<10 dB marginal). Updates every frame. |
| **Throughput Timeline** | Per-node throughput in Gbps, rolling history of last ~16 seconds. |
| **Latency Timeline** | Per-node latency in ms with a red dashed target line. |

### Second Row (right columns)

| Panel | Description |
|-------|-------------|
| **SINR History** | Rolling SINR time series for all active nodes. Background bands show quality zones (green/orange/red). |
| **SINR vs Distance** | Live scatter plot — current SINR vs distance to nearest gNB. Shows the path loss effect clearly. |
| **SINR CDF** | Cumulative Distribution Function — what fraction of time each node exceeds a given SINR. Useful for reliability analysis. |

### Full-Width Bottom Panel

| Panel | Description |
|-------|-------------|
| **Aggregate Throughput** | Total system throughput (solid line) + per-service dashed overlays (URLLC, eMBB, mMTC, XR, V2X). Includes handover counter in title. |

---

## Simulation State Machine

All simulation state is stored in a single `SimState` object:

```
SimState
│
├── env_name          ← active environment name
├── paused            ← animation paused flag
├── speed             ← time multiplier (0.1× – 6×)
├── t                 ← simulation clock [seconds]
├── frame             ← frame counter
│
├── pos[nid]          ← [x, y] current position (updated every step)
├── vel[nid]          ← [vx, vy] current velocity (may flip on bounce)
├── bounce[nid]       ← True = wall reflection, False = wrap-around
│
├── sinr_hist[nid]    ← list of last 400 SINR values [dB]
├── tp_hist[nid]      ← list of last 400 throughput values [Gbps]
├── lat_hist[nid]     ← list of last 400 latency values [ms]
│
├── trail_x[nid]      ← last 60 x-positions (for drawing trails)
├── trail_y[nid]      ← last 60 y-positions
│
├── prev_cell[nid]    ← index of gNB the node was connected to last step
├── handovers         ← total count of cell switches detected
│
├── total_tp          ← list of last 400 total-system throughput values
└── heatmaps          ← {env_name: (XX, YY, SINR_grid)} — cached, computed once
```

### Step function logic

```python
def step(self, dt=0.04):
    for each node:
        1. x += vx * dt,  y += vy * dt
        2. if bounce: reflect velocity at walls
           else: wrap around at edges (highway model)
        3. sinr = compute_sinr_thz(x, y, ...)
        4. tp = shannon_tp(sinr, bw_ghz)
        5. lat = max(0.05, lat_target * 2 * exp(-sinr/15))
        6. append sinr/tp/lat to rolling histories (trim if >400)
        7. append x/y to trail (trim if >60)
        8. nearest_cell = argmin(distance to each gNB)
        9. if nearest_cell != prev_cell: handovers++
```

---

## Mobility Models

Two boundary behaviors are implemented, matching NS3 mobility models:

### Bounce (RandomWalk2d equivalent)

```python
if bounce[nid]:
    if x <= 0 or x >= W:  vx *= -1   # reflect off left/right wall
    if y <= 0 or y >= H:  vy *= -1   # reflect off top/bottom wall
```

Used for: pedestrians, drones, students — nodes that stay within a bounded area.
**NS3 equivalent:** `ns3::RandomWalk2dMobilityModel` (with reflection mode)

### Wrap-Around (ConstantVelocity equivalent)

```python
else:
    if x > W + 2:  x = -2    # re-enter from left
    if x < -2:     x = W + 2  # re-enter from right
```

Used for: cars, AGVs, backhaul nodes — things that keep moving in one direction.
**NS3 equivalent:** `ns3::ConstantVelocityMobilityModel`

### Stationary

Nodes with `vx=0, vy=0` never move. Used for: base stations (in topology), cameras, robot arms.
**NS3 equivalent:** `ns3::ConstantPositionMobilityModel`

---

## Code Structure Walkthrough

```
6g_thz_live_sim.py
│
├── ── COLOUR PALETTE ──────────────────────────────────────────
│   BG, PANEL_BG, GRID_COL, TEXT_COL, MUTED, colour constants
│   SVC_COLOR: service type → hex colour string
│   NODE_STYLE: node type → {color, marker, sz, label}
│   MAT_COLOR / MAT_EDGE: building material → face/edge colour
│
├── ── THz PHYSICS FUNCTIONS ───────────────────────────────────
│   _THZ_ABS dict             absorption coefficients by frequency
│   thz_absorption_db(d, f)   molecular absorption loss
│   friis_db(d, f)            free-space path loss
│   compute_sinr_thz(...)     full SINR: Friis + THz abs + BF + shadowing
│   shannon_tp(sinr, bw)      Shannon capacity with η=0.65
│   sinr_qcolor(v)            SINR value → colour (green/orange/red)
│   build_heatmap_thz(cfg)    pre-compute 38×38 SINR grid
│
├── ── ENVIRONMENTS dict ───────────────────────────────────────
│   6 use cases, each containing:
│   desc, area, freq_hz, bw_ghz, tx_power_dbm, pen_loss_avg,
│   beamforming_gain_db, noise_floor_dbm, color, latency_target_ms,
│   buildings [(x,y,w,h,label,material,pen_loss)],
│   gnbs [(x,y,label,height)],
│   nodes [(id,type,x,y,vx,vy,service,bounce,ns3_model)]
│
├── ── SimState class ──────────────────────────────────────────
│   __init__()    set defaults, call reset()
│   cfg property  returns ENVIRONMENTS[env_name]
│   reset()       re-initialize positions, histories; build heatmap if needed
│   step(dt)      advance one time step (mobility + physics + history)
│   export_csv()  write current state to CSV file
│
├── ── FIGURE SETUP ────────────────────────────────────────────
│   plt.rcParams  dark theme settings
│   GridSpec      defines 8 plot axes + 7 control panel axes
│   fig.text()    title and subtitle
│
├── ── WIDGETS ─────────────────────────────────────────────────
│   radio_env     RadioButtons  — switch environment
│   chk_svc       CheckButtons  — filter by service type
│   chk_disp      CheckButtons  — toggle heatmap / trails / links
│   sl_spd        Slider        — simulation speed 0.1× – 6×
│   btn_pause     Button        — pause/resume
│   btn_reset     Button        — reset
│   btn_export    Button        — export CSV
│
├── ── draw_frame(_fn) ─────────────────────────────────────────
│   Called by FuncAnimation every 55 ms.
│   Calls SIM.step(), then redraws all 8 panels.
│
├── ── on_key(ev) ──────────────────────────────────────────────
│   Keyboard handler for SPACE, R, E, 1-6, +/-, H, T, L
│
└── ── STARTUP ─────────────────────────────────────────────────
    Pre-compute all heatmaps → FuncAnimation → plt.show()
```

---

## Interactive Controls

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume animation |
| `R` | Reset — restart node positions for current environment |
| `E` | Export current SINR/throughput/latency snapshot to CSV |
| `1` | Switch to XR Surgery |
| `2` | Switch to Auto Factory |
| `3` | Switch to Smart Intersection |
| `4` | Switch to THz Backhaul |
| `5` | Switch to Tunnel Rescue |
| `6` | Switch to Holo Classroom |
| `+` or `=` | Speed up simulation by 0.5× |
| `-` | Slow down simulation by 0.5× |
| `H` | Toggle SINR heatmap background on topology |
| `T` | Toggle node motion trails |
| `L` | Toggle link lines to nearest gNB |

### On-Screen Widgets

| Widget | Description |
|--------|-------------|
| **Use Case** radio buttons | Click any environment to switch immediately |
| **Service** checkboxes | Show/hide nodes by service type (URLLC, eMBB, mMTC, XR, V2X) |
| **Display** checkboxes | Toggle Heatmap / Trails / Links independently |
| **Speed** slider | Drag from 0.1× (slow-motion) to 6× (fast-forward) |
| **Pause** button | Same as SPACE key |
| **Reset** button | Same as R key |
| **CSV** button | Same as E key |

---

## CSV Export

Pressing `E` or clicking the CSV button creates a file named:
```
thz_export_<EnvName>_<timestamp>.csv
```

Columns:
```
Node, Type, Service, X_m, Y_m, SINR_dB, Throughput_Gbps, Latency_ms, NS3_Mobility
```

Example rows:
```csv
Node,Type,Service,X_m,Y_m,SINR_dB,Throughput_Gbps,Latency_ms,NS3_Mobility
Surg1,surgeon,URLLC,3.00,5.00,21.34,186.2,0.083,ns3::ConstantPositionMobilityModel
RobotL,robot,XR,4.87,4.50,19.12,156.8,0.097,ns3::ConstantVelocityMobilityModel
Holo,holo_disp,XR,6.00,7.50,22.81,205.4,0.071,ns3::ConstantPositionMobilityModel
```

This CSV can be imported into Excel, MATLAB, or Python for further analysis. The `NS3_Mobility` column documents which NS3 module would be used if porting the simulation to C++.

---

## Key Parameters to Tune

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `res=38` in `build_heatmap_thz()` | Physics | 38 | Heatmap grid resolution — increase for sharper heatmaps, slower startup |
| `interval=55` in `FuncAnimation` | Rendering | 55 ms | Frame period — decrease for smoother animation (needs faster CPU) |
| `dt=0.04` in `step()` | Physics | 40 ms | Simulation time step per frame |
| `400` in history trimming | SimState | 400 | Rolling history length (~16 s at dt=0.04) |
| `60` in trail trimming | SimState | 60 | Motion trail length in positions |
| `σ=1.8` in `rng.normal(0, 1.8)` | Physics | 1.8 dB | Shadowing standard deviation |
| `0.65` in `shannon_tp` | Physics | 0.65 | THz practical spectral efficiency |
| `0.20` in `pen * 0.20` | Physics | 20% | Fraction of wall loss applied as diffuse indoor attenuation |

---

## NS3 / C++ Mapping

This table shows the full Python → NS3 C++ correspondence for every physics element:

| Python function / parameter | NS3 C++ equivalent |
|-----------------------------|--------------------|
| `friis_db(d, f)` | `ns3::FriisPropagationLossModel` |
| `thz_absorption_db(d, f)` | `ns3::ThzSpectrumPropagationLossModel` |
| `rng.normal(0, 1.8)` shadowing | `ns3::NakagamiPropagationLossModel` |
| `shannon_tp(sinr, bw)` | `ns3::NrMacScheduler` (link adaptation) |
| `bounce=False` wrap | `ns3::ConstantVelocityMobilityModel` |
| `bounce=True` reflect | `ns3::RandomWalk2dMobilityModel` |
| stationary `vx=vy=0` | `ns3::ConstantPositionMobilityModel` |
| handover detection | `ns3::LteHandoverAlgorithm` |
| `sinr_hist` / `tp_hist` | `ns3::FlowMonitor` + `FlowMonitorHelper` |
| `export_csv()` | `ns3::FlowMonitor::SerializeToXmlFile()` |
| UDP traffic generation | `ns3::UdpClientHelper` + `UdpServerHelper` |
| `beamforming_gain_db` | `ns3::ThzDirectionalAntenna` |

---

## How to Add a New Use Case

1. Add a new entry to the `ENVIRONMENTS` dict:

```python
"My Use Case": {
    "desc": "Short description for the plot",
    "area": (50, 40),              # width × height in metres
    "freq_hz": 200e9,              # 200 GHz
    "bw_ghz": 70,                  # 70 GHz bandwidth
    "tx_power_dbm": 30,
    "pen_loss_avg": 18,            # average wall penetration loss
    "beamforming_gain_db": 28,     # MIMO BF gain
    "noise_floor_dbm": -83,
    "color": "#FF6347",            # environment accent colour
    "latency_target_ms": 1.5,
    "buildings": [
        (5, 5, 20, 15, "Lab A", "concrete", 22),
        (30, 5, 15, 15, "Lab B", "glass",    8),
    ],
    "gnbs": [
        (25, 20, "AP-Main", 3),
    ],
    "nodes": [
        ("Dev1", "sensor", 10, 10, 0.1, 0.0, "mMTC", True,
         "ns3::RandomWalk2dMobilityModel"),
        ("Dev2", "laptop", 35, 25, 0.0, 0.0, "eMBB", False,
         "ns3::ConstantPositionMobilityModel"),
    ],
}
```

2. Add any new node types to `NODE_STYLE`:

```python
"laptop": {"color": "#4CAF50", "marker": "s", "sz": 100, "label": "Laptop"},
```

3. The environment will appear automatically in the radio button list — no other changes needed.

---

## Dependencies

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| `matplotlib` | 3.7 | Plotting, animation, widgets |
| `numpy` | 1.24 | Array math, SINR computations |
| `scipy` | 1.10 | `gaussian_filter` for heatmap smoothing |

```bash
pip install matplotlib numpy scipy
```

No GPU required. Runs entirely on CPU. Tested on Python 3.9+.

### Backend options

The code uses `matplotlib.use("TkAgg")` at the top. If that backend is unavailable on your system, change it:

| System | Backend to use |
|--------|---------------|
| Linux (no display) | `"Agg"` (saves to file only) |
| macOS | `"MacOSX"` or `"Qt5Agg"` |
| Windows | `"TkAgg"` (default) or `"Qt5Agg"` |
| Jupyter | `%matplotlib widget` and remove `matplotlib.use()` |

---

## Glossary

| Term | Definition |
|------|------------|
| **SINR** | Signal-to-Interference-plus-Noise Ratio. Primary measure of wireless link quality in dB. Higher = better. |
| **THz** | Terahertz — frequencies 100 GHz to 10 THz. Used in 6G for extreme bandwidth. |
| **gNB** | Next-Generation Node B — the 5G/6G base station (what WiFi calls an "access point"). |
| **Friis PL** | Free-space path loss: energy dilutes as it spreads, proportional to distance squared and frequency squared. |
| **Molecular absorption** | THz energy absorbed by water vapour and O₂ in air — adds extra attenuation beyond Friis at THz frequencies. |
| **Transmission window** | A frequency range with low molecular absorption — 140 GHz and 300 GHz are key THz windows. |
| **BF gain** | Beamforming gain — electronically steering antenna energy toward a specific node, boosting received power. |
| **URLLC** | Ultra-Reliable Low-Latency Communication — 6G service type for safety-critical real-time applications. |
| **eMBB** | Enhanced Mobile Broadband — 6G service for high-throughput applications like video and holographics. |
| **mMTC** | Massive Machine-Type Communication — low-power, low-rate IoT sensor connectivity. |
| **XR** | Extended Reality — holographic, augmented reality, and virtual reality applications. |
| **V2X** | Vehicle-to-Everything — wireless communication between vehicles, infrastructure, pedestrians, and networks. |
| **Handover** | When a node switches its connection from one gNB to a closer/stronger one. |
| **HARQ** | Hybrid Automatic Repeat reQuest — retransmission protocol that increases latency at low SINR. |
| **CDF** | Cumulative Distribution Function — shows what fraction of time a value is below a given threshold. |
| **Shannon capacity** | Theoretical maximum data rate for a channel: `BW × log₂(1 + SINR)`. |
| **η (eta)** | Practical spectral efficiency — fraction of Shannon capacity actually achieved (set to 0.65 here). |
| **NS3** | Network Simulator 3 — open-source C++ discrete-event simulator for networking research. |
| **AGV** | Automated Guided Vehicle — autonomous robot navigating a factory floor. |
| **RSU** | Road Side Unit — roadside infrastructure node supporting V2X communications. |
| **NLOS** | Non-Line-of-Sight — signal path blocked by buildings or obstacles, causing additional loss. |
| **Shadowing** | Random signal variation caused by unmodelled objects. Modelled here as Gaussian noise on received power. |
| **Log-normal shadowing** | Standard deviation of signal variation in dB follows a normal distribution. σ = 1.8 dB in this code. |

---

## File Summary

```
6g_thz_live_sim.py          ← Complete simulation (~550 lines, single file)
README.md                   ← This document
thz_export_<env>_<ts>.csv   ← Created when you press E (optional)
```

---

*Physics models aligned with:*
- *3GPP TR 22.261 (6G service requirements)*
- *IEEE 802.15.3d (THz communications standard)*
- *ITU-R P.676 (atmospheric attenuation)*
- *Jornet & Akyildiz, IEEE Trans. Wireless Commun. 2011 (THz absorption)*
- *NS3 ThzSpectrumPropagationLoss module*

*Built with Python · matplotlib · NumPy · SciPy*
