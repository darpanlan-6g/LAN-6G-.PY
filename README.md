# 🛰️ LAN-6G-.PY — 3GPP 6G Network Live Simulation Suite

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/3GPP-6G%20NR%20FR3-brightgreen" />
  <img src="https://img.shields.io/badge/Frequency-24%20GHz%20%E2%80%93%201%20THz-purple" />
  <img src="https://img.shields.io/badge/License-MIT-orange" />
  <img src="https://img.shields.io/badge/Simulation-Real--Time%20Matplotlib-red" />
</p>

> **Interactive, real-time 6G network simulation** visualising SINR heatmaps, live node mobility, throughput timelines, handover detection and KPI dashboards — all powered by **3GPP-aligned THz physics** and rendered with Matplotlib.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Simulation Scripts](#simulation-scripts)
- [Physics & RF Model](#physics--rf-model)
- [Environments & Use Cases](#environments--use-cases)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Controls](#controls)
- [Dashboard Panels](#dashboard-panels)
- [CSV Export](#csv-export)
- [NS3 / C++ Alignment](#ns3--c-alignment)
- [Requirements](#requirements)

---

## Overview

This repository contains **three progressively advanced** Python live-simulation scripts that model next-generation 6G wireless networks operating in the **FR3 (24 GHz)** and **THz (100 GHz – 1 THz)** bands. Each simulation renders an animated, multi-panel dashboard in a Matplotlib window with full interactive controls.

| Script | Band | Environments | Key Feature |
|--------|------|-------------|-------------|
| `6g_nr_fr3_live_sim.py` | 24 GHz (FR3) | 4 | Core live simulation |
| `6g_thz_live_sim.py` | 100 GHz – 1 THz | 6 | THz use cases + NS3 alignment |
| `6g_thz_radar_heatmaps.py` | 100 GHz – 1 THz | 6 | SINR Radar charts + multi-layer heatmaps |

---

## Repository Structure

```
LAN-6G-.PY/
├── 6g_nr_fr3_live_sim.py       # 6G NR FR3 @ 24 GHz — 4 environments
├── 6g_thz_live_sim.py          # 6G THz real-life use cases — NS3 aligned
├── 6g_thz_radar_heatmaps.py    # THz + SINR Radar charts + multi-layer heatmaps
└── README.md
```

---

## Simulation Scripts

### 1. `6g_nr_fr3_live_sim.py` — 6G NR FR3 @ 24 GHz

The core simulation targeting **3GPP 6G NR FR3** at 24 GHz with 200–400 MHz bandwidth and 32×32 Massive MIMO. Models four real-world deployment environments with live node animation, SINR heatmaps, per-node throughput bars, handover counters, and CSV export.

**Environments:** Office · Urban Streets · Highway · Classroom

**Highlights:**
- 200–400 MHz bandwidth per environment
- Live SINR heatmap (pre-computed, Gaussian-smoothed)
- Animated pulse rings on gNB towers
- Per-node trail lines colour-coded by SINR quality
- Shannon capacity estimator with 0.6 efficiency factor

---

### 2. `6g_thz_live_sim.py` — 6G THz Real-Life Use Cases

An advanced simulation aligned with **NS3 C++ modules** (`ThzSpectrumPropagationLoss`, `FlowMonitor`, `LteHandoverAlgorithm`). Covers six realistic THz deployment scenarios with latency modelling, node NS3 mobility labels, and full CSV export including latency data.

**Use Cases:**
1. Holographic XR Surgery — 300 GHz, <1 ms URLLC
2. Autonomous Factory — 140 GHz, URLLC + mMTC
3. Smart City Intersection — 300 GHz, V2X, <0.5 ms
4. THz Terabit Backhaul — 1 THz, 300 GHz BW, eMBB
5. Tunnel Rescue — 100 GHz, confined space, URLLC
6. Holographic Classroom — 300 GHz, 10 Gbps/user eMBB

---

### 3. `6g_thz_radar_heatmaps.py` — SINR Radar + Multi-Layer Heatmaps

The most feature-rich simulation. Extends Script 2 with:
- **SINR Radar / Spider charts** — 6-axis KPI wheel per node (SINR, Throughput, Latency, Coverage, BF-Gain, Reliability)
- **Polar SINR map** — angular beamforming coverage in polar coordinates
- **Filled contour heatmaps** — with ISO-contour overlays and dB labels
- **Per-service SINR heatmap grid** — URLLC / eMBB / mMTC / XR / V2X side by side
- **Rolling SINR histogram** — live stacked distribution with mean/median markers

---

## Physics & RF Model

All three simulators implement **NS3-aligned** THz physics:

### Friis Free-Space Path Loss
```
PL_friis [dB] = 20·log₁₀(4π·d·f / c)
```
Equivalent C++: `ns3::FriisPropagationLossModel`

### THz Molecular Absorption Loss
```
L_abs [dB] = k(f) · d · 10 / ln(10)
```
Where `k(f)` is the molecular absorption coefficient (Jornet & Akyildiz, 2011):

| Frequency | k (1/m) |
|-----------|---------|
| 100 GHz   | 0.40    |
| 140 GHz   | 0.50    |
| 300 GHz   | 1.20    |
| 1 THz     | 8.00    |

Equivalent C++: `ns3::ThzSpectrumPropagationLossModel`

### Total Received Power
```
Rx [dBm] = Tx_power - Friis_PL - THz_absorption - penetration_loss + BF_gain + N(0, σ)
```
- σ = 1.8 dB shadowing
- BF gain = 18–40 dBi (Massive MIMO)

### SINR
```
SINR = Signal / (Interference + Noise)
```

### Shannon Throughput
```
Throughput = BW × log₂(1 + 10^(SINR/10)) × η
```
- η = 0.60 (FR3) / 0.65 (THz) — practical efficiency factor

### Latency Model
```
Latency_ms = max(0.05, Target_ms × 2 × e^(−SINR/15))
```

---

## Environments & Use Cases

### FR3 @ 24 GHz Environments

| Environment | Area | gNBs | Nodes | Services |
|-------------|------|------|-------|----------|
| Office | 100×80 m | 1 AP | 8 | URLLC, eMBB, mMTC |
| Urban Streets | 120×100 m | 2 gNB | 8 | URLLC, eMBB, V2X |
| Highway | 200×60 m | 2 gNB | 8 | URLLC, eMBB |
| Classroom | 90×70 m | 1 AP | 9 | URLLC, eMBB, mMTC |

### THz Use Cases

| # | Environment | Frequency | BW | Target Latency | Area |
|---|-------------|-----------|-----|----------------|------|
| 1 | XR Surgery | 300 GHz | 100 GHz | < 1 ms | 12×10 m |
| 2 | Auto Factory | 140 GHz | 50 GHz | < 2 ms | 80×60 m |
| 3 | Smart Intersection | 300 GHz | 80 GHz | < 0.5 ms | 100×100 m |
| 4 | THz Backhaul | 1 THz | 300 GHz | < 0.1 ms | 500×100 m |
| 5 | Tunnel Rescue | 100 GHz | 30 GHz | < 5 ms | 150×15 m |
| 6 | Holo Classroom | 300 GHz | 60 GHz | < 3 ms | 20×15 m |

### Node Types

| Type | Marker | Service | Description |
|------|--------|---------|-------------|
| 🔴 Surgeon | ★ | URLLC | Operating room personnel |
| 🟣 Robot Arm | ⬡ | URLLC / XR | Robotic surgery / factory arm |
| 🟢 IoT Sensor | + | mMTC | Fixed environmental sensor |
| 🟠 AGV | ◆ | URLLC | Autonomous guided vehicle |
| 🔵 Car | ▲ | V2X / URLLC | Autonomous vehicle |
| 🩵 Drone | ▼ | URLLC | UAV relay / survey |
| 🟢 Backhaul Node | ■ | eMBB | P2P THz backhaul terminal |
| 🔴 Rescuer | ✛ | URLLC | Body-worn rescue radio |
| 🩷 Holo Display | ❽ | XR | Holographic display unit |
| 🟣 Student | ● | eMBB | Mobile classroom device |
| 🩵 Camera | ✕ | URLLC | Fixed surveillance camera |
| 🟠 RSU | ⬡ | URLLC | Road-side unit |

---

## Features

### Simulation Engine
- ✅ Real-time node mobility with bounce / wrap-around boundary modes
- ✅ Per-step SINR, throughput, and latency computation
- ✅ Handover detection on cell-change events
- ✅ Pre-computed Gaussian-smoothed SINR heatmaps (cached per environment)
- ✅ Rolling history buffers (300–400 samples) for all KPI metrics

### Visualisation
- ✅ Animated topology with building overlays (concrete / glass / metal materials)
- ✅ Animated gNB pulse rings indicating coverage radius
- ✅ Node trails colour-coded by instantaneous SINR quality
- ✅ Link lines from node to best serving gNB
- ✅ SINR quality dot overlay per node (green / orange / red)
- ✅ Per-node live SINR horizontal bar chart
- ✅ Per-node throughput timeline
- ✅ Per-node latency timeline with target line
- ✅ SINR time series with quality bands
- ✅ SINR vs Distance scatter plot
- ✅ SINR CDF curves
- ✅ Aggregate system throughput with per-service breakdown
- ✅ **[Radar script]** 6-axis KPI radar / spider chart per node
- ✅ **[Radar script]** Polar SINR coverage map
- ✅ **[Radar script]** Filled contour SINR heatmap with ISO labels
- ✅ **[Radar script]** Per-service SINR heatmap grid (5 panels)
- ✅ **[Radar script]** Rolling stacked SINR histogram

### Interactive Controls
- ✅ Environment / use-case radio selector
- ✅ Service filter checkboxes (URLLC / eMBB / mMTC / XR / V2X)
- ✅ Display toggles (Heatmap / Trails / Links)
- ✅ Simulation speed slider (0.1× – 6×)
- ✅ Pause / Resume button
- ✅ Reset button
- ✅ CSV export button
- ✅ Full keyboard shortcut support

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/darpanlan-6g/LAN-6G-.PY.git
cd LAN-6G-.PY
```

### 2. Install dependencies
```bash
pip install matplotlib numpy scipy
```

> **Note:** Python 3.8 or higher is recommended.

### 3. (Optional) Backend setup

The scripts default to the **TkAgg** Matplotlib backend. If Tk is unavailable on your system, edit the backend line near the top of each script:

```python
# Options: "TkAgg"  "Qt5Agg"  "MacOSX"  "WXAgg"
matplotlib.use("TkAgg")
```

For Qt5 backend:
```bash
pip install PyQt5
```

---

## Usage

Run any of the three simulation scripts directly:

```bash
# 6G NR FR3 @ 24 GHz — Office / Urban / Highway / Classroom
python 6g_nr_fr3_live_sim.py

# 6G THz Real-Life Use Cases — NS3-aligned
python 6g_thz_live_sim.py

# 6G THz + SINR Radar Charts + Multi-Layer Heatmaps
python 6g_thz_radar_heatmaps.py
```

On first launch, the simulator pre-computes SINR heatmaps for all environments (takes a few seconds). Results are cached for the session.

---

## Controls

### Mouse (GUI Widgets)
| Widget | Action |
|--------|--------|
| Environment radio buttons | Switch active environment/use case |
| Service checkboxes | Filter visible node types by service class |
| Display checkboxes | Toggle heatmap / trails / link lines |
| Speed slider | Adjust simulation speed (0.1× – 6×) |
| ⏸ Pause / ▶ Resume | Pause or resume animation |
| ↺ Reset | Reset nodes and counters for current environment |
| ⬇ Export CSV | Save current SINR/TP/latency snapshot to CSV |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `R` | Reset current environment |
| `E` | Export data to CSV |
| `1` – `4` or `1` – `6` | Switch to environment by index |
| `+` / `=` | Increase simulation speed by 0.5× |
| `-` | Decrease simulation speed by 0.5× |
| `H` | Toggle SINR heatmap |
| `T` | Toggle node trails |
| `L` | Toggle gNB link lines |

---

## Dashboard Panels

### FR3 Simulation (`6g_nr_fr3_live_sim.py`)

| Panel | Description |
|-------|-------------|
| Network Topology | Animated node positions, buildings, gNB pulse rings, heatmap overlay |
| Live SINR Bars | Horizontal bars showing instantaneous SINR per node |
| Per-Node Throughput | Time series of throughput (Mbps) for each active node |
| Live KPIs | Service-class average SINR, throughput, handover counter |
| SINR Time Series | Rolling history for all active nodes |
| SINR vs Distance | Scatter plot of SINR against distance to nearest gNB |
| SINR CDF | Cumulative distribution of SINR per node |
| Aggregate Throughput | Total and per-service throughput over time (Mbps) |

### THz Simulations (both THz scripts, plus extras in radar version)

All FR3 panels plus:

| Panel | Description |
|-------|-------------|
| Latency Timeline | Per-node latency (ms) with target threshold line |
| SINR Contour Heatmap | Filled contourf with ISO-dB labels and colourbar |
| Polar SINR Map | Angular SINR coverage from primary gNB in polar coordinates |
| KPI Radar Chart | 6-axis spider chart: SINR · TP · Latency · Coverage · BF-Gain · Reliability |
| Per-Service Heatmaps | Side-by-side SINR maps for URLLC / eMBB / mMTC / XR / V2X |
| SINR Histogram | Rolling stacked bar histogram with mean/median lines |

---

## CSV Export

Press `E` or click **⬇ Export CSV** to save a snapshot. The output file is saved to the working directory.

### FR3 export format (`sinr_export_<env>_<timestamp>.csv`)
```
Node, Type, Service, SINR_dB, Throughput_Mbps, X, Y
```

### THz export format (`thz_export_<env>_<timestamp>.csv` / `thz_sinr_<env>_<timestamp>.csv`)
```
Node, Type, Service, X_m, Y_m, SINR_dB, Throughput_Gbps, Latency_ms, NS3_Mobility
```

---

## NS3 / C++ Alignment

Each node in the THz scripts carries a `ns3_model` tag mapping directly to an NS3 C++ mobility model:

| Python Behaviour | NS3 C++ Equivalent |
|------------------|--------------------|
| Constant position | `ns3::ConstantPositionMobilityModel` |
| Constant velocity, wrap-around | `ns3::ConstantVelocityMobilityModel` |
| Bounce at boundaries | `ns3::RandomWalk2dMobilityModel` |

Channel / PHY alignment:

| Python Model | NS3 C++ Module |
|--------------|----------------|
| Friis path loss | `ns3::FriisPropagationLossModel` |
| THz absorption | `ns3::ThzSpectrumPropagationLossModel` |
| SINR-based handover | `ns3::LteHandoverAlgorithm` |
| Flow statistics | `ns3::FlowMonitorHelper` |
| Traffic generation | `ns3::UdpClientHelper` |

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Runtime |
| matplotlib | ≥ 3.5 | Rendering, animation, widgets |
| numpy | ≥ 1.21 | Numerical computation |
| scipy | ≥ 1.7 | Gaussian filter for heatmaps |

Install all at once:
```bash
pip install matplotlib numpy scipy
```

---

## SINR Quality Reference

| SINR Range | Quality | Colour |
|------------|---------|--------|
| ≥ 20 dB (FR3) / ≥ 18 dB (THz) | Excellent | 🟢 Green |
| 12–20 dB / 10–18 dB | Good | 🟠 Orange |
| 4–12 dB / 3–10 dB | Fair | 🟠 Dark orange |
| < 4 dB / < 3 dB | Marginal | 🔴 Red |

---

## RF Parameters Reference

| Parameter | FR3 Value | THz Value |
|-----------|-----------|-----------|
| Frequency | 24.0 GHz | 100 GHz – 1 THz |
| Bandwidth | 200–400 MHz | 30–300 GHz |
| Wavelength | 12.5 mm | 0.3–3 mm |
| Numerology | μ=2, 60 kHz SCS | μ=4, 240 kHz SCS |
| gNB TX power | 23–46 dBm | 20–45 dBm |
| UE TX power | 23 dBm | 20–27 dBm |
| Noise floor | −90 dBm | −75 to −88 dBm |
| BF antenna | 32×32 MIMO | 1024–4096 elements |
| Modulation | 256-QAM | 256-QAM |
| Shadowing σ | 1.5 dB | 1.8 dB |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Author

**darpanlan-6g** — [GitHub Profile](https://github.com/darpanlan-6g)

---

<p align="center">
  <em>Built for 6G research, education, and network planning exploration.</em><br/>
  <em>Physics aligned with 3GPP TR 38.901, Jornet & Akyildiz THz channel model, and NS3 simulation framework.</em>
</p>
