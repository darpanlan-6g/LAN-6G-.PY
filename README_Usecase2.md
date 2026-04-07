# 6G NR FR3 @ 24 GHz — Interactive Live Simulation

> **A real-time animated Python simulation of a 3GPP 6G New Radio network operating in the FR3 mid-band at 24 GHz. Visualizes SINR, throughput, mobility, handovers, and network performance across four realistic environments with live interactive controls.**

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [Quick Start](#quick-start)
3. [How the Simulation Works](#how-the-simulation-works)
4. [FR3 Band — Why 24 GHz?](#fr3-band--why-24-ghz)
5. [Physics Engine](#physics-engine)
6. [Four Environments Explained](#four-environments-explained)
7. [Node Types and Services](#node-types-and-services)
8. [Visualization Panels (8 total)](#visualization-panels-8-total)
9. [SimState Class — The Core Engine](#simstate-class--the-core-engine)
10. [Mobility Models](#mobility-models)
11. [Complete Code Walkthrough](#complete-code-walkthrough)
12. [Interactive Controls Reference](#interactive-controls-reference)
13. [CSV Export Format](#csv-export-format)
14. [Tunable Parameters](#tunable-parameters)
15. [How to Add a New Environment](#how-to-add-a-new-environment)
16. [Dependencies and Backend](#dependencies-and-backend)
17. [Limitations](#limitations)
18. [Glossary](#glossary)

---

## What This Is

This is a **single Python file** (~880 lines) that opens an animated matplotlib window simulating a 6G NR (New Radio) wireless network. Every 55 ms it:

1. Moves all nodes (laptops, phones, cars, trucks, emergency vehicles, IoT sensors) according to their velocity
2. Computes the SINR at each node's current position using the Friis path loss model
3. Estimates throughput and detects handovers between base stations
4. Refreshes all 8 visualization panels with fresh live data
5. Responds to keyboard shortcuts and on-screen widget interactions

The four environments — **Office, Urban Streets, Highway, Classroom** — each have different building layouts, transmission powers, bandwidths, and node populations.

---

## Quick Start

### Install

```bash
pip install matplotlib numpy scipy
```

### Run

```bash
python 6g_nr_fr3_live_sim.py
```

On first launch it pre-computes SINR heatmaps for all 4 environments (takes ~5–15 seconds). They are cached for the rest of the session.

---

## How the Simulation Works

```
STARTUP (once at launch)
  │
  ├── Pre-compute 40×40 SINR grid for each environment  → cached
  ├── Build 22×13 inch matplotlib figure with 8 axes
  └── Register keyboard and widget callbacks

                    ↓  FuncAnimation fires every 55 ms
DRAW FRAME  (repeats forever)
  │
  ├── SIM.step(dt=0.05)                ← physics + movement
  │   ├── For every node:
  │   │   ├── new_x = x + vx * dt
  │   │   ├── new_y = y + vy * dt
  │   │   ├── apply boundary rule (bounce OR wrap-around)
  │   │   ├── SINR  = friis_pl + interference + noise model
  │   │   ├── Throughput = Shannon(SINR, bandwidth) × 0.6
  │   │   ├── append SINR and TP to rolling 300-sample history
  │   │   ├── append position to 80-sample trail
  │   │   └── if nearest gNB changed → handovers++
  │   └── accumulate total system throughput
  │
  └── Redraw 8 panels with new data
```

---

## FR3 Band — Why 24 GHz?

The 3GPP specification defines three frequency ranges for 6G NR:

| Range | Frequencies | Wavelength | Coverage | Typical use |
|-------|-------------|-----------|----------|-------------|
| **FR1** | Sub-6 GHz | 5–50 cm | Wide area | Macro cells |
| **FR2** | 24–52 GHz (mmWave) | 6–12 mm | Short range | Hotspots |
| **FR3** | 7–24 GHz | 12–43 mm | Mid range | **This simulation** |

**24 GHz sits at the top of FR3** and provides a practical compromise:

- **Better propagation than mmWave** — lower path loss than 28/39 GHz
- **~25% lower building penetration loss** compared to FR2 mmWave
- **200–400 MHz bandwidth** — more than FR1, less than full mmWave
- **Lower atmospheric absorption** — 24 GHz avoids the 60 GHz oxygen absorption peak
- Suitable for V2X, dense urban deployments, and indoor small cells

In the simulation, 24 GHz is fixed as the carrier. The bandwidth varies by environment: 200 MHz indoors, 400 MHz on the highway.

---

## Physics Engine

### Friis Path Loss

```python
def friis_pl(d_m, freq_ghz):
    return 20*log10(max(d_m, 0.5)) + 20*log10(freq_ghz*1e9) - 147.55
```

This is the standard **free-space path loss formula** in dB:

```
PL = 20·log10(d) + 20·log10(f) − 147.55
```

The constant `−147.55` comes from `20·log10(4π/c)` where `c = 3×10⁸ m/s`.

**At 24 GHz, typical path loss values:**

| Distance | Path Loss |
|----------|-----------|
| 1 m | 60 dB |
| 10 m | 80 dB |
| 50 m | 94 dB |
| 100 m | 100 dB |
| 500 m | 114 dB |

This is why the highway uses 46 dBm TX power while the office uses only 23 dBm — the highway must cover 200 m, the office only 100 m.

### SINR Computation

```python
def compute_sinr(nx, ny, gnbs, cfg, rng):
    for each gNB:
        d        = sqrt((nx−gx)² + (ny−gy)²)        # distance in metres
        PL       = friis_pl(d, 24.0)                  # free-space path loss
        Rx_power = Tx_power − PL − pen_loss×0.25      # received power
                   + Gaussian_noise(σ=1.5 dB)         # log-normal shadowing

    # Sort gNBs by received power (strongest first)
    signal      = 10^(strongest_Rx / 10)              # convert dBm → mW
    interference = Σ 10^(other_Rx / 10)               # sum all other gNBs
    noise        = 10^(−90 / 10)                      # thermal noise at −90 dBm

    SINR = signal / (interference + noise)            # linear ratio
    return 10·log10(SINR)                              # convert back to dB
```

**Key design choices:**
- `pen_loss × 0.25` — applies 25% of the wall/material penetration loss as diffuse indoor attenuation, not full blockage
- `σ = 1.5 dB` Gaussian noise — models log-normal shadowing from unmodelled obstacles
- Thermal noise fixed at `−90 dBm` — typical for a 200 MHz receiver at room temperature
- Strongest gNB is the signal; all others become interference — this is the co-channel interference model

### Shannon Throughput

```python
def shannon_tp(sinr_db, bw_mhz):
    return bw_mhz * log2(1 + 10^(sinr_db/10)) * 0.6   # Mbps
```

Shannon's theorem gives the theoretical capacity of a channel. The factor `0.6` is a **practical efficiency** representing:
- Coding and modulation overhead (~15%)
- Control channel overhead (~8%)
- Hardware imperfections and guard bands (~17%)

**Example throughputs at 200 MHz bandwidth:**

| SINR | Throughput |
|------|-----------|
| 5 dB | ~410 Mbps |
| 10 dB | ~720 Mbps |
| 15 dB | ~1,000 Mbps |
| 20 dB | ~1,300 Mbps |
| 25 dB | ~1,600 Mbps |

### SINR Quality Color Mapping

```python
def sinr_color_map(v):
    if v >= 20: return GREEN    # Excellent — high throughput
    if v >= 12: return ORANGE   # Good — usable
    if v >= 4:  return "#E67E22" # Fair — degraded
    return RED                  # Poor — near outage
```

This same function drives the color of node dots, trail lines, bar charts, and link lines across all panels.

---

## Four Environments Explained

### 1. Office (100 × 80 m)

| Parameter | Value |
|-----------|-------|
| Frequency | 24 GHz |
| Bandwidth | 200 MHz |
| TX Power | 23 dBm (200 mW — access point level) |
| Penetration loss | 20 dB (concrete/glass office) |
| Base stations | 1 indoor AP |
| Nodes | 2 laptops (stationary), 3 phones (moving), 3 IoT sensors |
| Services | eMBB (laptops/phones), mMTC (IoT) |

The office has one central access point covering a 100×80 m floorplan with four rooms of different materials. Phones bounce around; laptops and sensors are stationary.

### 2. Urban Streets (120 × 100 m)

| Parameter | Value |
|-----------|-------|
| Frequency | 24 GHz |
| Bandwidth | 200 MHz |
| TX Power | 40 dBm (10 W — street macro cell level) |
| Penetration loss | 18 dB (urban mix of concrete/glass/metal) |
| Base stations | 2 street-level gNBs |
| Nodes | 3 cars, 1 truck, 2 pedestrians, 2 RSUs |
| Services | URLLC (cars/RSUs), eMBB (truck/pedestrians) |

Cars move at ~2 m/s (7 km/h — urban crawl). Pedestrians bounce around the street area. Road Side Units (RSUs) are fixed infrastructure nodes.

### 3. Highway (200 × 60 m)

| Parameter | Value |
|-----------|-------|
| Frequency | 24 GHz |
| Bandwidth | 400 MHz (doubled for high-speed V2X) |
| TX Power | 46 dBm (40 W — roadside macro level) |
| Penetration loss | 8 dB (mostly open road, some barriers) |
| Base stations | 2 highway-side gNBs |
| Nodes | 5 cars + 2 trucks + 1 emergency vehicle |
| Services | URLLC (all vehicles — safety-critical V2X) |

Vehicles move at 2.5–4.8 m/s (9–17 km/h in simulation scale — scaled down from real highway speeds of 90–130 km/h for visual clarity). They wrap around the left/right edges to simulate continuous highway travel. Handovers between the two gNBs happen frequently as vehicles pass the midpoint.

### 4. Classroom (90 × 70 m)

| Parameter | Value |
|-----------|-------|
| Frequency | 24 GHz |
| Bandwidth | 200 MHz |
| TX Power | 23 dBm |
| Penetration loss | 20 dB |
| Base stations | 1 ceiling AP |
| Nodes | 6 student phones, 2 teacher laptops, 1 smart projector |
| Services | eMBB (all — high-bandwidth holographic education) |

The classroom environment mirrors the XR/holo education use case — all devices need consistent high throughput from a single ceiling-mounted access point.

---

## Node Types and Services

### Node tuple format

Each node in the `"nodes"` list is a tuple:

```python
(id, type, x0, y0, vx, vy, service, bounce)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier displayed on the plot (`"L1"`, `"Car1"`, etc.) |
| `type` | str | Visual style key — must be in `NODE_STYLE` |
| `x0`, `y0` | float | Starting position in metres |
| `vx`, `vy` | float | Velocity in metres per simulation second |
| `service` | str | 3GPP service class: `URLLC`, `eMBB`, or `mMTC` |
| `bounce` | bool | `True` = wall reflection; `False` = edge wrap-around |

### Node visual styles

| Type | Color | Marker | Typical use |
|------|-------|--------|-------------|
| `laptop` | Green | Square | Fixed workstation |
| `phone` | Purple | Circle | Mobile user |
| `car` | Blue | Triangle up | Passenger vehicle |
| `truck` | Orange | Diamond | Heavy vehicle |
| `emergency` | Red | Star | Emergency service |
| `pedestrian` | Pink | Plus circle | Person on foot |
| `rsu` | Yellow | Hexagon | Road Side Unit |
| `iot` | Teal | Cross | IoT sensor/device |

### Service types

| Service | Color | Meaning | Typical latency / throughput target |
|---------|-------|---------|-------------------------------------|
| **URLLC** | Red | Ultra-Reliable Low-Latency Communication | < 1 ms, 99.9999% reliability |
| **eMBB** | Blue | Enhanced Mobile Broadband | > 1 Gbps peak |
| **mMTC** | Green | Massive Machine-Type Communication | Low power, low rate, huge density |

---

## Visualization Panels (8 total)

The 22×13 inch figure uses a `GridSpec` layout with the main simulation area in the top 63% and the control strip in the bottom 27%.

### Main area panels

#### Panel 1 — Network Topology + Heatmap (large, left, spans 2 rows)
The primary view. Shows:
- **Pre-computed SINR heatmap** as a `pcolormesh` background — green = strong signal, orange/red = weak
- **Buildings** as rounded rectangles, colored by material (dark grey = concrete, dark blue = glass, dark brown = metal)
- **gNB towers** as upward triangles with three translucent coverage rings of decreasing opacity and an animated expanding pulse ring
- **Node markers** with their type-specific symbol and color
- **SINR quality dot** — small colored dot in the top-right of each node marker showing current signal quality
- **Link lines** from each node to its nearest gNB, colored by SINR quality
- **Motion trails** — last 80 positions connected as a `LineCollection`, colored by SINR at each point

#### Panel 2 — Live SINR Bars (top right, row 0 col 1)
Horizontal bar chart showing each node's current SINR in dB. Bars are colored green/orange/red by quality. Two vertical dashed reference lines at 20 dB (excellent) and 12 dB (minimum usable). Node IDs on the left in their service color.

#### Panel 3 — Throughput Timeline (top right, row 0 col 2)
Per-node throughput in Mbps over the last ~15 seconds. Each node gets a distinct color from the `tab20` colormap with a legend.

#### Panel 4 — Live KPI Stats (top right, row 0 col 3)
Text scorecard showing for each active service type:
- Average SINR in dB
- Average throughput in Mbps
- Whether the service KPI is being met (✓ or ✗)
- Running totals: handover count, elapsed simulation time, environment name

#### Panel 5 — SINR Time Series (row 1, col 1)
Rolling SINR history for all active nodes. Background shading: green band above 20 dB, orange band 12–20 dB, red band below 12 dB. Horizontal dashed reference lines at 20 and 12 dB.

#### Panel 6 — SINR vs Distance Scatter (row 1, col 2)
Live scatter plot of current SINR vs current distance to nearest gNB for every active node. Clearly shows the path loss relationship — nodes closer to the gNB cluster in the upper left (high SINR, short distance).

#### Panel 7 — SINR CDF (row 1, col 3)
Cumulative Distribution Function of SINR history. X-axis: SINR in dB. Y-axis: probability that SINR ≤ x. A CDF that stays right of the 12 dB line indicates reliable coverage.

#### Panel 8 — Aggregate Throughput (full-width bottom row)
Total system throughput in Mbps (solid colored line) with a shaded fill. Dashed overlays for each service type (URLLC/eMBB/mMTC). Handover count shown in the title.

---

## SimState Class — The Core Engine

All simulation state lives in a single `SimState` instance called `SIM`.

```
SIM (SimState instance)
│
├── env_name          ← which of 4 environments is active
├── paused            ← animation pause flag
├── speed             ← time multiplier (0.1× – 5×)
├── t                 ← simulation clock in seconds
├── frame             ← integer frame counter
│
├── positions[nid]    ← [x, y] — updated every step
├── velocities[nid]   ← [vx, vy] — may flip sign on bounce
├── bounce[nid]       ← True = wall reflection, False = wrap
│
├── sinr_hist[nid]    ← rolling list, last 300 SINR values [dB]
├── tp_hist[nid]      ← rolling list, last 300 throughput values [Mbps]
│
├── trail_x[nid]      ← last 80 x-positions for trail drawing
├── trail_y[nid]      ← last 80 y-positions
│
├── prev_cell[nid]    ← index of last-connected gNB (for HO detection)
├── handovers         ← total handover event count
│
├── total_tp          ← list of last 300 total-system TP values
└── heatmaps          ← {env_name: (XX, YY, grid)} — computed once, cached
```

### step() method — line by line

```python
def step(self, dt=0.05):
    if self.paused: return          # do nothing if paused

    self.t += dt * self.speed       # advance simulation clock
    self.frame += 1

    total_tp = 0.0
    for node in cfg["nodes"]:
        # 1. Move the node
        x += vx * dt
        y += vy * dt

        # 2. Apply boundary condition
        if bounce:
            if hit_wall: flip velocity, clamp position
        else:
            if past_right_edge: teleport to left edge  ← highway wrap
            if past_left_edge:  teleport to right edge

        # 3. Compute signal quality
        sv = compute_sinr(x, y, gnbs, cfg, rng)    # dB
        tp = shannon_tp(sv, bw_mhz)                # Mbps

        # 4. Update rolling histories (trim if >300)
        sinr_hist.append(sv)
        tp_hist.append(tp)

        # 5. Update motion trail (trim if >80)
        trail_x.append(x); trail_y.append(y)

        # 6. Detect handover
        nearest_gnb = argmin(distance to each gNB)
        if nearest_gnb != prev_cell: handovers++
        prev_cell = nearest_gnb

        total_tp += tp

    # 7. Record system-level throughput
    total_tp_list.append(total_tp)
```

---

## Mobility Models

### Bounce (wall reflection)

```python
if x <= 0 or x >= W:
    vx *= -1               # reverse horizontal velocity
    x = clip(x, 0.5, W-0.5)  # push back inside boundary
```

Used for: phones, pedestrians — nodes that roam within a bounded space.

### Wrap-around (edge teleport)

```python
if vx > 0 and x > W + 5:  x = -5   # re-enter from left
if vx < 0 and x < -5:     x = W + 5 # re-enter from right
```

Used for: vehicles on the highway — they exit one side and immediately re-enter the other, simulating continuous traffic flow.

### Stationary

Nodes with `vx=0, vy=0` never move. Used for: access points, RSUs, laptops, IoT sensors.

---

## Complete Code Walkthrough

```
6g_nr_fr3_live_sim.py
│
├── ─── COLOUR CONSTANTS ─────────────────────────────────────────────
│   BG, PANEL_BG, GRID_COL, TEXT_COL, MUTED_COL
│   Accent colours: ACC_BLUE, ACC_GREEN, ACC_RED, ACC_ORG, ACC_PRP, ACC_TEAL
│   SERVICE_COLOR: maps URLLC/eMBB/mMTC → color hex
│   NODE_STYLE: maps node type → {color, marker, size, label}
│   MAT_COLOR / MAT_EDGE: maps building material → face/edge color
│
├── ─── ENVIRONMENTS dict (lines ~64–155) ────────────────────────────
│   "Office"        → 100×80 m, 1 AP, 8 nodes
│   "Urban Streets" → 120×100 m, 2 gNBs, 8 nodes
│   "Highway"       → 200×60 m, 2 gNBs, 8 nodes, 400 MHz BW
│   "Classroom"     → 90×70 m, 1 AP, 9 nodes
│
├── ─── PHYSICS FUNCTIONS (lines ~156–192) ───────────────────────────
│   friis_pl(d, f)           Friis free-space path loss in dB
│   compute_sinr(...)        Full SINR: signal / (interference + noise)
│   shannon_tp(sinr, bw)     Shannon capacity × 0.6 efficiency
│   sinr_color_map(v)        dB value → hex color string
│   build_heatmap(cfg)       40×40 SINR grid + Gaussian smoothing
│
├── ─── SimState class (lines ~194–312) ──────────────────────────────
│   __init__()    defaults, call reset()
│   cfg           property: returns ENVIRONMENTS[env_name]
│   reset()       re-init positions/velocities/histories; build heatmap
│   step(dt)      one physics + movement tick (called per frame)
│   export_csv()  snapshot current state to CSV file
│
├── ─── FIGURE SETUP (lines ~314–372) ────────────────────────────────
│   plt.rcParams  dark theme settings
│   GridSpec 3×4  top area with 8 axes
│   GridSpec 1×7  bottom control strip
│   Title text
│
├── ─── WIDGETS (lines ~374–530) ──────────────────────────────────────
│   radio_env    RadioButtons  — 4 environment options
│   chk_svc      CheckButtons  — URLLC / eMBB / mMTC filters
│   chk_opts     CheckButtons  — Heatmap / Trails / Links toggles
│   sl_spd       Slider        — simulation speed
│   btn_pause    Button        — pause/resume
│   btn_reset    Button        — reset
│   btn_export   Button        — export CSV
│   ax_legend    Static legend — node type reference
│   ax_info      Static panel  — RF parameter summary
│
├── ─── draw_frame() (lines ~532–820) ────────────────────────────────
│   Called every 55 ms by FuncAnimation.
│   Calls SIM.step(), then redraws all 8 panels.
│   Each panel follows the same pattern:
│     ax.cla()          ← clear old content
│     ax.set_facecolor  ← dark theme
│     ax.set_title/xlabel/ylabel
│     [draw content]
│
├── ─── on_key() (lines ~822–840) ─────────────────────────────────────
│   Keyboard event handler.
│
└── ─── STARTUP (lines ~842–877) ──────────────────────────────────────
    Pre-compute all heatmaps.
    Launch FuncAnimation.
    plt.show().
```

---

## Interactive Controls Reference

### Keyboard

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume animation |
| `R` | Reset — restart all node positions |
| `E` | Export snapshot to CSV |
| `1` | Switch to Office environment |
| `2` | Switch to Urban Streets environment |
| `3` | Switch to Highway environment |
| `4` | Switch to Classroom environment |
| `+` or `=` | Increase simulation speed by 0.5× |
| `-` | Decrease simulation speed by 0.5× |
| `H` | Toggle SINR heatmap background |
| `T` | Toggle motion trails |
| `L` | Toggle link lines to gNB |

### On-Screen Widgets

| Widget | Type | Function |
|--------|------|----------|
| **Environment** | Radio buttons | Instantly switches environment and resets |
| **Service Filter** | Checkboxes | Show/hide URLLC / eMBB / mMTC nodes |
| **Display** | Checkboxes | Toggle Heatmap / Trails / Links |
| **Speed** | Slider | 0.1× (slow-motion) to 5× (fast-forward) |
| **Pause** | Button | Same as SPACE |
| **Reset** | Button | Same as R |
| **Export CSV** | Button | Same as E |

---

## CSV Export Format

Pressing `E` or the Export CSV button saves a file:
```
sinr_export_<EnvironmentName>_<timestamp>.csv
```

Columns:
```csv
Node, Type, Service, SINR_dB, Throughput_Mbps, X, Y
```

Example:
```csv
Node,Type,Service,SINR_dB,Throughput_Mbps,X,Y
L1,laptop,eMBB,18.42,892.1,12.0,18.0
P1,phone,URLLC,14.71,710.3,22.4,57.8
IoT1,iot,mMTC,11.55,548.2,30.0,60.0
Car1,car,URLLC,16.33,810.5,45.2,50.0
```

This CSV captures a single instant. Run multiple exports at different simulation times to build a dataset for analysis.

---

## Tunable Parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `res=40` in `build_heatmap()` | Physics | 40 | Heatmap resolution — 40×40 = 1600 SINR evaluations. Increase to 60–80 for sharper maps; slower startup. |
| `interval=55` in `FuncAnimation` | Rendering | 55 ms | Frame period. Lower = smoother animation; needs faster CPU. |
| `dt=0.05` in `step()` | Physics | 0.05 s | Simulation time per frame. Multiplied by `speed` slider. |
| `300` in history trimming | SimState | 300 | Rolling window length in samples (~15 s at dt=0.05). |
| `80` in trail trimming | SimState | 80 | Trail length in positions. |
| `sigma=1.2` in `gaussian_filter` | Heatmap | 1.2 | Smoothing radius for heatmap. Higher = more blurred. |
| `σ=1.5` in `rng.normal(0, 1.5)` | Physics | 1.5 dB | Shadowing standard deviation. |
| `−90 dBm` noise floor | Physics | −90 | Thermal noise — change to −85 for worse sensitivity, −95 for better. |
| `0.6` in `shannon_tp` | Physics | 0.6 | Practical efficiency factor (60% of Shannon limit). |
| `0.25` in `pen * 0.25` | Physics | 25% | Fraction of penetration loss applied as diffuse attenuation. |

---

## How to Add a New Environment

1. Add an entry to the `ENVIRONMENTS` dict:

```python
"My Campus": {
    "area": (150, 120),         # metres W × H
    "freq_ghz": 24.0,
    "bw_mhz": 200,
    "tx_power_dbm": 30,
    "pen_loss_avg": 15,
    "color": "#9B59B6",         # accent color for this environment
    "buildings": [
        (10, 10, 50, 40, "Library", "concrete", 22),
        (80, 10, 40, 40, "Labs",    "glass",      8),
    ],
    "gnbs": [
        (75, 60, "gNB-Camp", 12),
    ],
    "nodes": [
        ("Stu1", "phone",  20, 50,  0.3,  0.1, "eMBB",  True),
        ("Stu2", "phone",  90, 50, -0.2,  0.2, "eMBB",  True),
        ("AP1",  "iot",    75, 60,  0.0,  0.0, "mMTC",  False),
    ],
}
```

2. The environment will automatically appear in the RadioButtons widget — no other code changes needed.

3. To add a new node type, add an entry to `NODE_STYLE`:

```python
"bicycle": {"color": "#F39C12", "marker": "p", "size": 85, "label": "Bicycle"},
```

---

## Dependencies and Backend

| Package | Version | Purpose |
|---------|---------|---------|
| `matplotlib` | ≥ 3.7 | All rendering, animation, widgets |
| `numpy` | ≥ 1.24 | Array maths, SINR computation |
| `scipy` | ≥ 1.10 | `gaussian_filter` for heatmap smoothing |

```bash
pip install matplotlib numpy scipy
```

### Changing the backend

The first line of the simulation sets the GUI backend:

```python
matplotlib.use("TkAgg")   # top of file
```

| System | Recommended backend |
|--------|-------------------|
| Linux with display | `TkAgg` (default) or `Qt5Agg` |
| Linux headless | `Agg` (saves frames to file, no window) |
| macOS | `MacOSX` or `Qt5Agg` |
| Windows | `TkAgg` (default) |
| JupyterLab | Remove `matplotlib.use()` and add `%matplotlib widget` |

---

## Limitations

This is a **simplified simulation model**, not a full 3GPP protocol stack. Specifically:

- **No multipath fading** — no Rayleigh/Rician small-scale fading beyond the Gaussian shadowing term
- **No beamforming steering** — the beamforming gain is a fixed offset, not a directional model
- **No HARQ** — no retransmission model; latency is not tracked in this version
- **Free-space only** — no ray-tracing, no diffraction, no realistic NLOS model beyond the penetration loss constant
- **No PHY/MAC layers** — no slot structure, scheduling, or modulation order selection
- **No 3GPP NR numerology** — bandwidth is treated as a continuous Shannon channel, not subcarrier-structured
- Throughput is in **Mbps**, not mapped to real NR resource blocks or MCS tables
- Node velocities are **scaled down** for visual clarity — highway vehicles move at ~15 km/h in simulation units, not 100+ km/h

---

## Glossary

| Term | Definition |
|------|------------|
| **FR3** | Frequency Range 3 — 3GPP designation for 6G NR bands in the 7–24 GHz range |
| **gNB** | Next-Generation Node B — the 6G/5G base station |
| **SINR** | Signal-to-Interference-plus-Noise Ratio — the primary measure of wireless link quality, in dB |
| **Friis PL** | Free-space path loss formula: `PL = 20·log(d) + 20·log(f) − 147.55` |
| **URLLC** | Ultra-Reliable Low-Latency Communication — used for safety-critical V2X and industrial control |
| **eMBB** | Enhanced Mobile Broadband — used for high-throughput video, AR/VR |
| **mMTC** | Massive Machine-Type Communication — used for dense IoT sensors |
| **V2X** | Vehicle-to-Everything — communication between vehicles, infrastructure, and pedestrians |
| **Handover** | When a mobile node switches its connection to a nearer/stronger gNB |
| **Shadowing** | Random variation in signal strength caused by obstacles not explicitly modelled |
| **Shannon capacity** | Theoretical maximum data rate: `C = BW × log₂(1 + SINR)` |
| **CDF** | Cumulative Distribution Function — shows what fraction of time a quantity is below a threshold |
| **RSU** | Road Side Unit — fixed infrastructure node for V2X communication |
| **Penetration loss** | Extra signal loss when a radio wave passes through a wall or building material |
| **FuncAnimation** | Matplotlib class that calls a function repeatedly to create animation |
| **GridSpec** | Matplotlib layout manager that divides a figure into a grid of axes |
| **LineCollection** | Matplotlib object that draws many line segments efficiently — used for trails |
| **pcolormesh** | Matplotlib function for rendering a color-filled grid — used for the SINR heatmap |
| **Gaussian smoothing** | Blurring filter applied to the heatmap grid to remove sharp discontinuities |
| **Log-normal shadowing** | Shadowing modelled as a Gaussian distribution in dB (linear-scale log-normal) |

---

## File Summary

```
6g_nr_fr3_live_sim.py               ← Complete simulation (single file, ~880 lines)
README_6g_nr_fr3_live_sim.md        ← This document
sinr_export_<env>_<ts>.csv          ← Created when you press E (optional output)
```

---

*Simulation models aligned with 3GPP TR 38.101-2 (FR2/FR3 specifications) and TR 22.261 (6G service requirements).*
*Friis formula per ITU-R P.525. Shadowing model per 3GPP TR 38.901.*
*Built with Python · matplotlib · NumPy · SciPy*
