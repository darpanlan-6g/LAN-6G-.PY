# 6G THz Digital Twin Simulator

> **A real-time interactive Python simulation that runs a physical 6G THz wireless network and its digital twin replica side-by-side, with live synchronisation, fidelity scoring, anomaly detection, and prediction error tracking across six real-world use cases.**

---

## Table of Contents

1. [What Is a Digital Twin?](#what-is-a-digital-twin)
2. [What This Simulator Does](#what-this-simulator-does)
3. [Quick Start](#quick-start)
4. [How the Dual-World Simulation Works](#how-the-dual-world-simulation-works)
5. [Physics Engine](#physics-engine)
6. [Six Use Cases](#six-use-cases)
7. [Digital Twin Metrics Explained](#digital-twin-metrics-explained)
8. [All Visualization Panels (12 total)](#all-visualization-panels-12-total)
9. [DigitalTwin Class — Complete Internals](#digitaltwin-class--complete-internals)
10. [Anomaly Injection System](#anomaly-injection-system)
11. [Complete Code Walkthrough](#complete-code-walkthrough)
12. [Interactive Controls](#interactive-controls)
13. [Colour Coding Reference](#colour-coding-reference)
14. [Key Parameters to Tune](#key-parameters-to-tune)
15. [How to Add a New Use Case](#how-to-add-a-new-use-case)
16. [Dependencies](#dependencies)
17. [Limitations](#limitations)
18. [Glossary](#glossary)

---

## What Is a Digital Twin?

A **Digital Twin** is a continuously updated virtual model of a real physical system. The twin receives live sensor data from the physical world, maintains an internal state that mirrors reality, and can be used to:

- **Monitor** the real system without direct physical access
- **Predict** future states based on current trends
- **Detect anomalies** when the twin's model diverges from reality
- **Simulate what-if scenarios** by manipulating only the virtual copy
- **Optimise** the physical system by running experiments on the twin first

In the context of 6G wireless networks, a Digital Twin of the radio access network enables operators to predict coverage failures, detect interference events, and simulate handover strategies without affecting live traffic.

```
┌─────────────────────┐         ┌─────────────────────┐
│   PHYSICAL WORLD    │         │   DIGITAL TWIN       │
│                     │         │                      │
│  Real 6G nodes      │──sync──►│  Virtual replicas    │
│  Real SINR          │  (ms)   │  Predicted SINR      │
│  Real positions     │         │  Estimated positions │
│  Real throughput    │         │  Modelled throughput │
│                     │◄─alert──│  Anomaly detection   │
└─────────────────────┘         └─────────────────────┘
```

**Key principle:** The twin is never perfectly identical to reality — it has its own noise, its own update latency, and can drift. The simulation makes all of this visible and measurable.

---

## What This Simulator Does

This is a **single Python file** (~680 lines) that opens an animated window with two live maps side by side — the physical world and its digital twin — plus 10 additional analysis panels. Every 60 ms it:

1. Moves all real nodes (robots, cars, drones, etc.) according to physics
2. Computes real SINR using THz propagation models
3. Updates the digital twin with a slight lag and independent measurement noise
4. Scores twin fidelity per node
5. Detects position and SINR divergence
6. Refreshes all 12 panels with the latest comparison data

The highlight feature is **anomaly injection** (`A` key / ⚡ button): one node's twin deliberately drifts out of sync with reality, and you can watch the anomaly detector fire, the fidelity score drop, and the heatmap difference panel light up in red.

---

## Quick Start

### Install

```bash
pip install matplotlib numpy scipy
```

### Run

```bash
python 6g_thz_digital_twin.py
```

On first launch, heatmaps are pre-computed for all 6 environments (~15–30 s). Cached in memory for the session.

---

## How the Dual-World Simulation Works

```
STARTUP (once)
  Pre-compute SINR heatmaps for all 6 environments
  Create figure with 12 panels + control strip
  Register keyboard / button callbacks

               ↓  FuncAnimation fires every 60 ms
DRAW FRAME (forever)

  DT.step(dt=0.04)
  │
  ├── For every node:
  │   │
  │   ├── PHYSICAL WORLD
  │   │   ├── x += vx * dt  (real movement)
  │   │   ├── apply boundary (bounce / wrap)
  │   │   ├── SINR  = calc_sinr(real_position, rng_PHYSICAL)
  │   │   ├── TP    = thz_tp(SINR, BW)
  │   │   ├── lat   = lat_ms(SINR, cfg)
  │   │   ├── update rolling history (300 samples)
  │   │   ├── update trail (70 positions)
  │   │   └── detect handover
  │   │
  │   └── DIGITAL TWIN
  │       ├── if ANOMALY ACTIVE for this node:
  │       │     twin drifts at 1.8× real speed for 60 frames
  │       │     anomaly_flag = True
  │       ├── else (normal sync):
  │       │     twin_pos += (real_pos - twin_pos) × 0.12   ← gradual catch-up
  │       │     + rng_TWIN.normal(0, 0.05)                 ← independent noise
  │       ├── twin SINR = calc_sinr(twin_position, rng_TWIN, noise_offset)
  │       ├── fidelity = 100 - position_error / diagonal × 800
  │       │             - SINR_error × 0.5
  │       ├── fidelity = 0.92 × prev_fidelity + 0.08 × new_fidelity  (EMA)
  │       ├── prediction_error = EMA(hypot(real_pos - twin_pos))
  │       └── sync_latency_ms = EMA(pos_error / speed × 1000)
  │
  └── Redraw all 12 panels

```

### Two independent random number generators

This is the most important design detail. The physical world uses `rng = np.random.default_rng(42)` and the twin uses `rng_t = np.random.default_rng(99)`. They produce different noise sequences, which is why the twin's SINR never perfectly matches reality even when positions are close — exactly as in real deployments where the network management system's model of channel conditions differs from what the UE actually measures.

---

## Physics Engine

### THz Molecular Absorption

```python
_KF = {0.060e12: 0.011,  # 60 GHz  — oxygen band
       0.100e12: 0.003,  # 100 GHz — window
       0.140e12: 0.003,  # 140 GHz — preferred industrial
       0.183e12: 0.174,  # 183 GHz — H₂O absorption line
       0.220e12: 0.004,  # 220 GHz — window
       0.300e12: 0.013,  # 300 GHz — indoor preferred
       0.340e12: 0.069,  # 340 GHz — H₂O absorption
       0.500e12: 0.035,  # 500 GHz
       1.000e12: 0.868}  # 1 THz   — higher absorption
```

The `mol_abs(d, f)` function log-linearly interpolates `k(f)` between the nearest two tabulated points and returns `k × d` in dB — direct attenuation proportional to distance. Source: Jornet & Akyildiz (2011), aligned with ITU-R P.676.

### Free-Space Path Loss (Friis)

```python
def friis(d, f):
    return 20 * log10(4 * pi * d * f / c)    # dB
```

### Full SINR Computation

```python
for each gNB:
    path_loss = friis(d, f) + mol_abs(d, f) + pen × 0.2
    doppler   = 10·log10(speed × f / (c × 240kHz))   # Doppler phase noise
    nlos      = building_loss × 0.5  if inside building
    K         = Rician K-factor (per environment)
    ν = √(K/(K+1)),  σ = 1/√(2(K+1))                # Rician parameters
    fade = 20·log10(√((ν+σ·N₁)² + (σ·N₂)²))         # Rician fading [dB]
    Rx_power = Tx_power - path_loss + BF_gain + fade - doppler - nlos

SINR = strongest_Rx / (sum_interference + thermal_noise)
```

### Shannon Capacity → Throughput

```python
def thz_tp(sv, bw, env_type):
    eta = {indoor:0.62, outdoor:0.58, vehicular:0.52, tunnel:0.48, p2p:0.70}
    capacity = bw × min(log₂(1 + 10^(sv/10)) × eta, 12.0)   # Gbps
    # 12.0 bit/s/Hz ceiling = 4096-QAM modulation limit
```

### Latency Model

```python
def lat_ms(sv, cfg):
    harq_rounds = 0 if sv≥20 else 1 if sv≥12 else 2 if sv≥5 else 3
    return max(0.01, lat_target × 0.4 × exp(-sv/18) + harq_rounds × 0.125 + 0.003)
```

---

## Six Use Cases

| # | Environment | Freq | BW | Peak TP | Latency | K-factor | Key nodes |
|---|-------------|------|----|---------|---------|---------|-----------|
| 1 | **XR Surgery** | 300 GHz | 100 GHz | 800 Gbps | 0.8 ms | 8.0 | Surgeons, robots, cameras, holo-display |
| 2 | **Auto Factory** | 140 GHz | 50 GHz | 200 Gbps | 1.5 ms | 3.0 | AGVs, robot arms, sensors |
| 3 | **V2X Crossroad** | 300 GHz | 80 GHz | 400 Gbps | 0.3 ms | 2.0 | Cars, drones, cameras, RSUs |
| 4 | **THz Backhaul** | 1 THz | 300 GHz | 1800 Gbps | 0.1 ms | 20.0 | Rooftop P2P, relay, drones |
| 5 | **Tunnel Rescue** | 100 GHz | 30 GHz | 80 Gbps | 3.0 ms | 1.5 | Rescuers, drones, sensors |
| 6 | **Holo Classroom** | 300 GHz | 60 GHz | 300 Gbps | 2.0 ms | 6.0 | Students, holo-displays |

The **Rician K-factor** is especially important for the Digital Twin: a high K (like 20.0 for P2P backhaul) means very stable signal → twin stays tightly synced. A low K (like 1.5 for tunnel) means heavy fading variation → twin can drift more.

### Node tuple format

```python
(id, type, x0, y0, vx, vy, service, bounce, speed_mps)
```

The 9th field `speed_mps` is used to compute the Doppler penalty and sync latency: faster-moving nodes are harder for the twin to track.

---

## Digital Twin Metrics Explained

### Fidelity (0–100%)

Measures how accurately the digital twin reflects the physical asset. Computed per node every step:

```python
pos_err  = hypot(real_x - twin_x, real_y - twin_y)
diag     = hypot(W, H)                            # area diagonal
raw_fid  = max(0, 100 - (pos_err / diag) × 800)  # position component
raw_fid -= abs(real_sinr - twin_sinr) × 0.5       # SINR divergence penalty
fidelity = 0.92 × prev_fidelity + 0.08 × raw_fid # exponential moving average
```

The EMA factor (0.92/0.08) smooths rapid fluctuations — fidelity rises and falls gradually rather than jumping each frame.

| Fidelity | Status | Color |
|----------|--------|-------|
| ≥ 90% | SYNCED — twin accurately mirrors reality | Green |
| 70–90% | DRIFT — acceptable but degrading | Amber |
| < 70% | DESYNC — twin has diverged significantly | Red |

### Sync Latency (ms)

How long it takes the digital twin's position to reflect the real node's current position, estimated from the position gap:

```python
sync_ms = position_error / speed × 1000   # if moving
        = lat_target × 0.2                # if stationary
sync_latency = 0.9 × prev_sync + 0.1 × sync_ms  # EMA
```

Stationary nodes (robots, cameras, sensors) have near-zero sync latency because their position never changes.

### Prediction Error (metres)

The absolute distance between the twin's estimated position and the real position:

```python
prediction_err = EMA(hypot(real_x - twin_x, real_y - twin_y))
```

| Error | Status |
|-------|--------|
| < 0.5 m | Excellent — twin knows exactly where the node is |
| 0.5–1.5 m | Acceptable |
| > 1.5 m | Poor — twin has significantly wrong position |

### Sync Factor (0.12)

The fundamental parameter controlling how fast the twin catches up to reality:

```python
twin_pos += (real_pos - twin_pos) × 0.12   # each frame
```

This means the twin closes 12% of the gap per frame (~60 ms), giving a time constant of roughly `0.06 / 0.12 = 0.5 seconds`. Increase this constant for a faster, more responsive twin; decrease it to simulate a slow or intermittently connected twin.

---

## All Visualization Panels (12 total)

### Row 0 — Dual World Maps + KPI Panel

#### Panel 1 — Physical World Map
The real 6G network. Dark background (`#0D1520`). Shows:
- SINR heatmap (pre-computed, static background)
- Buildings colored by material (concrete/glass/metal/wall)
- gNB towers with three translucent coverage rings + animated expanding pulse ring
- Nodes with their type-specific marker and color
- Solid link lines to nearest gNB, colored by SINR quality (green/orange/red)
- Cyan motion trails — opacity increases along trail direction
- Small SINR quality dot in the top-right of each node

#### Panel 2 — Digital Twin Replica Map
The virtual copy. Slightly bluer background (`#0A1530`). All node markers are colored `DT_BLUE` (cyan-blue) with a `DT_BLUE` outline instead of white, and drawn at 82% opacity to visually distinguish them from real nodes. Extra features:
- **Fidelity ring** — a larger circle around each twin node, colored green/amber/red based on that node's current fidelity
- **Dashed link lines** instead of solid (to show these are virtual connections)
- **⚠ warning icon** above any node in anomaly state
- Blue-tinted spine border (1.8 px) on the axes
- gNB pulse rings in `DT_BLUE` instead of the environment accent color

#### Panel 3 — KPI Radar (spider chart)
6-axis radar chart showing average performance across all nodes:

| Axis | Formula |
|------|---------|
| SINR | `(avg_sinr + 5) / 30` |
| Rate | `avg_tp / peak_gbps` |
| Latency | `1 - avg_lat / (target × 4)` |
| Coverage | `1 - nearest_gnb_dist / (0.5 × diagonal)` |
| Fidelity | `avg_fidelity / 100` |
| Reliability | same as SINR score |

#### Panel 4 — Twin vs Real Scorecard
Per-service text table showing:
- `R:XX dB` — real average SINR
- `T:XX dB` — twin average SINR
- `Δ X.X` — divergence, colored green (<1.5 dB), amber (<3 dB), or red (≥3 dB)

#### Panel 5 — Sync Latency Bars
Bar chart per node showing sync latency in ms. Red dashed target line. Bars colored green/amber/red vs the environment latency target.

#### Panel 6 — Twin Fidelity Bars
Bar chart per node showing fidelity %. Reference lines at 90% (green, excellent) and 70% (amber, warning threshold).

### Row 1 — Time-Series Comparisons

#### Panel 7 — SINR History: Physical vs Twin
Rolling SINR time series for all nodes. Solid lines = physical, dashed lines = twin, same color per node (from `tab20` colormap). Background bands: green (≥22 dB), orange (10–22 dB), red (< 10 dB).

#### Panel 8 — Throughput: Physical vs Twin
Total system throughput. Solid white line = physical, dashed cyan line = twin. Shared fill area under each curve. Yellow dotted line = peak target.

#### Panel 9 — Latency: Physical vs Twin
Per-node latency timelines. Solid = physical, dashed = twin, same color per node. Red dashed target threshold line.

### Row 2 — Analysis Panels

#### Panel 10 — Live SINR Bars (dual)
Each node gets two horizontal bars stacked vertically:
- **Top bar** (at `i + 0.18`): physical SINR, colored green/orange/red
- **Bottom bar** (at `i - 0.18`): twin SINR, always `DT_BLUE`

Values printed at right end of each bar.

#### Panel 11 — Prediction Engine
Bar chart showing current prediction error (position offset) in metres per node. Reference lines at 0.5 m (green — excellent) and 1.5 m (red — poor). Bars colored per threshold.

#### Panel 12 — Anomaly & Fault Detection
Text status panel showing:
- Large status badge: **ALL NOMINAL** (green) / **DRIFT WARNING** (amber) / **ANOMALY ACTIVE** (red)
- List of anomaly nodes and their status
- Average fidelity and handover count
- Instruction to press ⚡ to inject a fault

#### Panel 13 — SINR Heatmap Difference
Filled contour plot showing the absolute difference `|real_SINR - twin_SINR|` across the entire area:
- Green = small difference (< 1 dB) — twin matches reality well
- Yellow/orange = 2–3 dB difference — noticeable drift
- Red = > 3 dB difference — significant divergence

Red lines connect each real node (white dot) to its twin replica (blue dot), showing the position gap visually. The twin's heatmap includes `gaussian_filter` noise to simulate model uncertainty.

---

## DigitalTwin Class — Complete Internals

```
DigitalTwin
│
├── ename             ← active environment name
├── paused            ← animation pause flag
├── speed             ← time multiplier
├── show_hm           ← toggle SINR heatmap
├── inject_anomaly    ← anomaly active flag
├── anomaly_node      ← node ID under anomaly
├── anomaly_timer     ← countdown frames remaining (60 frames max)
│
├── rng               ← physical world RNG (seed 42)
├── rng_t             ← twin world RNG (seed 99) — DIFFERENT sequence
│
├── PHYSICAL WORLD STATE ────────────────────────────────────────────
│   phys_pos[nid]     ← [x, y] real position
│   phys_vel[nid]     ← [vx, vy] real velocity
│   phys_bnc[nid]     ← bounce boundary flag
│   phys_spd[nid]     ← speed in m/s (for Doppler + sync latency)
│   phys_sinr[nid]    ← rolling list of real SINR [dB], max 300
│   phys_tp[nid]      ← rolling list of real TP [Gbps], max 300
│   phys_lat[nid]     ← rolling list of real latency [ms], max 300
│   phys_trail_x/y    ← last 70 real positions (for trail drawing)
│
├── DIGITAL TWIN STATE ──────────────────────────────────────────────
│   twin_pos[nid]     ← [x, y] virtual position (lags behind real)
│   twin_vel[nid]     ← [vx, vy] virtual velocity
│   twin_sinr[nid]    ← rolling list of twin SINR [dB], max 300
│   twin_tp[nid]      ← rolling throughput
│   twin_lat[nid]     ← rolling latency
│   twin_trail_x/y    ← last 70 twin positions
│
├── QUALITY METRICS (per node) ──────────────────────────────────────
│   sync_latency[nid]   ← EMA of update delay [ms]
│   fidelity[nid]       ← EMA of accuracy score [0–100%]
│   prediction_err[nid] ← EMA of position offset [m]
│   anomaly_flag[nid]   ← currently in anomaly state
│
└── SYSTEM METRICS ──────────────────────────────────────────────────
    total_phys_tp       ← rolling system TP (physical)
    total_twin_tp       ← rolling system TP (twin)
    handovers           ← cumulative handover count
    prev_cell[nid]      ← for handover detection
```

### The sync mechanism in detail

```python
# Normal operation (no anomaly):
sync_factor = 0.12
twin_x = twin_x + (real_x - twin_x) * sync_factor + rng_t.normal(0, 0.05)
twin_y = twin_y + (real_y - twin_y) * sync_factor + rng_t.normal(0, 0.05)
```

This is a **first-order tracker** (exponential smoothing):
- `sync_factor = 0.12` means: close 12% of the gap each frame
- The `rng_t.normal(0, 0.05)` term adds 0.05 m measurement noise from the twin's own sensor model
- Together these ensure the twin always slightly lags behind reality

---

## Anomaly Injection System

Pressing `A` or clicking ⚡ calls `DT.inject()`:

```python
def inject(self):
    moving = [nodes with vx≠0 or vy≠0]  # only mobile nodes
    self.anomaly_node  = random.choice(moving)
    self.anomaly_timer = 60              # 60 × 0.04s = 2.4 seconds
    self.inject_anomaly = True
```

During the anomaly, the selected node's twin moves at **1.8×** the real node's speed:

```python
# Anomaly state:
twin_x += vx * dt * 1.8    # faster than reality → rapidly drifts away
twin_y += vy * dt * 1.8
anomaly_flag[nid] = True
```

This simulates what happens in real deployments when:
- A sensor stops transmitting location updates
- The network management system's mobility model diverges from reality
- A software bug causes the twin to use stale data
- A GPS failure leaves the twin relying on dead-reckoning

After 60 frames (~2.4 s real-time), the timer expires, the anomaly flag clears, and the twin gradually re-syncs with reality (you can watch the fidelity score recover).

**Visible effects during anomaly:**
- `⚠` icon appears above the affected node on the twin map
- Fidelity ring turns `DT_RED` on the twin map
- Global SYNC banner changes to `⚠ ANOMALY DETECTED` in red
- HUD text turns red
- Prediction error bar for that node spikes red
- Red line in heatmap diff panel grows longer
- Anomaly panel badge switches to `ANOMALY ACTIVE`

---

## Complete Code Walkthrough

```
6g_thz_digital_twin.py
│
├── ── COLOUR PALETTE ──────────────────────────────────────────────────
│   BG, PBG (physical), TBG (twin — bluer)
│   DT_BLUE, DT_GREEN, DT_AMBER, DT_RED  ← Digital Twin accent colours
│   SVC_C: service type → colour mapping
│   NS: node type → {c, m, s, l} (colour, marker, size, label)
│   MC / ME: material → face/edge colour
│   _SINR_CM: blue-black → white colormap for SINR heatmaps
│   _DIFF_CM: green → red colormap for difference map
│
├── ── THz PHYSICS ─────────────────────────────────────────────────────
│   _KF dict               tabulated absorption coefficients
│   mol_abs(d, f)           log-linear interpolation → k × d [dB]
│   friis(d, f)             free-space path loss [dB]
│   calc_sinr(x,y,gnbs,cfg,rng,spd,noise_offset)
│                           full SINR: Friis + mol abs + Rician + Doppler + NLOS
│   thz_tp(sv, bw, env_t)   Shannon × η [Gbps], capped at 12 bit/s/Hz
│   lat_ms(sv, cfg)         HARQ-aware latency model [ms]
│   sinr_col(v)             SINR value → colour string
│   build_heatmap(cfg)      36×36 SINR grid + Gaussian smoothing σ=1.5
│
├── ── ENV dict ────────────────────────────────────────────────────────
│   6 use cases, each with:
│   area, freq_hz, bw_ghz, tx_dbm, bf_db, noise_dbm, lat_ms, K, pen,
│   peak_gbps, buildings, gnbs, nodes
│
├── ── DigitalTwin class ────────────────────────────────────────────────
│   __init__()          set defaults, call reset()
│   cfg  property       returns ENV[ename]
│   reset()             re-init all physical + twin state; build heatmap
│   step(dt)            one physics tick: physical movement, twin sync,
│                       fidelity scoring, anomaly handling
│   inject()            trigger anomaly on random mobile node
│
├── ── KPI RADAR ────────────────────────────────────────────────────────
│   avg_kpi(cfg, dt)    compute 6 normalised KPI scores 0–1
│   draw_radar(ax, vals, color, title)   draw spider chart on polar axes
│
├── ── FIGURE LAYOUT ───────────────────────────────────────────────────
│   4-row outer GridSpec
│   Row 0: 3 columns — Physical map | Twin map | 2×2 KPI sub-grid
│   Row 1: SINR cmp | TP cmp | Latency cmp
│   Row 2: SINR bars | Prediction | Anomaly text | Heatmap diff
│   Bottom: 6-column control strip
│
├── ── WIDGETS ─────────────────────────────────────────────────────────
│   radio (RadioButtons)    6 use case selector
│   sl    (Slider)          speed 0.1× – 5×
│   btn_pause               pause/resume
│   btn_reset               reset all state
│   btn_anom                inject anomaly
│   axl   (Node legend)     static reference panel
│   axi   (DT legend)       sync colour meanings
│
├── ── draw_map() ──────────────────────────────────────────────────────
│   Shared renderer for physical and twin maps.
│   is_twin flag controls: blue tints, dashed links, fidelity rings,
│   reduced opacity, DT_BLUE markers.
│
├── ── draw_frame() ────────────────────────────────────────────────────
│   Called every 60 ms. Calls DT.step(), updates SYNC banner,
│   calls draw_map() twice, then redraws all 10 analysis panels.
│
├── ── on_key() ────────────────────────────────────────────────────────
│   SPACE, R, H, A, Q, +, -, 1–6
│
└── ── STARTUP ─────────────────────────────────────────────────────────
    Pre-compute all heatmaps → launch FuncAnimation → plt.show()
```

---

## Interactive Controls

### Keyboard

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume animation |
| `R` | Reset — reinitialise all physical and twin state |
| `A` | **Inject anomaly** — desync a random mobile node |
| `H` | Toggle SINR heatmap backgrounds |
| `1` – `6` | Switch use case environment |
| `+` / `=` | Increase simulation speed by 0.5× |
| `-` | Decrease simulation speed by 0.5× |
| `Q` | Quit and close window |

### On-Screen Widgets

| Widget | Type | Function |
|--------|------|----------|
| **Use Case** | Radio buttons | Switch environment instantly |
| **Speed** | Slider | 0.1× (slow-motion) to 5× (fast-forward) |
| **⏸ Pause** | Button | Same as SPACE |
| **↺ Reset** | Button | Same as R |
| **⚡ Anomaly** | Button | Same as A |

---

## Colour Coding Reference

### Digital Twin status colours

| Colour | Hex | Meaning |
|--------|-----|---------|
| `DT_GREEN` | `#00FF9F` | Synced — fidelity ≥ 90% |
| `DT_AMBER` | `#FFB800` | Drifting — fidelity 70–90% |
| `DT_RED` | `#FF3030` | Anomaly / Desync — fidelity < 70% |
| `DT_BLUE` | `#00BFFF` | Twin replica markers and outlines |

### SINR quality colours

| Colour | SINR range | Quality |
|--------|-----------|---------|
| Green `#1FD16A` | ≥ 22 dB | Excellent |
| Light green `#60EF90` | 15–22 dB | Good |
| Orange `#E89010` | 10–15 dB | Fair |
| Dark orange `#F08020` | 4–10 dB | Marginal |
| Red `#F04545` | < 4 dB | Poor / outage |

---

## Key Parameters to Tune

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `sync_factor = 0.12` | `step()` | 0.12 | How fast twin catches up. Increase → faster sync, decrease → more lag |
| `× 1.8` in anomaly drift | `step()` | 1.8 | How fast twin diverges during anomaly |
| `anomaly_timer = 60` | `inject()` | 60 frames | Duration of anomaly (~2.4 s) |
| `rng_t.normal(0, 0.05)` | `step()` | 0.05 m | Twin position noise per frame |
| `noise_offset = rng_t.normal(0, 0.5)` | SINR calc | 0.5 dB | Twin SINR measurement noise |
| `0.92 / 0.08` EMA weights | fidelity calc | 0.92 | Fidelity smoothing — lower = faster response |
| `× 800` in fidelity formula | fidelity calc | 800 | Sensitivity to position error |
| `× 0.5` SINR penalty | fidelity calc | 0.5 | Sensitivity to SINR divergence |
| `res=36` in `build_heatmap` | heatmap | 36 | Grid resolution (36×36 = 1296 points) |
| `interval=60` in `FuncAnimation` | animation | 60 ms | Frame period |
| `HIST = 300` | state | 300 | Rolling history length (~12 s) |

---

## How to Add a New Use Case

Add an entry to the `ENV` dict. The `9th` field of each node tuple is `speed_mps` — the node's real-world speed, used for Doppler calculation and sync latency:

```python
"Smart Warehouse": {
    "label": "Automated Warehouse · 140 GHz · Pick-and-Place Robots",
    "color": TEAL, "env_t": "indoor",
    "area": (60, 50),
    "freq_hz": 140e9, "bw_ghz": 40, "tx_dbm": 28, "bf_db": 26,
    "noise_dbm": -85, "lat_ms": 1.0, "K": 5.0, "pen": 20, "peak_gbps": 150,
    "buildings": [
        (0, 0, 60, 50, "Warehouse", "metal", 30),
        (5, 5, 15, 15, "Storage A", "metal", 28),
        (40, 5, 15, 15, "Storage B", "metal", 28),
    ],
    "gnbs": [(30, 25, "AP-WH", 6)],
    "nodes": [
        ("Bot1", "agv",    5, 25, 0.8, 0.0, "URLLC", False, 0.8),
        ("Bot2", "agv",   55, 25, -0.7, 0.0, "URLLC", False, 0.7),
        ("S1",  "sensor", 10, 10, 0.0, 0.0, "mMTC",  False, 0.0),
    ],
}
```

The environment will automatically appear in the RadioButtons widget.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `matplotlib` | ≥ 3.7 | All rendering, animation, widgets |
| `numpy` | ≥ 1.24 | Array maths, random number generation |
| `scipy` | ≥ 1.10 | `gaussian_filter` for heatmap smoothing |

```bash
pip install matplotlib numpy scipy
```

Change the backend at the top of the file if needed:

| System | Backend |
|--------|---------|
| Linux | `TkAgg` (default) or `Qt5Agg` |
| macOS | `MacOSX` or `Qt5Agg` |
| Windows | `TkAgg` |
| Headless | `Agg` |

---

## Limitations

- **No actual sensor data** — all "physical world" measurements are computed from the same physics model, not from real hardware. In a real DT, physical data would come from actual UE measurements.
- **No network protocol stack** — the twin does not model the actual 5G/6G protocol layers (PDCP, RLC, MAC, PHY) that would introduce real sync delay.
- **Simplified anomaly model** — real anomalies include sensor failures, clock drift, software bugs, network partitions. Here only positional drift is modelled.
- **No feedback loop** — a real DT can send commands back to the physical system to correct faults. This simulator is read-only.
- **Static heatmaps** — the SINR heatmap is pre-computed once per environment; it does not dynamically update as nodes move.
- **Free-space physics only** — no ray-tracing, no building-by-building NLOS computation beyond the penetration loss constant.

---

## Glossary

| Term | Definition |
|------|------------|
| **Digital Twin** | A continuously updated virtual model of a real physical system |
| **Fidelity** | How closely the twin's state matches physical reality, expressed as 0–100% |
| **Sync Latency** | Time for the twin to receive and incorporate an update from the physical world |
| **Anomaly** | Condition where the twin's model has diverged unacceptably from reality |
| **Prediction Error** | Distance between the twin's predicted position and the real measured position |
| **Sync Factor** | Rate at which the twin's position converges toward the real position each frame |
| **EMA** | Exponential Moving Average — a smoothing technique where `new = α × old + (1-α) × sample` |
| **Rician fading** | Small-scale signal variation with a direct LOS component; K-factor controls LOS strength |
| **K-factor** | Rician K: ratio of direct-path power to scattered-path power. K=20 = strong LOS |
| **Doppler penalty** | SINR degradation from frequency shift caused by node velocity |
| **NLOS** | Non-Line-of-Sight — path obstructed by a building; modelled as extra attenuation |
| **ITU-R P.676** | International standard for atmospheric attenuation of radio waves |
| **HARQ** | Hybrid ARQ — retransmission protocol; more rounds at low SINR → higher latency |
| **gNB** | Next-Generation Node B — 6G base station |
| **BF gain** | Beamforming gain — signal boost from steering antenna toward a specific node |
| **SINR** | Signal-to-Interference-plus-Noise Ratio — primary wireless link quality metric |
| **Shannon capacity** | `C = BW × log₂(1 + SINR)` — theoretical maximum data rate |
| **4096-QAM** | Maximum modulation order used at high SINR — 12 bits per symbol |
| **FuncAnimation** | Matplotlib class calling a function repeatedly to animate a figure |
| **RNG seed** | Initial value for a random number generator — same seed = same sequence |

---

## File Summary

```
6g_thz_digital_twin.py          ← Complete simulation (single file, ~680 lines)
README.md                       ← This document
```

---

*Physics: Friis FSPL · ITU-R P.676 molecular absorption · Rician K-factor fading*
*Digital Twin: ISO 23247 concepts · IEC 62832 reference model*
*6G targets: 3GPP TR 22.261 service requirements*
*Built with Python · matplotlib · NumPy · SciPy*