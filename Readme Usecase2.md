# 📡 6G NR FR3 @ 24 GHz — Interactive Live Simulation

## 🚀 Overview

This project is a **real-time interactive simulation of 6G NR (FR3 band at 24 GHz)** using Python and Matplotlib. It visualizes wireless communication performance across multiple environments with live updates of SINR, throughput, mobility, and network behavior.

The simulator is designed for:

* Research demonstrations
* Wireless communication learning
* Network performance visualization
* 6G / V2X experimentation

---

## 🎯 Features

* ✅ Real-time node movement & animation
* 📊 Live SINR heatmap visualization
* 📶 Per-node SINR and throughput tracking
* 🔄 Handover detection between base stations
* 🎛 Interactive controls (pause, speed, filters)
* 🌍 Multiple environments:

  * Office
  * Urban Streets
  * Highway
  * Classroom
* 📡 Service types:

  * URLLC (Ultra Reliable Low Latency)
  * eMBB (Enhanced Mobile Broadband)
  * mMTC (Massive IoT)
* 📁 CSV export of simulation data
* 📈 Multiple plots:

  * SINR time series
  * Throughput timeline
  * SINR vs distance
  * CDF distribution
  * Aggregate throughput

---

## 🛠 Requirements

Install dependencies:

```bash
pip install matplotlib numpy scipy
```

---

## ▶️ How to Run

```bash
python 6g_nr_fr3_live_sim.py
```

---

## 🧠 Code Structure Explained

### 1. Environment Configuration

The `ENVIRONMENTS` dictionary defines simulation scenarios:

* Area size
* Frequency & bandwidth
* Base stations (gNBs)
* Buildings with material properties
* Nodes (devices with mobility)

Example:

```python
ENVIRONMENTS = {
    "Office": {
        "area": (100, 80),
        "freq_ghz": 24.0,
        ...
    }
}
```

---

### 2. Physics Layer

#### 📡 Path Loss (Friis Model)

```python
def friis_pl(d_m, freq_ghz):
```

Calculates signal attenuation over distance.

#### 📶 SINR Calculation

```python
def compute_sinr(...)
```

* Computes signal strength from nearest base station
* Adds interference + noise
* Returns SINR (dB)

#### 📊 Throughput (Shannon Capacity)

```python
def shannon_tp(sinr_db, bw_mhz):
```

Uses Shannon formula to estimate data rate.

---

### 3. Simulation State (`SimState` class)

This is the **core engine** of the simulation.

#### Key Responsibilities:

* Maintain node positions & velocities
* Track SINR and throughput history
* Handle movement and boundary conditions
* Detect handovers
* Manage simulation time

#### Important Methods:

##### 🔄 Reset Simulation

```python
def reset(self):
```

##### ⏱ Simulation Step

```python
def step(self, dt=0.05):
```

* Updates positions
* Computes SINR & throughput
* Stores history

##### 📁 Export Data

```python
def export_csv(self):
```

---

### 4. Visualization (Matplotlib)

The UI is built using:

* `matplotlib.pyplot`
* `gridspec` for layout
* `FuncAnimation` for real-time updates

#### Panels:

* 🗺 Topology + heatmap
* 📊 SINR bars
* 📈 Throughput timeline
* 📉 SINR time series
* 📍 SINR vs distance
* 📦 CDF
* 📊 Aggregate throughput

---

### 5. Interactive Controls

#### 🎛 Widgets:

* **RadioButtons** → Switch environment
* **CheckButtons** → Toggle services & display
* **Slider** → Adjust speed
* **Buttons**:

  * Pause/Resume
  * Reset
  * Export CSV

---

### 6. Keyboard Shortcuts

| Key   | Action             |
| ----- | ------------------ |
| SPACE | Pause / Resume     |
| R     | Reset simulation   |
| E     | Export CSV         |
| 1–4   | Switch environment |
| + / - | Change speed       |

---

### 7. Heatmap Generation

```python
def build_heatmap(cfg):
```

* Computes SINR across grid
* Applies Gaussian smoothing
* Cached for performance

---

## 📊 Output

### Real-Time Visualization Includes:

* Node movement with trails
* Dynamic SINR coloring
* Base station coverage rings
* Live KPIs (SINR, throughput, handovers)

---

## 📁 CSV Export Format

```csv
Node,Type,Service,SINR_dB,Throughput_Mbps,X,Y
```

---

## ⚙️ Customization

You can easily modify:

### ➤ Add New Environment

Edit `ENVIRONMENTS` dictionary

### ➤ Add New Node Type

Update:

```python
NODE_STYLE = {...}
```

### ➤ Change Physics Model

Modify:

* `friis_pl()`
* `compute_sinr()`

---

## 🧪 Use Cases

* 📡 6G / 5G research demos
* 🚗 V2X (Vehicle-to-Everything) simulation
* 🧠 Teaching wireless communication
* 📊 Network performance analysis

---

## ⚠️ Notes

* This is a **simulation model**, not a full 3GPP-compliant stack
* SINR and throughput are **approximations**
* Designed for visualization, not production deployment

---

## 👨‍💻 Author

Developed for **6G NR FR3 simulation and visualization** experiments. 

---

If you want, I can also:

* convert this into a **one-page PDF**
* add **architecture diagram**
* or simplify it for **college submission/report**

