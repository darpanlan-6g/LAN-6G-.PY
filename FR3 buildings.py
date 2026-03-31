"""
=============================================================================
  6G ISAC DIGITAL TWIN — 3D INTERACTIVE ENVIRONMENT (ns-3 FR3 V2X Merge)
  Features:
    - Scenario ported directly from C++ ns-3 (FR3 @ 24 GHz, 200 MHz BW)
    - 5 gNBs, 8 Buildings (Concrete, Glass, Metal), RSUs
    - Vehicles (Cars, Trucks, Emergency) & Pedestrian Mobility
    - 3D Isometric & Top-Down Camera controls
    - Dynamic LOS/NLOS link coloring & ISAC Radar Pulse
=============================================================================
"""

import math
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider, RadioButtons

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS (3D Geometry & Intersection)
# ─────────────────────────────────────────────────────────────────────────────

def get_cube_faces(x, y, z, dx, dy, dz):
    """Generates the 6 faces of a 3D box for Poly3DCollection."""
    vertices = [
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ]
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]], # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]], # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]], # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]], # Back
        [vertices[1], vertices[2], vertices[6], vertices[5]], # Right
        [vertices[4], vertices[7], vertices[3], vertices[0]]  # Left
    ]
    return faces

def intersect_line_aabb_2d(p1, p2, box):
    """Checks if a 2D line segment intersects a 2D bounding box."""
    x1, y1 = p1
    x2, y2 = p2
    bx, by, bw, bh = box
    
    if (bx <= x1 <= bx+bw and by <= y1 <= by+bh) or (bx <= x2 <= bx+bw and by <= y2 <= by+bh):
        return True

    edges = [
        ((bx, by), (bx+bw, by)), ((bx+bw, by), (bx+bw, by+bh)),
        ((bx+bw, by+bh), (bx, by+bh)), ((bx, by+bh), (bx, by))
    ]
    
    def ccw(A, B, C): return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    def intersect(A, B, C, D): return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    for edge in edges:
        if intersect((x1, y1), (x2, y2), edge[0], edge[1]):
            return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
#  DIGITAL TWIN ENGINE (Merged with ns-3 Scenario)
# ─────────────────────────────────────────────────────────────────────────────

class IsacDigitalTwin3D:
    def __init__(self):
        self.sim_time = 0.0
        self.dt = 0.05
        
        # Radar State
        self.radar_radius = 1.0
        self.radar_max_radius = 60.0
        self.radar_pulse_speed = 1.5
        self.radar_collection = None
        
        # Camera State
        self.view_mode = "Isometric"
        
        # ns-3 C++ Topology Port
        # 5 gNBs (Blue)
        self.gnbs = [
            {"id": 0, "x": 25, "y": 90, "z": 15},
            {"id": 1, "x": 75, "y": 90, "z": 15},
            {"id": 2, "x": 50, "y": 50, "z": 15}, # Central ISAC Node
            {"id": 3, "x": 25, "y": 10, "z": 15},
            {"id": 4, "x": 75, "y": 10, "z": 15}
        ]
        
        # RSUs (Yellow)
        self.rsus = [
            {"id": "R1", "x": 30, "y": 50, "z": 5, "color": "#facc15"},
            {"id": "R2", "x": 70, "y": 50, "z": 5, "color": "#facc15"}
        ]
        
        # 8 Buildings from C++ Script (Concrete, Glass, Metal)
        self.buildings = [
            {"x": 10, "y": 10, "w": 10, "d": 15, "h": 30, "color": "#64748b", "alpha": 0.8}, # B1 Concrete
            {"x": 25, "y": 15, "w": 10, "d": 15, "h": 40, "color": "#38bdf8", "alpha": 0.5}, # B2 Glass
            {"x": 40, "y": 20, "w": 10, "d": 15, "h": 15, "color": "#334155", "alpha": 0.9}, # B3 Metal
            {"x": 15, "y": 40, "w": 10, "d": 15, "h": 20, "color": "#64748b", "alpha": 0.8}, # B4 Concrete
            {"x": 55, "y": 45, "w": 15, "d": 15, "h": 12, "color": "#38bdf8", "alpha": 0.5}, # B5 Glass
            {"x": 75, "y": 25, "w": 12, "d": 15, "h": 35, "color": "#64748b", "alpha": 0.8}, # B6 Concrete
            {"x": 85, "y": 60, "w": 8,  "d": 8,  "h": 8,  "color": "#334155", "alpha": 0.9}, # B7 Metal
            {"x": 20, "y": 70, "w": 10, "d": 15, "h": 12, "color": "#64748b", "alpha": 0.8}  # B8 Concrete
        ]
        
        # UEs (Vehicles & Pedestrians ported from C++)
        self.ues = []
        
        # 7 Cars (Green)
        speeds_kmh = [80, 75, 90, 70, 85, 95, 78]
        for i in range(7):
            self.ues.append({"id": f"C{i}", "x": 5, "y": 30+i*5, "z": 1.5, "vx": speeds_kmh[i]/3.6, "vy": 0, "color": "#22c55e", "type": "linear"})
            
        # 2 Trucks (Orange)
        self.ues.append({"id": "T1", "x": 10, "y": 70, "z": 2.0, "vx": 65/3.6, "vy": 0, "color": "#f97316", "type": "linear"})
        self.ues.append({"id": "T2", "x": 90, "y": 75, "z": 2.0, "vx": -60/3.6, "vy": 0, "color": "#f97316", "type": "linear"})
        
        # 1 Emergency (Red)
        self.ues.append({"id": "E1", "x": 20, "y": 50, "z": 1.5, "vx": 120/3.6, "vy": 0, "color": "#ef4444", "type": "linear"})
        
        # 5 Pedestrians (Purple) - using circular/figure-8 approximations for 3D visual flair
        self.ues.append({"id": "P1", "x": 50, "y": 30, "z": 1.5, "angle": 0.0, "radius": 15, "speed": 1.0, "color": "#a855f7", "type": "circle"})
        self.ues.append({"id": "P2", "x": 70, "y": 70, "z": 1.5, "angle": 3.14, "radius": 10, "speed": 0.8, "color": "#a855f7", "type": "circle"})
        
        self.ue_lines = []
        self.ue_scatter = None
        self.rsu_scatter = None
        
        self.setup_ui()

    def setup_ui(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(14, 9), facecolor="#020617")
        self.fig.canvas.manager.set_window_title('6G FR3 V2X Digital Twin - 3D')
        
        self.ax = self.fig.add_axes([0.05, 0.15, 0.9, 0.8], projection='3d')
        self.ax.set_facecolor("#020617")
        self.ax.grid(False)
        self.ax.axis('off')
        
        # NS-3 Bounds (0 to 100m)
        self.ax.set_xlim([0, 100])
        self.ax.set_ylim([0, 100])
        self.ax.set_zlim([0, 60])
        
        # HUD Overlays
        self.hud_title = self.fig.text(0.05, 0.92, "3GPP 6G NR FR3 @ 24 GHz", color="white", fontsize=16, fontweight="bold")
        self.hud_stats = self.fig.text(0.35, 0.92, "🟢 CONNECTED  -- Gbps", color="#a3e635", fontsize=12, fontweight="bold", backgroundcolor="#1e293b")
        self.hud_alert = self.fig.text(0.05, 0.88, "📡 NetAnim Topology Ported. ISAC Pulse active on gNB-2.", color="#facc15", fontsize=10)
        
        # UI Widgets
        ax_slider = self.fig.add_axes([0.65, 0.05, 0.25, 0.03], facecolor="#1e293b")
        self.slider_speed = Slider(ax_slider, 'ISAC Pulse Speed  ', 0.5, 5.0, valinit=self.radar_pulse_speed, color="#38bdf8")
        self.slider_speed.on_changed(self.update_speed)
        
        ax_radio = self.fig.add_axes([0.2, 0.02, 0.15, 0.08], facecolor="#020617")
        self.radio_view = RadioButtons(ax_radio, ('Isometric', 'Top-Down Map'), activecolor="#38bdf8")
        for label in self.radio_view.labels: label.set_color("white")
        self.radio_view.on_clicked(self.update_view)

        self.draw_static_environment()
        self.update_view("Isometric")

    def update_speed(self, val):
        self.radar_pulse_speed = val

    def update_view(self, label):
        self.view_mode = label
        if label == "Isometric":
            self.ax.view_init(elev=35, azim=45)
        else:
            self.ax.view_init(elev=90, azim=-90)

    def draw_static_environment(self):
        # Ground grid (0 to 100m)
        xx, yy = np.meshgrid(np.linspace(0, 100, 10), np.linspace(0, 100, 10))
        zz = np.zeros_like(xx)
        self.ax.plot_wireframe(xx, yy, zz, color="#334155", alpha=0.3, linewidth=0.5)

        # Draw 5 gNBs
        for g in self.gnbs:
            t_faces = get_cube_faces(g["x"]-1, g["y"]-1, 0, 2, 2, g["z"])
            self.ax.add_collection3d(Poly3DCollection(t_faces, facecolors="#2563eb", linewidths=1, edgecolors='#60a5fa', alpha=0.5))
        
        # Draw Buildings
        for b in self.buildings:
            faces = get_cube_faces(b["x"], b["y"], 0, b["w"], b["d"], b["h"])
            self.ax.add_collection3d(Poly3DCollection(faces, facecolors=b["color"], linewidths=1, edgecolors='#94a3b8', alpha=b["alpha"]))

        # Draw RSUs
        rsu_x = [r["x"] for r in self.rsus]
        rsu_y = [r["y"] for r in self.rsus]
        rsu_z = [r["z"] for r in self.rsus]
        self.rsu_scatter = self.ax.scatter(rsu_x, rsu_y, rsu_z, color="#facc15", s=100, marker='^', zorder=10)

        # Init beam lines
        for _ in self.ues:
            line, = self.ax.plot([], [], [], lw=1.5)
            self.ue_lines.append(line)

    def calculate_sphere_coords(self, radius):
        # Pulse originates from central gNB (id=2) at (50, 50, 15)
        u = np.linspace(0, 2 * np.pi, 25)
        v = np.linspace(0, np.pi, 25)
        x = radius * np.outer(np.cos(u), np.sin(v)) + 50
        y = radius * np.outer(np.sin(u), np.sin(v)) + 50
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + 15
        
        z[z < 0] = 0 
        return x, y, z

    def run(self):
        plt.ion()
        print("\n" + "="*60)
        print(" 🌐 6G NS-3 SCENARIO PORTED TO 3D PYTHON ENGINE")
        print("="*60 + "\n")

        colors_array = [ue["color"] for ue in self.ues]

        while plt.fignum_exists(self.fig.number):
            self.sim_time += self.dt
            
            ue_x, ue_y, ue_z = [], [], []
            tput_display = 0.0
            
            for i, ue in enumerate(self.ues):
                # Mobility Updates
                if ue["type"] == "linear":
                    ue["x"] += ue["vx"] * self.dt
                    # Loop boundaries for continuous visual
                    if ue["x"] > 100: ue["x"] = 0
                    if ue["x"] < 0: ue["x"] = 100
                    ux, uy, uz = ue["x"], ue["y"], ue["z"]
                else: # Circular pedestrians
                    ue["angle"] += ue["speed"] * self.dt
                    ux = ue["x"] + math.cos(ue["angle"]) * ue["radius"]
                    uy = ue["y"] + math.sin(ue["angle"]) * ue["radius"]
                    uz = ue["z"]
                    
                ue_x.append(ux); ue_y.append(uy); ue_z.append(uz)
                
                # Connect to nearest gNB
                best_g = min(self.gnbs, key=lambda g: math.hypot(ux-g["x"], uy-g["y"]))
                tx, ty, tz = best_g["x"], best_g["y"], best_g["z"]
                
                # Check Intersection
                is_blocked = False
                for b in self.buildings:
                    box = (b["x"], b["y"], b["w"], b["d"])
                    if intersect_line_aabb_2d((tx, ty), (ux, uy), box):
                        is_blocked = True
                        break
                
                self.ue_lines[i].set_data_3d([tx, ux], [ty, uy], [tz, uz])
                
                dist = math.hypot(ux-tx, uy-ty)
                if is_blocked:
                    self.ue_lines[i].set_color("#f59e0b") # NLOS Orange
                    tput_display += max(0.1, 1.5 - (dist/40.0))
                else:
                    self.ue_lines[i].set_color("#4ade80") # LOS Green
                    tput_display += max(0.5, 4.5 - (dist/50.0))

            # Update UE visual blocks
            if self.ue_scatter: self.ue_scatter.remove()
            self.ue_scatter = self.ax.scatter(ue_x, ue_y, ue_z, c=colors_array, s=80, marker='s', edgecolors="white", depthshade=False, zorder=15)

            # Update ISAC Radar Pulse
            self.radar_radius += (self.radar_pulse_speed * 10.0) * self.dt
            if self.radar_radius > self.radar_max_radius:
                self.radar_radius = 1.0
                
            if self.radar_collection:
                self.radar_collection.remove()
                
            rx, ry, rz = self.calculate_sphere_coords(self.radar_radius)
            self.radar_collection = self.ax.plot_wireframe(rx, ry, rz, color="#4ade80", alpha=0.15, linewidth=0.8)

            # Update HUD Data
            self.hud_stats.set_text(f"🟢 AGGREGATE TPUT:  {tput_display:.2f} Gbps")

            self.fig.canvas.draw_idle()
            plt.pause(0.02)

if __name__ == "__main__":
    IsacDigitalTwin3D().run()