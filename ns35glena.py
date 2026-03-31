import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np

# ==========================================
# 1. Simulation Parameters & Topology
# ==========================================
sim_time = 20.0  # seconds
dt = 0.1         # 100ms frame update rate
frames = int(sim_time / dt)
bounds = 100.0   # 100m x 100m area

# Static Nodes: gNBs (Blue) and RSUs (Yellow)
gnb_positions = np.array([
    [25.0, 90.0], [75.0, 90.0], [50.0, 50.0], 
    [25.0, 10.0], [75.0, 10.0]
])
rsu_positions = np.array([[10.0, 50.0], [90.0, 50.0]])

# Buildings (Grey) - [x_min, y_min, width, height]
buildings = [
    [30.0, 30.0, 10.0, 10.0], # b1: Box(30,40, 30,40)
    [60.0, 60.0, 10.0, 10.0]  # b2: Box(60,70, 60,70)
]

# ==========================================
# 2. Mobility Setup
# ==========================================
num_vehicles = 10
# Speeds in km/h converted to m/s
car_speeds_kmh = np.array([80, 75, 90, 70, 85, 95, 78, 65, 60, 120])
vehicle_velocities = car_speeds_kmh * (1000.0 / 3600.0)

# Initial Vehicle Positions (x=0, y=0, 10, 20...)
vehicle_pos = np.zeros((num_vehicles, 2))
for i in range(num_vehicles):
    vehicle_pos[i] = [0.0, i * 10.0]

# Initial Pedestrian Positions (Random within bounds)
num_pedestrians = 5
ped_pos = np.random.uniform(0, bounds, (num_pedestrians, 2))

# ==========================================
# 3. Visualization Setup
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, bounds)
ax.set_ylim(0, bounds)
ax.set_title("6G FR3 V2X 24GHz Topology & Mobility", fontsize=14)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.grid(True, linestyle='--', alpha=0.6)

# Draw Buildings
for b in buildings:
    rect = patches.Rectangle((b[0], b[1]), b[2], b[3], linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5)
    ax.add_patch(rect)

# Plot Static Nodes
ax.scatter(gnb_positions[:, 0], gnb_positions[:, 1], c='blue', s=100, marker='^', label='gNBs')
ax.scatter(rsu_positions[:, 0], rsu_positions[:, 1], c='yellow', s=100, marker='s', edgecolors='black', label='RSUs')

# Initialize Dynamic Scatters
scat_cars = ax.scatter([], [], c='green', s=60, label='Cars')
scat_trucks = ax.scatter([], [], c='orange', s=80, marker='s', label='Trucks')
scat_emerg = ax.scatter([], [], c='red', s=80, marker='*', label='Emergency')
scat_peds = ax.scatter([], [], c='purple', s=40, label='Pedestrians')

ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

# ==========================================
# 4. Animation Update Function
# ==========================================
def update(frame):
    global vehicle_pos, ped_pos
    
    # Update Vehicles (Constant Velocity moving right)
    vehicle_pos[:, 0] += vehicle_velocities * dt
    
    # Wrap around logic for continuous simulation
    vehicle_pos[:, 0] = vehicle_pos[:, 0] % bounds
    
    # Update Pedestrians (Random Walk: random angle, uniform speed 0.8-1.5 m/s)
    angles = np.random.uniform(0, 2 * np.pi, num_pedestrians)
    speeds = np.random.uniform(0.8, 1.5, num_pedestrians)
    ped_pos[:, 0] += np.cos(angles) * speeds * dt
    ped_pos[:, 1] += np.sin(angles) * speeds * dt
    
    # Keep pedestrians within bounds (bounce/clip)
    ped_pos = np.clip(ped_pos, 0, bounds)

    # Update scatter plot data based on your specific indices
    scat_cars.set_offsets(vehicle_pos[0:7])       # Indices 0-6
    scat_trucks.set_offsets(vehicle_pos[7:9])     # Indices 7-8
    scat_emerg.set_offsets(vehicle_pos[9:10])     # Index 9
    scat_peds.set_offsets(ped_pos)
    
    return scat_cars, scat_trucks, scat_emerg, scat_peds

# Create and run animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=dt*1000, blit=True)

plt.tight_layout()
plt.show()