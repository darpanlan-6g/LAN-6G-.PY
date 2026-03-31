"""
=============================================================================
  6G DIGITAL TWIN MULTI-ENVIRONMENT SIMULATOR
  Demonstrating real-time telemetry across 4 distinct urban scenarios.
  Includes automated real-time CSV data logging for post-simulation analysis.
=============================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import collections
import csv
from datetime import datetime

# --- Configuration & Data Storage ---
HISTORY_LENGTH = 50

# 1. Office Data
office_time = collections.deque(maxlen=HISTORY_LENGTH)
office_occupancy = collections.deque(maxlen=HISTORY_LENGTH)
office_hvac_power = collections.deque(maxlen=HISTORY_LENGTH)

# 2. Urban Street Data
urban_time = collections.deque(maxlen=HISTORY_LENGTH)
urban_vehicles = collections.deque(maxlen=HISTORY_LENGTH)
urban_pedestrians = collections.deque(maxlen=HISTORY_LENGTH)

# 3. Highway Data
highway_time = collections.deque(maxlen=HISTORY_LENGTH)
highway_stress = collections.deque(maxlen=HISTORY_LENGTH)

# 4. Classroom Data
class_time = collections.deque(maxlen=HISTORY_LENGTH)
class_co2 = collections.deque(maxlen=HISTORY_LENGTH)
class_engagement = collections.deque(maxlen=HISTORY_LENGTH)

t = 0

# --- CSV Export Setup ---
# Generates a unique filename based on the current date and time
filename = f"digital_twin_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Write the header row for the spreadsheet
csv_writer.writerow([
    'Real_Time', 'Sim_Tick', 
    'Office_Occupancy', 'Office_HVAC_kW', 
    'Urban_Vehicles', 'Urban_Pedestrians', 
    'Highway_Stress_Pct', 
    'Class_CO2_ppm', 'Class_Engagement_Score'
])

print(f"[*] Initializing 6G Digital Twin Simulation...")
print(f"[*] Live data is being recorded to: {filename}")
print("[*] Close the plot window to end the simulation and save the file.")

# --- Setup the Dashboard Layout ---
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('6G Digital Twin Live Telemetry Dashboard', fontsize=18, fontweight='bold', color='#00f2ff')

# --- Simulation Logic (The "Digital Twin" Brain) ---
def simulate_sensor_data():
    global t
    t += 1
    
    # 1. Smart Office
    occ = random.randint(20, 100)
    hvac = occ * 1.5 + random.uniform(-10, 10) 
    office_time.append(t)
    office_occupancy.append(occ)
    office_hvac_power.append(hvac)
    
    # 2. Urban Street
    veh = random.randint(10, 50)
    ped = random.randint(0, 30)
    urban_time.append(t)
    urban_vehicles.append(veh)
    urban_pedestrians.append(ped)
    
    # 3. Highway
    base_stress = random.uniform(10, 20)
    if random.random() > 0.85: 
        base_stress += random.uniform(40, 70) 
    highway_time.append(t)
    highway_stress.append(base_stress)
    
    # 4. Classroom
    current_co2 = class_co2[-1] if len(class_co2) > 0 else 400
    if current_co2 > 1000:
        current_co2 -= random.randint(100, 200) 
    else:
        current_co2 += random.randint(10, 50) 
        
    eng = max(0, 100 - ((current_co2 - 400) / 10)) + random.uniform(-5, 5)
    class_time.append(t)
    class_co2.append(current_co2)
    class_engagement.append(eng)

    # --- Write current step to CSV ---
    timestamp = datetime.now().strftime('%H:%M:%S')
    csv_writer.writerow([
        timestamp, t,
        occ, round(hvac, 2),
        veh, ped,
        round(base_stress, 2),
        current_co2, round(eng, 2)
    ])
    # Flush ensures data is written to disk immediately, preventing data loss when you exit
    csv_file.flush() 

# --- Plotting/Updating Logic ---
def update_dashboard(frame):
    simulate_sensor_data()
    
    for ax in axs.flat:
        ax.cla()
        ax.grid(True, color='#2a2a2a', linestyle='--', alpha=0.7)
    
    # Top-Left: The Office
    axs[0, 0].plot(office_time, office_occupancy, label='Occupancy (People)', color='#00f2ff', lw=2)
    axs[0, 0].plot(office_time, office_hvac_power, label='HVAC Power (kW)', color='#ff6b6b', lw=2)
    axs[0, 0].set_title('1. Smart Office: Occupancy vs Energy', color='white')
    axs[0, 0].legend(loc='upper left')
    
    # Top-Right: Urban Streets
    axs[0, 1].plot(urban_time, urban_vehicles, label='Vehicles', color='#f9ca24', lw=2)
    axs[0, 1].plot(urban_time, urban_pedestrians, label='Pedestrians', color='#badc58', lw=2)
    axs[0, 1].set_title('2. Urban Intersection: Traffic Flow', color='white')
    axs[0, 1].legend(loc='upper left')
    
    # Bottom-Left: Highways
    axs[1, 0].plot(highway_time, highway_stress, label='Bridge Structural Stress (%)', color='#ff7979', lw=2)
    axs[1, 0].fill_between(highway_time, highway_stress, color='#ff7979', alpha=0.3)
    axs[1, 0].axhline(y=75, color='red', linestyle='--', label='Warning Threshold')
    axs[1, 0].set_title('3. Highway: Live Infrastructure Health', color='white')
    axs[1, 0].legend(loc='upper left')
    
    # Bottom-Right: Classrooms
    ax_co2 = axs[1, 1]
    ax_eng = ax_co2.twinx() 
    
    ax_co2.plot(class_time, class_co2, label='CO2 Levels (ppm)', color='#c7ecee', lw=2)
    ax_eng.plot(class_time, class_engagement, label='Engagement Score', color='#686de0', lw=2, linestyle=':')
    
    ax_co2.set_title('4. 6G Classroom: Environment vs Engagement', color='white')
    ax_co2.set_ylabel('CO2 (ppm)', color='#c7ecee')
    ax_eng.set_ylabel('Engagement', color='#686de0')

# --- Run the Animation ---
ani = animation.FuncAnimation(fig, update_dashboard, interval=500, cache_frame_data=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Clean up file when the window is closed
csv_file.close()
print("[*] Simulation ended. CSV file saved successfully.")