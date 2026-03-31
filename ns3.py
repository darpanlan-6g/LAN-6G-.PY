import sys
from ns import core, network, mobility, internet, applications, nr, buildings, netanim, spectrum, point_to_point, propagation

# ==============================================================================
# GLOBAL DATA LOGGERS & CALLBACKS
# ==============================================================================
sinr_file = open("master_sinr_trace.csv", "w")
sinr_file.write("Time_s,UE_ID,Cell_ID,SINR_dB\n")

def sinr_trace_callback(context, rnti, cell_id, sinr, cc_id):
    t = core.Simulator.Now().GetSeconds()
    # Convert linear SINR to dB
    sinr_db = 10 * core.math.log10(sinr) if sinr > 0 else -100.0
    sinr_file.write(f"{t:.5f},{rnti},{cell_id},{sinr_db:.2f}\n")

# ==============================================================================
# MAIN ENGINE
# ==============================================================================
def main():
    # 1. PARAMETERS
    frequency = 24.0e9
    sim_time = 20.0
    num_gnbs = 5
    num_vehicles = 10
    num_pedestrians = 2

    # Enable Logging
    core.LogComponentEnable("UdpClient", core.LOG_LEVEL_ERROR)
    core.LogComponentEnable("UdpServer", core.LOG_LEVEL_ERROR)

    # 2. PHYSICAL BUILDINGS (8 Block Mixture)
    # Format: x_min, x_max, y_min, y_max, z_min, z_max
    def create_building(x_min, x_max, y_min, y_max, height, wall_type):
        b = buildings.Building()
        b.SetBoundaries(core.Box(x_min, x_max, y_min, y_max, 0, height))
        b.SetExtWallsType(wall_type)
        return b

    # Create 8 Buildings
    create_building(10, 20, 10, 25, 30, buildings.Building.ConcreteWithWindows)
    create_building(25, 35, 15, 30, 40, buildings.Building.Wood)
    create_building(40, 50, 20, 35, 15, buildings.Building.StoneBlocks)
    create_building(15, 25, 40, 55, 20, buildings.Building.ConcreteWithWindows)
    create_building(55, 70, 45, 60, 12, buildings.Building.Wood)
    create_building(75, 87, 25, 40, 35, buildings.Building.ConcreteWithWindows)
    create_building(85, 93, 60, 68, 8, buildings.Building.StoneBlocks)
    create_building(20, 30, 70, 85, 12, buildings.Building.ConcreteWithWindows)

    # 3. NODES & INFRASTRUCTURE
    gnb_nodes = network.NodeContainer()
    gnb_nodes.Create(num_gnbs)
    
    vehicle_nodes = network.NodeContainer()
    vehicle_nodes.Create(num_vehicles)
    
    ped_nodes = network.NodeContainer()
    ped_nodes.Create(num_pedestrians)
    
    remote_host_container = network.NodeContainer()
    remote_host_container.Create(1)

    # 4. MOBILITY
    mob_helper = mobility.MobilityHelper()
    
    # gNBs static
    gnb_pos = mobility.ListPositionAllocator()
    gnb_pos.Add(core.Vector(25, 90, 15))
    gnb_pos.Add(core.Vector(75, 90, 15))
    gnb_pos.Add(core.Vector(50, 50, 15))
    gnb_pos.Add(core.Vector(25, 10, 15))
    gnb_pos.Add(core.Vector(75, 10, 15))
    mob_helper.SetPositionAllocator(gnb_pos)
    mob_helper.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mob_helper.Install(gnb_nodes)

    # Vehicles linear
    mob_helper.SetMobilityModel("ns3::ConstantVelocityMobilityModel")
    mob_helper.Install(vehicle_nodes)
    for i in range(num_vehicles):
        m = vehicle_nodes.Get(i).GetObject(mobility.ConstantVelocityMobilityModel.GetTypeId())
        m.SetPosition(core.Vector(5, 30 + i * 5, 1.5))
        m.SetVelocity(core.Vector(80/3.6, 0, 0))

    # Pedestrians waypoint
    mob_helper.SetMobilityModel("ns3::WaypointMobilityModel")
    mob_helper.Install(ped_nodes)
    p0 = ped_nodes.Get(0).GetObject(mobility.WaypointMobilityModel.GetTypeId())
    p0.AddWaypoint(mobility.Waypoint(core.Seconds(0), core.Vector(5, 15, 1.5)))
    p0.AddWaypoint(mobility.Waypoint(core.Seconds(5), core.Vector(15, 15, 1.5))) # Building 1
    p0.AddWaypoint(mobility.Waypoint(core.Seconds(15), core.Vector(25, 20, 1.5)))

    buildings.BuildingsHelper.Install(gnb_nodes)
    buildings.BuildingsHelper.Install(vehicle_nodes)
    buildings.BuildingsHelper.Install(ped_nodes)

    # 5. NR HELPER & EPC
    nr_helper = nr.NrHelper()
    epc_helper = nr.NrPointToPointEpcHelper()
    nr_helper.SetEpcHelper(epc_helper)

    # Channel Mixture
    # Note: For custom RIS in Python, we use the 3GPP channel which is closest to the physics requested
    channel = spectrum.MultiModelSpectrumChannel()
    loss_model = propagation.ThreeGppUmaPropagationLossModel()
    loss_model.SetAttribute("Frequency", core.DoubleValue(frequency))
    channel.AddPropagationLossModel(loss_model)

    bwp = nr.BandwidthPartInfo()
    bwp.m_centralFrequency = frequency
    bwp.m_channelBandwidth = 200e6
    bwp.m_bwpId = 0
    bwp.SetChannel(channel)
    bwp_vec = [bwp]

    nr_helper.SetGnbPhyAttribute("TxPower", core.DoubleValue(40.0))
    gnb_devs = nr_helper.InstallGnbDevice(gnb_nodes, bwp_vec)
    
    all_ue_nodes = network.NodeContainer()
    all_ue_nodes.Add(vehicle_nodes)
    all_ue_nodes.Add(ped_nodes)
    ue_devs = nr_helper.InstallUeDevice(all_ue_nodes, bwp_vec)

    # 6. INTERNET & TRAFFIC
    stack = internet.InternetStackHelper()
    stack.Install(gnb_nodes)
    stack.Install(all_ue_nodes)
    stack.Install(remote_host_container)

    p2p = point_to_point.PointToPointHelper()
    p2p.SetDeviceAttribute("DataRate", core.StringValue("100Gbps"))
    pgw = epc_helper.GetPgwNode()
    internet_devs = p2p.Install(pgw, remote_host_container.Get(0))
    
    address = internet.Ipv4AddressHelper()
    address.SetBase("1.0.0.0", "255.0.0.0")
    address.Assign(internet_devs)
    
    ue_ip_ifaces = epc_helper.AssignUeIpv4Address(ue_devs)

    # Attach & UDP Apps
    for i in range(all_ue_nodes.GetN()):
        nr_helper.AttachToGnb(ue_devs.Get(i), gnb_devs.Get(i % num_gnbs))
        
        server = applications.UdpServerHelper(1234)
        s_apps = server.Install(all_ue_nodes.Get(i))
        s_apps.Start(core.Seconds(0.1))
        s_apps.Stop(core.Seconds(sim_time))
        
        client = applications.UdpClientHelper(ue_ip_ifaces.GetAddress(i), 1234)
        client.SetAttribute("Interval", core.TimeValue(core.MilliSeconds(10)))
        client.SetAttribute("PacketSize", core.UintegerValue(1000))
        c_apps = client.Install(remote_host_container.Get(0))
        c_apps.Start(core.Seconds(1.0))
        c_apps.Stop(core.Seconds(sim_time))

    # 7. TRACING & RUN
    core.Config.Connect("/NodeList/*/DeviceList/*/$ns3::NrUeNetDevice/ComponentCarrierMapUe/*/NrUePhy/DlDataSinr",
                        core.MakeCallback(sinr_trace_callback))
    
    anim = netanim.AnimationInterface("master_mixture_twin_python.xml")
    
    print("🚀 RUNNING PYTHON 6G MASTER MIXTURE...")
    core.Simulator.Stop(core.Seconds(sim_time))
    core.Simulator.Run()
    core.Simulator.Destroy()
    
    sinr_file.close()
    print("✅ Simulation Complete. Results in master_sinr_trace.csv")

if __name__ == '__main__':
    main()