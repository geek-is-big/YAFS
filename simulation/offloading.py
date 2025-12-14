"""
    Offloading simulation
"""
import os
import time
import json
import random
import logging.config
from enum import Enum

import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from yafs.core import Sim
from yafs.application import Application, Message, fractional_selectivity
from yafs.topology import Topology
from yafs.population import Statical
from yafs.placement import JSONPlacement
from yafs.selection import First_ShortestPath
from yafs.distribution import deterministic_distribution
from yafs.stats import Stats

# from jsonPopulation import JSONPopulation

class TimeUnit(Enum):
    SECOND = 1
    MILLISECOND = 2

APP_NAME = "OverMobileCase"
TIME_UNIT = TimeUnit.SECOND

def watt_to_wpt(watt: float) -> float:
    match TIME_UNIT:
        case TimeUnit.SECOND:
            return watt / 3600.0
        case TimeUnit.MILLISECOND:
            return watt / (3600.0 * 1000.0)
        case _:
            raise ValueError("Unknown TIME_UNIT")
        
def ips_to_ipt(ips: float) -> float:
    match TIME_UNIT:
        case TimeUnit.SECOND:
            return ips
        case TimeUnit.MILLISECOND:
            return ips * 10 ** -3
        case _:
            raise ValueError("Unknown TIME_UNIT")

def create_topology() -> Topology:
    """
    TOPOLOGY
    """
    topology_json = {}
    topology_json["entity"] = []
    topology_json["link"] = []

    # "COST" only used to calculate get_cost_cloud() which is currently commmented
    # IPT - Instructions Per Time Unit, where Time Unit is what is 1 in simulation
    # In out case Time Unit is 1 second so IPT = MIPS and WATT = Power/3600
    # For the case when Time Unit is 1 millisecond the IPT = MIPS * 10 ^ -3

    # TODO: Move description to a point where JSON will be loaded from file
    # Fog - PCEngines ALIX 3D2 (500MHzx86 CPU, 256MB of RAM):
    # - IPT: 500 * 10^6 instructions per second (500 MIPS) -> 500 * 10^3 IPT
    # - RAM: 256 MB
    # - CPU Power Consumption: 0.9W (AMD GeodeTM LX Processors Data Book)
    # - Link Power: Trans 4.9W / Recv 3.7W
    ##   Achilles and the Tortoise: Power Consumption in IEEE 802.11n and IEEE 802.11g Networks
    ##.  https://www.robertoriggio.net/papers/greencom2013.pdf
    #
    # Modile - Galaxy S4 (1.9GHz Quad-Core Krait 300 CPU, 2GB of RAM):
    # - IPT: 1.9 * 10^9 instructions per second (1900 MIPS) -> 1.9 * 10^6 IPT
    # - RAM: 2 GB
    # - CPU Power Consumption: 1.3W (1890Mhz, full utilization)
    # - WiFi Link Power: -50dBm Trans: 654mW / Recv: 451mW
    #                    -80dBm Trans: 1113mW / Recv: 633mW
    # - BLE (4.0) Link Power: Recv: 174mW
    ##   Smartphone Energy Drain in the Wild: Analysis and Implications
    ##.  https://engineering.purdue.edu/~ychu/publications/TR-ECE-15-03.pdf
    #
    # SmartWatch - LG Urbane watch (768GHz Quad-Core ARM Cortex-A7 CPU, 512MB of RAM):
    # - IPT: 768 * 10^6 instructions per second -> 768 * 10^3 IPT
    # - RAM: 512 MB
    # - CPU Power Consumption: 361mW
    # - WiFi Link Power: Trans: 739.9 / Recv: 400.1mW
    # - BLE (4.1) Link Power: Trans: 180.7mW / Recv: 174.9mW
    ##   Poster: Measuring and Optimizing Android Smartwatch Energy Consumption
    ##.  https://dl.acm.org/doi/10.1145/2973750.2985259
    cloud_dev    = {"id": 0, "model": "cloud-device", "type": "CLOUD", "IPT": 5000 * 10 ** 6, "RAM": 40000, "WATT": 0.0}
    fog_dev = {"id": 1, "model": "fog-device", "type": "FOG",
               "IPT": ips_to_ipt(500 * 10 ** 6),
               "RAM": 256,
               "WATT": watt_to_wpt(0.9)}
    mobile_dev = {"id": 2, "model": "mobile-device", "type": "EDGE",
                  "IPT": ips_to_ipt(1.9 * 10 ** 9),
                  "RAM": 2000,
                  "WATT": watt_to_wpt(1.3)}
    smartwatch_dev   = {"id": 3, "model": "smartwatch-device", "type": "IOT",
                        "IPT": ips_to_ipt(768 * 10 ** 6),
                        "RAM": 256,
                        "WATT": watt_to_wpt(0.361)}

    # BLE 4.0/4.1 Modulation Rate: 1 Mb/s, Max Throughput: 0.305 Mb/s
    ##  Data Transmission Efficiency in Bluetooth Low Energy Versions
    ## https://www.mdpi.com/1424-8220/19/17/3746
    link1 = {"s": 3, "d": 2, "BW": 0.305, "PR": 0,
             "WATT_TRANS": watt_to_wpt(0.181), "WATT_RECV":watt_to_wpt(0.174)}  # SW - Mobile

    # 100 Mbit/s = 12,5 Mb/s
    link2 = {"s": 2, "d": 1, "BW": 12.5, "PR": 0,
             "WATT_TRANS": watt_to_wpt(0.654), "WATT_RECV": watt_to_wpt(3.7)} # Mobile - Fog
    link3 = {"s": 1, "d": 0, "BW": 12.5, "PR": 0,
             "WATT_TRANS": watt_to_wpt(4.9), "WATT_RECV": watt_to_wpt(5)} # Fog - Cloud

    topology_json["entity"].append(cloud_dev)
    topology_json["entity"].append(smartwatch_dev)
    topology_json["entity"].append(mobile_dev)
    topology_json["entity"].append(fog_dev)
    topology_json["link"].append(link1)
    topology_json["link"].append(link2)
    topology_json["link"].append(link3)

    t = Topology()
    t.load(topology_json)

    return t


def create_application():
    # APPLICATION
    a = Application(name=APP_NAME)

    # (SmartWatch) --> (Mobile) --> (Fog) --> (Cloud)
    a.set_modules([{"Cloud": {"Type": Application.TYPE_SINK}},
                   {"Fog": {"RAM": 1024, "Type": Application.TYPE_MODULE}},
                   {"Mobile": {"RAM": 1024, "Type": Application.TYPE_MODULE}},
                   {"SmartWatch":{"Type":Application.TYPE_SOURCE}}
                   ])
    """
    Messages among MODULES
    """
    m_sensor_data = Message("M.SW-M", "SmartWatch", "Mobile", instructions=100 * 10 ** 6, bytes=100000)
    m_state = Message("M.M-F", "Mobile", "Fog", instructions=0, bytes=20000)
    m_state_to_cloud = Message("M.F-C", "Fog", "Cloud", instructions=0, bytes=20000)

    """
    Defining which messages will be dynamically generated # the generation is controlled by Population algorithm
    """
    # deterministic_distribution for add_service_source
    a.add_source_messages(m_sensor_data)

    """
    MODULES/SERVICES: Definition of Generators and Consumers (AppEdges and TupleMappings in iFogSim)
    """
    # MODULE SERVICES
    a.add_service_module("Mobile", m_sensor_data, m_state, fractional_selectivity, threshold=1.0)
    a.add_service_module("Fog", m_state, m_state_to_cloud, fractional_selectivity, threshold=1.0)

    return a


def main(stop_time, it,folder_results):

    """
    TOPOLOGY
    """
    t = create_topology()

    print(t.G.nodes()) # nodes id can be str or int

    # Plotting the graph
    pos=nx.spring_layout(t.G)
    nx.draw_networkx(t.G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(t.G, pos,alpha=0.5,font_size=5,verticalalignment="top")


    """
    APPLICATION or SERVICES
    """
    # dataApp = json.load(open('data/appDefinition.json'))
    # apps = create_applications_from_json(dataApp)
    app = create_application()

    """
    SERVICE PLACEMENT 
    """
    placementJson = {
        "initialAllocation": [
            {"app": APP_NAME, "module_name": "Fog", "id_resource": 1},
            {"app": APP_NAME, "module_name": "Mobile", "id_resource": 2},
        ]
    }
    placement = JSONPlacement(name="Placement", json=placementJson)

    
    """
    POPULATION algorithm
    """
    pop = Statical("Statical")
    #For each type of sink modules we set a deployment on some type of devices
    #A control sink consists on:
    #  args:
    #     model (str): identifies the device or devices where the sink is linked
    #     number (int): quantity of sinks linked in each device
    #     module (str): identifies the module from the app who receives the messages
    pop.set_sink_control({"model":"cloud-device", "number":1, "module":app.get_sink_modules()})

    #In addition, a source includes a distribution function:
    dDistribution = deterministic_distribution(name="Deterministic", time=1)
    pop.set_src_control({"model": "smartwatch-device", "number":1, "message": app.get_message("M.SW-M"), "distribution": dDistribution})

    # populationJSON = {
    #     "sinks": [
    #         {"app": APP_NAME, "module_name": "Cloud", "id_resource": 0},
    #     ],
    #     "sources":[
    #         {"app": APP_NAME, "message":"M.SW-M", "time":100,"id_resource":3},
    #     ]

    # }
    # pop = JSONPopulation(name="Statical",json=populationJSON,iteration=0)

    
    """
    Defining ROUTING algorithm to define how path messages in the topology among modules
    """
    selectorPath = First_ShortestPath()

    
    """
    SIMULATION ENGINE
    """
    s = Sim(t, default_results_path=folder_results+"sim_trace")

    
    """
    Deploy services == APP's modules
    """
    s.deploy_app2(app, placement, pop, selectorPath)
    
    
    """
    RUNNING
    """
    logging.info(" Performing simulation: %i " % it)
    s.run(stop_time)  # To test deployments put test_initial_deploy a TRUE
    s.print_debug_assignaments()

    s1 = Stats(defaultPath=os.path.join(os.getcwd(), folder_results, "sim_trace"))
    s1.showResults(total_time=stop_time, topology=t, multiplier=1440)


if __name__ == '__main__':
    LOGGING_CONFIG = Path(__file__).parent / 'logging.ini'
    logging.config.fileConfig(LOGGING_CONFIG)

    folder_results = Path("results/")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results)+"/"

    nIterations = 1  # iteration for each experiment
    simulationDuration = 60

    # Iteration for each experiment changing the seed of randoms
    for iteration in range(nIterations):
        random.seed(iteration)
        logging.info("Running experiment it: - %i" % iteration)

        start_time = time.time()
        main(stop_time=simulationDuration,
             it=iteration,folder_results=folder_results)

        print("\n--- %s seconds ---" % (time.time() - start_time))

    print("Simulation Done!")
  
    # Analysing the results. 
    dfl = pd.read_csv(folder_results+"sim_trace"+"_link.csv")
    print("Number of total messages between nodes: %i"%len(dfl))

    df = pd.read_csv(folder_results+"sim_trace.csv")
    print("Number of requests handled by deployed services: %i"%len(df))

    dfapp2 = df[df.app == 2].copy() # a new df with the requests handled by app 2
    print(dfapp2.head())
    
    dfapp2.loc[:,"transmission_time"] = dfapp2.time_emit - dfapp2.time_reception # Transmission time
    dfapp2.loc[:,"service_time"] = dfapp2.time_out - dfapp2.time_in

    print("The average service time of app2 is: %0.3f "%dfapp2["service_time"].mean())

    print("The app2 is deployed in the folling nodes: %s"%np.unique(dfapp2["TOPO.dst"]))
    print("The number of instances of App2 deployed is: %s"%np.unique(dfapp2["DES.dst"]))
    
    # -----------------------
    # PLAY WITH THIS EXAMPLE!
    # -----------------------
    # Add another app2-instance in allocDefinition.json file adding the next data and run the main.py file again to see the new results:
    # {
    #   "module_name": "2_01",
    #   "app": 2,
    #   "id_resource": 3
    # },
    ## What has happened to the results? Take a look at the network image available in the results folder to understand the "allocation" of app2-related entities.
    
    # ! IMPORTANT. The scheduler & routing algorithm (aka. selectorPath = DeviceSpeedAwareRouting()) chooses the instance that will attend the request according to the latency -in this case-.
    #  For that reason, the initial instance deployed at node 0 is not used. It is further away than the instance located at node3.
    # Add another app2-user at node 16, add the next json inside of userDefinition.json file and try again. Enjoy it! 
    # {
    #   "id_resource": 16,
    #   "app": 2,
    #   "message": "M.USER.APP.2",
    #   "lambda": 100
    # },