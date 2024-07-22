# IMPORTS
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import optparse
from tqdm import tqdm
from statistics import mean
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import action_list as al
import action_list_TLS

from nn_targ import TrafficController, TrafficAgent

# SET UP SUMO TRACI CONNECTION
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# TRACI IMPORTS (IDK IF THIS NEEDS TO BE HERE OR CAN BE AT THE TOP)
from sumolib import checkBinary
import traci
import traci.constants as tc

# FUNCTION TO GET INCOMING EDGES FOR A GIVEN LIGHT
def incoming_cont_edges(light):
    controlled_links = traci.trafficlight.getControlledLinks(light)
    incoming_lanes = {link[0][0] for link in controlled_links}
    incoming_edges = {lane.split('_')[0] for lane in incoming_lanes if traci.lane.getLinks(lane)}
    return incoming_edges

# FUNCTION TO GET SURROUNDING EDGES FOR A GIVEN TARGET LIGHT; IF LIGHT IS WITHIN 500 METERS OF TARGET LIGHT, ADD TO LIST
def surrounding_cont_edges(target_light, light_list, distance_buffer=500):
    x1, y1 = traci.junction.getPosition(target_light)
    surrounding_edges = []
    for light in light_list:
        if light == target_light:
            continue
        x2, y2 = traci.junction.getPosition(light)
        distance = traci.simulation.getDistance2D(x1, y1, x2, y2)
        if distance < distance_buffer:
            controlled_links = traci.trafficlight.getControlledLinks(light)
            controlled_edges = {link[0][0].split('_')[0] for link in controlled_links}
            surrounding_edges.extend(controlled_edges)

    return surrounding_edges 

# FUNCTION TO GET QUEUE INFORMATION FOR A GIVEN EDGE
def queue_info(edges):
    vehicles_per_edge_NS = vehicles_per_edge_EW = 0
    max_wait_time_NS = max_wait_time_EW = 0

    for i in edges:
        vehicle_id = traci.edge.getLastStepVehicleIDs(i)
        for v in vehicle_id:
            lane_id = traci.vehicle.getLaneID(v)
            lane_len = traci.lane.getLength(lane_id)
            veh_position = traci.vehicle.getLanePosition(v)
            if lane_len - veh_position <= 50:
                lane_shape = traci.lane.getShape(lane_id)
                start, end = lane_shape[0], lane_shape[-1]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                angle = math.atan2(dy, dx)  # Angle in radians
                angle = math.degrees(angle) % 360

                direction = None
                if 45 <= angle < 135:
                    direction = "N"
                elif 135 <= angle < 225:
                    direction = "W"
                elif 225 <= angle < 315:
                    direction = "S"
                else:
                    direction = "E"

                current_wait_time = traci.vehicle.getWaitingTime(v)
                if direction in ["N", "S"]:
                    vehicles_per_edge_NS += 1
                    if current_wait_time > max_wait_time_NS:
                        max_wait_time_NS = current_wait_time
                else:  # direction in ["E", "W"]
                    vehicles_per_edge_EW += 1
                    if current_wait_time > max_wait_time_EW:
                        max_wait_time_EW = current_wait_time

    return vehicles_per_edge_NS, vehicles_per_edge_EW, max_wait_time_NS, max_wait_time_EW

# ADJUST TRAFFIC LIGHTS
def adjust_traffic_light(junction, junc_time, junc_state):
    traci.trafficlight.setRedYellowGreenState(junction, junc_state)
    traci.trafficlight.setPhaseDuration(junction, junc_time)

def closest_loops(light):
    loop_ids = traci.inductionloop.getIDList()
    light_pos = traci.junction.getPosition(light)
    close_loops = {}
    close_loops[light] = []
    for loop_id in loop_ids:
        lane_id = traci.inductionloop.getLaneID(loop_id)
        loop_pos = traci.inductionloop.getPosition(loop_id)
        lane_shape = traci.lane.getShape(lane_id)

        lane_len = traci.lane.getLength(lane_id)
        ratio = loop_pos / lane_len
        loop_pos = (
            lane_shape[0][0] * (1 - ratio) + lane_shape[-1][0] * ratio,
            lane_shape[0][1] * (1 - ratio) + lane_shape[-1][1] * ratio
        )

        distance = math.sqrt((light_pos[0] - loop_pos[0])**2 + (light_pos[1] - loop_pos[1])**2)
        direction = None
        if distance < 25:
            start, end = lane_shape[0], lane_shape[-1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            angle = math.atan2(dy, dx)  # Angle in radians
            angle = math.degrees(angle) % 360
        
            if 45 <= angle < 135:
                direction = "N"
            elif 135 <= angle < 225:
                direction = "W"
            elif 225 <= angle < 315:
                direction = "S"
            else:
                direction = "E"

            close_loops[light].append((loop_id, direction))
    return close_loops

def main():
    sumoBinary = checkBinary('sumo-gui')
    # traci.start([sumoBinary, "-c", "Data\Test4\BigGridTest.sumocfg", "--no-warnings"])
    # traci.start([sumoBinary, "-c", "Data\Test5\Rymal-upperRedHill.sumocfg", "--no-warnings"])
    # traci.start([sumoBinary, "-c", "Data\TestCARLA\TwoIntersectionTest_Test.sumocfg", "--no-warnings"])
    # traci.start([sumoBinary, "-c", "Data\ThesisMap\ThesisMap.sumocfg", "--no-warnings"])
    traci.start([sumoBinary, "-c", "Data\ThesisMapFull\ThesisMapFull_Draft2.sumocfg", "--no-warnings"])

    lights = traci.trafficlight.getIDList()
    specific_lights = {'26', '85', '142', '68', '42'}

    light_num = list(range(len(lights)))
    end_time = traci.simulation.getEndTime()
    target_edges = dict()
    surrounding_edges = dict()  
    current_phase = dict()
    actions = dict()
    action = None
    light_times = dict()
    initial_yellow_phase = dict()
    eastWest = dict()
    northSouth = dict()
    print(f"Light IDs: {lights}")
    print(f"Light Numbers: {light_num}")
    print(f"End Time: {end_time}")

    max_state_size = 24
    agent = TrafficAgent(
        gamma=0.99, 
        epsilon=1.0, 
        lr=0.001, 
        input_size=max_state_size, 
        hidden1_size=512, 
        hidden2_size=512, 
        output_size=6, 
        batch_size=64, 
        lights=light_num)
    
    model_state_dict = torch.load('Models/ThesisModel_INDIVIDUAL_ACTIONS.bin')
    agent.q_eval.load_state_dict(model_state_dict)
    print("Model loaded")

    for light in specific_lights:
        target_edges[light] = incoming_cont_edges(light)
        surrounding_edges[light] = surrounding_cont_edges(light, specific_lights)
        print(f"Target Edges for light {light}: {target_edges[light]}")
        print(f"Surrounding Edges for light {light}: {surrounding_edges[light]}")
        # GET CURRENT PHASE DURATION AND STATE
        current_phase[light] = traci.trafficlight.getRedYellowGreenState(light)
        print(f"Current Phase for light {light}: {current_phase[light]}")
        light_times[light] = traci.trafficlight.getPhaseDuration(light)
        # DEFINE INITIAL YELLOW PHASE 
        initial_yellow_phase[light] = current_phase[light].replace('G', 'y').replace('g', 'y')
        # SELECT APPROPRIATE ACTION LIST BASED ON LIGHT INDEX COUNT
        eastWest['26'] = action_list_TLS.eastWest26
        northSouth['26'] = action_list_TLS.northSouth26
        eastWest['85'] = action_list_TLS.eastWest85
        northSouth['85'] = action_list_TLS.northSouth85
        eastWest['142'] = action_list_TLS.eastWest142
        northSouth['142'] = action_list_TLS.northSouth142
        eastWest['68'] = action_list_TLS.eastWest68
        northSouth['68'] = action_list_TLS.northSouth68
        eastWest['42'] = action_list_TLS.eastWest42
        northSouth['42'] = action_list_TLS.northSouth42
        
    step = 0

    while step <= end_time:
        traci.simulationStep()
        if traci.simulation.getTime() == 240:
            print(f'EGO Vehicles released.')
        step += 1
        # print(f"Step: {step}")

        for light in specific_lights:
            # GET PHASE OF EACH LIGHT
            light_times[light] -= 1
   
            # TARGET EDGE QUEUE INFORMATION
            vehicles_per_edge_NS, vehicles_per_edge_EW, max_wait_time_NS, max_wait_time_EW = queue_info(target_edges[light])

            # SURROUNDING EDGE QUEUE INFORMATION
            S_vehicles_per_edge_NS, S_vehicles_per_edge_EW, S_max_wait_time_NS, S_max_wait_time_EW = queue_info(surrounding_edges[light])

            # CHECK FOR ANY VEHICLES IN OPPOISTE DIRECTION; IF NONE CONTINUE GREEN FOR CURRENT DIRECTION; IF THERE ARE VEHICLES, CONTINUE
            if current_phase[light][0] == 'G' and light_times[light] == 0:
                    if max_wait_time_NS <= 5:
                        light_times[light] += 3
            if current_phase[light][0] == 'r' and current_phase[light][5] == 'G' and light_times[light] == 0:
                    if max_wait_time_EW <= 5:
                        light_times[light] += 3

            if 'y' in current_phase[light] and light_times[light] == 0:

                # GET STATE OF TRAFFIC LIGHT
                state = [vehicles_per_edge_NS, max_wait_time_NS, vehicles_per_edge_EW,  max_wait_time_EW, \
                        S_vehicles_per_edge_NS, S_max_wait_time_NS, S_vehicles_per_edge_EW, S_max_wait_time_EW]
                state += [0] * (max_state_size - len(state))
                    
                # CONVERT STATE TO TENSOR AND ISOLATE Q-VALUES
                state_tensor = torch.tensor(state, dtype=torch.float32)
                predicted_phase = agent.q_eval(state_tensor)
                q_values = predicted_phase[1]

                # GET ACTION
                action = torch.argmax(q_values).item()

                # ADJUST TRAFFIC LIGHT
                # CHECK WHICH DIRECTION JUST HAD A YELLOW LIGHT, SELECT ACTION FOR OPPOSITE DIRECTION
                if current_phase[light][0] == 'y':
                    actions[light] = northSouth[light]
                else:
                    actions[light] = eastWest[light]
                adjust_traffic_light(light, actions[light][action][0], actions[light][action][1])
                current_phase[light] = actions[light][action][1]
                light_times[light] = actions[light][action][0]

            elif action is not None and 'G' in current_phase[light] and light_times[light] == 0:
                adjust_traffic_light(light, actions[light][action][2], actions[light][action][3])
                current_phase[light] = actions[light][action][3]
                light_times[light] = actions[light][action][2]
                continue
            
            elif action is None and 'G' in current_phase[light] and light_times[light] == 0:
                adjust_traffic_light(light, 5, initial_yellow_phase[light])
                current_phase[light] = initial_yellow_phase[light]
                light_times[light] = 5
                continue

    traci.close()

if __name__ == "__main__":
    main()