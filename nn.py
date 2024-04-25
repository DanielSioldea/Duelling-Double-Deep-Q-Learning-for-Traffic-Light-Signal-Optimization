# IMPORTS
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import optparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

# OPTIONS (CUS WHY NOT)
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

# FUNCTION TO GET INCOMING EDGES FOR A GIVEN LIGHT
def incoming_cont_edges(light):
    controlled_links = traci.trafficlight.getControlledLinks(light)
    incoming_lanes = {link[0][0] for link in controlled_links}
    incoming_edges = {lane.split('_')[0] for lane in incoming_lanes if traci.lane.getLinks(lane)}
    # print(f"The incoming edges for light {light} are {incoming_edges}")
    return incoming_edges

# FUNCTION TO GET SURROUNDING EDGES FOR A GIVEN TARGET LIGHT; NEED TO MODIFY TO ONLY GET EDGES THAT
# FEED INTO TARGET LIGHT
def surrounding_cont_edges(target_light, light_list, distance_buffer=1000):
    x1, y1 = traci.junction.getPosition(target_light)
    surrounding_edges = []
    # vehicles_to_target = {}
    for light in light_list:
        if light == target_light:
            continue
        x2, y2 = traci.junction.getPosition(light)
        distance = traci.simulation.getDistance2D(x1, y1, x2, y2)
        if distance < distance_buffer:
            controlled_links = traci.trafficlight.getControlledLinks(light)
            controlled_edges = {link[0][0].split('_')[0] for link in controlled_links}
            surrounding_edges.extend(controlled_edges)
            # for edge in controlled_edges:
            #     vehicles = traci.edge.getLastStepVehicleIDs(edge)
            #     for v_id in vehicles:
            #         current_edge = traci.vehicle.getRoadID(v_id)
            #         if current_edge in incoming_cont_edges(target_light):
            #             vehicles_to_target[edge] = vehicles_to_target.get(edge, 0) + 1
            #             print(f"Vehicle {v_id} is leaving {current_edge} from {light} to {target_light}")
    return surrounding_edges #, vehicles_to_target

# FUNCTION TO GET NUMBER OF VEHICLES STOPPED, MAX VEH WAITING TIME, AND SUM OF WAITING TIME IN EACH EDGE
def queue_info(edges):
    # vehicles_per_edge = dict()
    # vehicle_id = dict()
    # vehicle_wait_time = dict()
    # max_wait_time = dict()
    info = dict()
    for i in edges:
        info[i] = {
            'vehicles_per_edge': 0,
            'vehicle_wait_time': 0,
            'max_wait_time': 0
        }
        # vehicles_per_edge[i] = 0
        # vehicle_wait_time[i] = 0
        # max_wait_time[i] = 0
        # vehicles_per_edge[i] = traci.edge.getLastStepHaltingNumber(i)
        info[i]['vehicles_per_edge'] = traci.edge.getLastStepHaltingNumber(i)
        vehicle_id = traci.edge.getLastStepVehicleIDs(i)
        for v in vehicle_id:
            current_wait_time = traci.vehicle.getWaitingTime(v)
            # vehicle_wait_time[i] += current_wait_time
            # if current_wait_time > max_wait_time[i]:
            #     max_wait_time[i] = current_wait_time
            info[i]['vehicle_wait_time'] += current_wait_time
            if current_wait_time > info[i]['max_wait_time']:
                info[i]['max_wait_time'] = current_wait_time
    return info #vehicles_per_edge, vehicle_wait_time, max_wait_time

# ADJUST TRAFFIC LIGHTS
def adjust_traffic_light(junction, junc_time, junc_state):
    traci.trafficlight.setRedYellowGreenState(junction, junc_state)
    traci.trafficlight.setPhaseDuration(junction, junc_time)

## NEURAL NETWORK
class TrafficController(nn.Module):
    def __init__(self, input_size, lr, hidden1_size, hidden2_size, output_size):
        super(TrafficController, self).__init__()
        self.input_size = input_size
        self.lr = lr
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        #self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(self.input_size, self.hidden1_size)      # FIRST HIDDEN LAYER HAS 12 NEURONS
        # self.activ1 = nn.ReLU()                                         # ReLU ACTIVATION FUNCTION
        self.hidden2 = nn.Linear(self.hidden1_size, self.hidden2_size)    # SECOND HIDDEN LAYER HAS 8 NEURONS
        # self.activ2 = nn.ReLU()                                         # ReLU ACTIVATION FUNCTION
        self.output = nn.Linear(self.hidden2_size, self.output_size)      # OUTPUT LAYER HAS ONE NEURON
        # self.activ_out = nn.Sigmoid()                                   # SIGMOID ACTIVATION FUNCTION (ENSURES OUTPUT BETWEEN 0 AND 1)

        # OPTIMIZER AND LOSS FUNCTION (TEST DIFFERENT OPTIONS)
        self.optim = optim.Adam(self.parameters(), lr = self.lr)
        self.loss = nn.MSELoss()
        # SELECT GPU OR CPU DEPENDING ON WHAT IS AVAILABLE
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        print(f"Network is using {self.device} device.")
    
    # DEFINE HOW NN FEEDS FORWARD INTO LAYERS
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        actions = self.output(x)
        return actions
    
## DEFINE AGENT CLASS FOR LEARNING
class TrafficAgent:
    def __init__(self, gamma, epsilon, lr, input_size, hidden1_size, hidden2_size, output_size, max_memory_size, batch_size, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.action_space = [i for i in range(output_size)]
        max_memory_size = 100000
        self.mem_size = max_memory_size
        self.mem_cntr = 0
        self.batch_size = batch_size

        self.q_eval = TrafficController(self.lr, self.input_size, self.hidden1_size, self.hidden2_size, self.output_size)

        self.state_memory = np.zeros((self.mem_size, self.input_size), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.input_size), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done, light):
        index = self.mem_cntr % self.mem_size
        self.state_memory[light][index] = state
        self.new_state_memory[light][index] = state_
        self.reward_memory[light][index] = reward
        self.action_memory[light][index] = action
        self.terminal_memory[light][index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float32).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def learn(self):
        self.q_eval.optim.zero_grad()
        if self.mem_cntr < self.batch_size:
            return
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        done_batch = self.terminal_memory[batch]

        state_batch = torch.tensor(state_batch).to(self.q_eval.device)
        action_batch = torch.tensor(action_batch).to(self.q_eval.device)
        reward_batch = torch.tensor(reward_batch).to(self.q_eval.device)
        new_state_batch = torch.tensor(new_state_batch).to(self.q_eval.device)
        done_batch = torch.tensor(done_batch).to(self.q_eval.device)

        # q_eval = self.q_eval.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        q_eval = self.q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.q_eval.forward(new_state_batch)
        q_next[done_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optim.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

# MAIN FUNCTION
def main():
    # GET OPTIONS
    avg_losses = []
    # START SUMO FOR INITIAL CONFIGURATION
    sumoBinary = checkBinary('sumo')
    traci.start([sumoBinary, "-c", "Data\Test2\SmallGrid.sumocfg"])

    agent = TrafficAgent(gamma=0.99, epsilon=1.0, lr=0.001, input_size=4, hidden1_size=256, hidden2_size=256, output_size=4, batch_size=64)
    # DEFINE NETWORK PARAMETERS
    scores, eps_history = [], []
    epochs = 1
    
    lights = traci.trafficlight.getIDList() 
    print(f"Light IDs: {lights}")
    end_time = traci.simulation.getEndTime()
    # print(f"End time: {end_time}")

    traci.close()
    # TRAIN MODEL
    for epoch in range(epochs):
        traci.start([sumoBinary, "-c", "Data\Test2\SmallGrid.sumocfg"])
        light_choice = [
            ["rrrrGGGgrrrrGGGg", "rrrryyyyrrrryyyy"],
            ["GGGgrrrrGGGgrrrr", "yyyyrrrryyyyrrrr"] 
        ]
        step = 0
        total_loss = 0
        num_iters = 0

        prev_queue_time = dict()
        prev_queue_length = dict()
        light_times = dict()
        prev_action = dict()

        for light_id, light in enumerate(lights):
            light_times[light] = 0
            prev_queue_time[light] = 0
            prev_queue_length[light] = 0
            prev_action[light_id] = 0
            # target_edges = incoming_cont_edges(light)
            # print(f"Edges into light {light}: {target_edges}")
            # surrounding_edges = surrounding_cont_edges(light, lights)
            # print(f"Surrounding edges for light {light}: {surrounding_edges}")

        while step <= end_time:
            # SIMULATION STEP
            traci.simulationStep()
            print(f"Step {step}")
            # queue_info(edges)
            for light_id, light in enumerate(lights):
                # MAIN SIGNALIZED EDGES
                target_edges = incoming_cont_edges(light)
                # SURROUNDING SIGNALIZED EDGES
                surrounding_edges = surrounding_cont_edges(light, lights)

                # GET STATE
                state = {**queue_info(target_edges), **queue_info(surrounding_edges)}
                # state = {**queue_info(target_edges)}
                # state = {**queue_info(surrounding_edges)}
                print(f"State: {state}")


                


                # total_loss += loss.item()
                num_iters += 1
            step += 1
            

            # current_state = traci.trafficlight.getRedYellowGreenState(junctions[0])
            # print(f"Current state: {current_state}")
        
        avg_loss = total_loss / num_iters
        avg_losses.append(avg_loss)

            # PRINT LOSS
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")

        # CLOSE SUMO
    traci.close()

if __name__ == "__main__":
    main()