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

    return surrounding_edges 

# FUNCTION TO GET QUEUE INFORMATION FOR A GIVEN EDGE
def queue_info(edges):
    vehicles_per_edge = dict()
    vehicle_id = dict()
    vehicle_wait_time = dict()
    max_wait_time = dict()
    for i in edges:
        vehicles_per_edge[i] = 0
        vehicle_wait_time[i] = 0
        max_wait_time[i] = 0
        vehicles_per_edge[i] = traci.edge.getLastStepHaltingNumber(i)
        vehicle_id[i] = traci.edge.getLastStepVehicleIDs(i)
        for v in vehicle_id[i]:
            current_wait_time = traci.vehicle.getWaitingTime(v)
            vehicle_wait_time[i] += current_wait_time
            if current_wait_time > max_wait_time[i]:
                max_wait_time[i] = current_wait_time

    return vehicles_per_edge, vehicle_wait_time, max_wait_time

# ADJUST TRAFFIC LIGHTS
def adjust_traffic_light(junction, junc_time, junc_state):
    traci.trafficlight.setRedYellowGreenState(junction, junc_state)
    traci.trafficlight.setPhaseDuration(junction, junc_time)

## NEURAL NETWORK
class TrafficController(nn.Module):
    def __init__(self, lr, input_size, hidden1_size, hidden2_size, output_size):
        super(TrafficController, self).__init__()
        self.lr = lr
        self.input_size = input_size
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
    def __init__(self, gamma, epsilon, lr, input_size, hidden1_size, hidden2_size, output_size, batch_size, max_memory_size=100000, eps_end=0.01, eps_dec=5e-4):
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
        # max_memory_size = 100000
        self.mem_size = max_memory_size
        self.mem_cntr = 0
        self.batch_size = batch_size

        self.q_eval = TrafficController(self.lr, self.input_size, self.hidden1_size, self.hidden2_size, self.output_size)
        # self.q_eval = TrafficController(self.lr, input_size = input_size, hidden1_size=256, hidden2_size=256, output_size=output_size)

        self.state_memory = np.zeros((self.mem_size, self.input_size), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.input_size), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done, light):
        index = self.mem_cntr % self.mem_size
        # self.state_memory[light][index] = state
        # self.new_state_memory[light][index] = state_
        # self.reward_memory[light][index] = reward
        # self.action_memory[light][index] = action
        # self.terminal_memory[light][index] = done
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float32).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    # def calculate_duration(self, light, action):
    #     min_duration = 5
    #     max_duration = 60
    #     calc = 
    #     duration = min_duration + calc * (max_duration - min_duration)
    #     return duration

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

    agent = TrafficAgent(gamma=0.99, epsilon=1.0, lr=0.001, input_size=16, hidden1_size=256, hidden2_size=256, output_size=12, batch_size=64)
    # DEFINE NETWORK PARAMETERS
    scores, eps_history = [], []
    epochs = 5
    
    lights = traci.trafficlight.getIDList() 
    print(f"Light IDs: {lights}")
    end_time = traci.simulation.getEndTime()
    # traci.close()
    # TRAIN MODEL
    for epoch in range(epochs):
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
        # sumoBinary = checkBinary('sumo')
        # traci.start([sumoBinary, "-c", "Data\Test2\SmallGrid.sumocfg"])
        actions = [
            [60, "rrrrGGGgrrrrGGGg", 5, "rrrryyyyrrrryyyy"],
            [60, "GGGgrrrrGGGgrrrr", 5, "yyyyrrrryyyyrrrr"],
            [50, "rrrrGGGgrrrrGGGg", 5, "rrrryyyyrrrryyyy"],
            [50, "GGGgrrrrGGGgrrrr", 5, "yyyyrrrryyyyrrrr"],
            [40, "rrrrGGGgrrrrGGGg", 5, "rrrryyyyrrrryyyy"],
            [40, "GGGgrrrrGGGgrrrr", 5, "yyyyrrrryyyyrrrr"],
            [30, "rrrrGGGgrrrrGGGg", 5, "rrrryyyyrrrryyyy"],
            [30, "GGGgrrrrGGGgrrrr", 5, "yyyyrrrryyyyrrrr"],
            [20, "rrrrGGGgrrrrGGGg", 5, "rrrryyyyrrrryyyy"],
            [20, "GGGgrrrrGGGgrrrr", 5, "yyyyrrrryyyyrrrr"],
            [10, "rrrrGGGgrrrrGGGg", 5, "rrrryyyyrrrryyyy"],
            [10, "GGGgrrrrGGGgrrrr", 5, "yyyyrrrryyyyrrrr"]
        ]
        step = 0
        total_loss = 0
        num_iters = 0

        prev_state = dict()
        light_times = dict()
        prev_action = dict()

        for light_id, light in enumerate(lights):
            light_times[light] = 0
            prev_state[light_id] = 0
            prev_action[light_id] = 0

        while step <= end_time:
            # SIMULATION STEP
            traci.simulationStep()
            print(f"Step {step}")
            # queue_info(edges)
            for light_id, light in enumerate(lights):
                # MAIN SIGNALIZED EDGES
                target_edges = incoming_cont_edges(light)
                vehicles_per_edge, vehicle_wait_time, max_wait_time = queue_info(target_edges)
                vehicle_total = sum(vehicles_per_edge.values())
                max_wait = sum(max_wait_time.values())

                # SURROUNDING SIGNALIZED EDGES
                surrounding_edges = surrounding_cont_edges(light, lights)
                S_vehicles_per_edge, S_vehicle_wait_time, S_max_wait_time = queue_info(surrounding_edges)
                S_vehicle_total = sum(S_vehicles_per_edge.values())
                S_max_wait = sum(S_max_wait_time.values())

                # GET STATE VALUES IN FORM [Edge1_value, Edge2_value, ...]
                state_ = list(vehicles_per_edge.values()) + list(max_wait_time.values()) \
                            + list(S_vehicles_per_edge.values()) + list(S_max_wait_time.values())
                # print(f"State: {state_}")
                state = prev_state[light_id]
                prev_state[light_id] = state_
                # REWARD FUNCTION WITH VARYING WEIGHTS ON EACH VALUE
                reward = round(-1*max_wait - 0.8*vehicle_total - 0.1*S_max_wait - 0.1*S_vehicle_total, 2)
                # STORE TRANSITION
                agent.store_transition(state, prev_action[light_id], reward, state_, (step==end_time), light_id)
                
                # CHOOSE ACTION
                action = agent.choose_action(state_)
                print(f"Action: {action}")
                prev_action[light_id] = action
                # # ADJUST TRAFFIC LIGHTS
                adjust_traffic_light(light, actions[action][0], actions[action][1])
                adjust_traffic_light(light, actions[action][2], actions[action][3])

                # print(list(max_wait_time.values()))
                # print(f"State: {state}")
                # total_loss += loss.item()
                num_iters += 1
            step += 1
        
        avg_loss = total_loss / num_iters
        avg_losses.append(avg_loss)

        # CLOSE SUMO
    traci.close()

if __name__ == "__main__":
    main()