# IMPORTS
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import optparse
from tqdm import tqdm
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mplcursors

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

# NEURAL NETWORK
class TrafficController(nn.Module):
    def __init__(self, lr, input_size, hidden1_size, hidden2_size, output_size):
        super(TrafficController, self).__init__()
        self.lr = lr
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        self.hidden1 = nn.Linear(self.input_size, self.hidden1_size)
        self.hidden2 = nn.Linear(self.hidden1_size, self.hidden2_size)                             
        self.output = nn.Linear(self.hidden2_size, self.output_size)  

        # self.hidden1_A = nn.Linear(self.input_size, self.hidden1_size)
        # self.output_A = nn.Linear(self.hidden1_size, self.output_size)                                         
        # self.hidden2_A = nn.Linear(self.hidden1_size, self.hidden2_size)                             
        # self.output_A = nn.Linear(self.hidden2_size, self.output_size)   

        # self.hidden1_v = nn.Linear(self.input_size, 1)
                              

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
    def __init__(self, gamma, epsilon, lr, input_size, hidden1_size, hidden2_size, output_size, batch_size, lights, max_memory_size=100000, eps_end=0.01, eps_dec=5e-4):
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
        self.lights = lights
        self.mem_size = max_memory_size
        self.mem_cntr = 0
        self.batch_size = batch_size

        self.q_eval = TrafficController(self.lr, self.input_size, self.hidden1_size, self.hidden2_size, self.output_size)

        self.memory = dict()
        for light in lights:
            self.memory[light] = {
                'state': np.zeros((self.mem_size, self.input_size), dtype=np.float32),
                'new_state': np.zeros((self.mem_size, self.input_size), dtype=np.float32),
                'action': np.zeros(self.mem_size, dtype=np.int32),
                'reward': np.zeros(self.mem_size, dtype=np.float32),
                'terminal': np.zeros(self.mem_size, dtype=np.bool_),
                'mem_cntr': 0
            }

    def store_transition(self, state, action, reward, state_, done, light):
        index = self.memory[light]['mem_cntr'] % self.mem_size
        self.memory[light]['state'][index] = state
        self.memory[light]['new_state'][index] = state_
        self.memory[light]['reward'][index] = reward
        self.memory[light]['action'][index] = action
        self.memory[light]['terminal'][index] = done
        self.memory[light]['mem_cntr'] += 1

    def sample_buffer(self,batch_size, light):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.memory[light]['state'][batch]
        actions = self.memory[light]['action'][batch]
        rewards = self.memory[light]['reward'][batch]
        new_states = self.memory[light]['new_state'][batch]
        dones = self.memory[light]['terminal'][batch]

        return states, actions, rewards, new_states, dones 

    def choose_action(self, observation):
        # actions = None
        # state = torch.tensor([observation], dtype=torch.float32).to(self.q_eval.device)
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float32).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            # print(f"Q-values: {actions}")
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        # print(f"Q-values: {actions}, chosen action: {action}")
        return action

    def learn(self, light):
        self.q_eval.optim.zero_grad()
        if self.memory[light]['mem_cntr'] < self.batch_size:
            return
        # max_mem = min(self.memory[light]['mem_cntr'], self.mem_size)
        # batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch = np.arange(self.memory[light]['mem_cntr'], dtype=np.int32)
        #batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = self.memory[light]['state'][batch]
        action_batch = self.memory[light]['action'][batch]
        reward_batch = self.memory[light]['reward'][batch]
        new_state_batch = self.memory[light]['new_state'][batch]
        done_batch = self.memory[light]['terminal'][batch]

        state_batch = torch.tensor(state_batch).to(self.q_eval.device)
        action_batch = torch.tensor(action_batch).to(self.q_eval.device)
        reward_batch = torch.tensor(reward_batch).to(self.q_eval.device)
        new_state_batch = torch.tensor(new_state_batch).to(self.q_eval.device)
        done_batch = torch.tensor(done_batch).to(self.q_eval.device)

        # q_eval = self.q_eval.forward(state_batch)[batch_index, action_batch]
        q_eval = self.q_eval.forward(state_batch)[batch, action_batch]
        q_next = self.q_eval.forward(new_state_batch)
        q_next[done_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optim.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

# MAIN FUNCTION
def main():
    # START SUMO FOR INITIAL CONFIGURATION
    sumoBinary = checkBinary('sumo')
    # PICK CONFIGURATION FILE
    # traci.start([sumoBinary, "-c", "Data\Test2\SmallGrid.sumocfg", "--no-warnings"])
    traci.start([sumoBinary, "-c", "Data\Test4\BigGridTest.sumocfg", "--no-warnings"])
    
    # DEFINE NETWORK PARAMETERS
    waiting_time = list()
    waiting_ammt = list()
    epochs = 200
    
    lights = traci.trafficlight.getIDList() 
    print(f"Light IDs: {lights}")
    light_num = list(range(len(lights)))
    print(f"Light Numbers: {light_num}")
    end_time = traci.simulation.getEndTime()
    max_state_size = 24
    agent = TrafficAgent(
        gamma=0.99, 
        epsilon=1.0, 
        lr=0.001, 
        input_size=max_state_size, 
        hidden1_size=256, 
        hidden2_size=256, 
        output_size=12, 
        batch_size=64, 
        lights=light_num)
    traci.close()

    # TRAIN MODEL
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # SELECT CONFIGURATION FILE FOR SIMULATION
        # traci.start([sumoBinary, "-c", "Data\Test2\SmallGrid.sumocfg", "--no-warnings"])
        traci.start([sumoBinary, "-c", "Data\Test4\BigGridTest.sumocfg", "--no-warnings"])

        # ACTIONS FOR 4-WAY INTERSECTIONS
        actions_4_way = [
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
        # ACTIONS FOR 3-WAY INTERSECTIONS
        actions_3_way = [
            [60, "GGgrrGGG", 5, "yyyrryyy"],
            [60, "rrrGGGrr", 5, "rrryyyrr"],
            [50, "GGgrrGGG", 5, "yyyrryyy"],
            [50, "rrrGGGrr", 5, "rrryyyrr"],
            [40, "GGgrrGGG", 5, "yyyrryyy"],
            [40, "rrrGGGrr", 5, "rrryyyrr"],
            [30, "GGgrrGGG", 5, "yyyrryyy"],
            [30, "rrrGGGrr", 5, "rrryyyrr"],
            [20, "GGgrrGGG", 5, "yyyrryyy"],
            [20, "rrrGGGrr", 5, "rrryyyrr"],
            [10, "GGgrrGGG", 5, "yyyrryyy"],
            [10, "rrrGGGrr", 5, "rrryyyrr"]
        ]

        # INITIALIZE VARIABLES AND LISTS
        step = 0
        wait_total = []
        count_total = []
        prev_state = dict()
        light_times = dict()
        current_duration = dict()
        current_phase = dict()
        initial_yellow_phase = dict()
        prev_action = dict()
        action = 999999999

        for light_id, light in enumerate(lights):
            # GET INITIAL PHASE DURATION; SET STATES TO 0 BEFORE SIMULATION
            light_times[light] = traci.trafficlight.getPhaseDuration(light)
            prev_state[light_id] = 0
            prev_action[light_id] = 0

            # GET CURRENT PHASE DURATION AND STATE
            current_duration[light] = traci.trafficlight.getPhaseDuration(light)
            current_phase[light] = traci.trafficlight.getRedYellowGreenState(light)

            # DEFINE INITIAL YELLOW PHASE 
            initial_yellow_phase[light] = current_phase[light].replace('G', 'y').replace('g', 'y')

        while step <= end_time:
            
            # SIMULATION STEP
            traci.simulationStep()
            step += 1

            for light_id, light in enumerate(lights):
                # MAIN SIGNALIZED EDGES
                target_edges = incoming_cont_edges(light)
                
                # SELECT ACTIONS BASED ON NUMBER OF EDGES
                if len(target_edges) == 4:
                    actions = actions_4_way
                else:
                    actions = actions_3_way
                    
                # GET QUEUE INFORMATION
                vehicles_per_edge, vehicle_wait_time, max_wait_time = queue_info(target_edges)

                # GET TOTAL VEHICLES AND MAX WAIT TIME
                vehicle_total = sum(vehicles_per_edge.values())
                count_total.append(vehicle_total)

                max_wait = sum(max_wait_time.values())
                wait_total.append(max_wait)

                # SURROUNDING SIGNALIZED EDGES
                surrounding_edges = surrounding_cont_edges(light, lights)
                S_vehicles_per_edge, S_vehicle_wait_time, S_max_wait_time = queue_info(surrounding_edges)
                S_vehicle_total = sum(S_vehicles_per_edge.values())
                S_max_wait = sum(S_max_wait_time.values())

                light_times[light] -= 1

                # IF LIGHT IS YELLOW AND TIME IS UP, SELECT ACTION
                if light_times[light] == 0 and 'y' in current_phase[light]:
                    # GET STATE VALUES IN FORM [Edge1_value, Edge2_value, ...]
                    state_ = list(vehicles_per_edge.values()) + list(max_wait_time.values()) \
                                + list(S_vehicles_per_edge.values()) + list(S_max_wait_time.values())
                    state_ += [0] * (max_state_size - len(state_))
                    state = prev_state[light_id]
                    prev_state[light_id] = state_

                    # REWARD FUNCTION WITH VARYING WEIGHTS ON EACH VALUE
                    reward = -1 * (round(1*max_wait + 1*vehicle_total + 0.05*S_max_wait + 0.05*S_vehicle_total, 2))
                    # print(f"Reward: {reward}")

                    # STORE TRANSITION
                    agent.store_transition(state, prev_action[light_id], reward, state_, (step==end_time), light_id)

                    # CHOOSE ACTION
                    action = agent.choose_action(state_)
                    prev_action[light_id] = action

                    # ADJUST TRAFFIC LIGHTS
                    adjust_traffic_light(light, actions[action][0], actions[action][1])
                    current_duration[light] = actions[action][0]
                    current_phase[light] = actions[action][1]
                    light_times[light] = current_duration[light] - 1

                    # LEARN
                    agent.learn(light_id)
                    continue

                # IF LIGHT IS GREEN AND TIME IS UP AND NOT IN INITIAL STATE, SELECT CORRESPONDING YELLOW LIGHT ACTION   
                elif 'G' in current_phase[light] and light_times[light] == 0 and action != 999999999 or \
                         'g' in current_phase[light] and light_times[light] == 0 and action != 999999999:
                    adjust_traffic_light(light, actions[action][2], actions[action][3])  
                    current_phase[light] = actions[action][3]
                    light_times[light] = actions[action][2] - 1
                    continue
                
                # IF LIGHT IS GREEN AND TIME IS UP AND IN INITIAL STATE, SELECT INITIAL YELLOW LIGHT ACTION
                elif ('G' in current_phase[light] and light_times[light] == 0 and action == 999999999) or \
                       ('g' in current_phase[light] and light_times[light] == 0 and action == 999999999):
                    adjust_traffic_light(light, 5, initial_yellow_phase[light])
                    current_phase[light] = initial_yellow_phase[light]
                    light_times[light] = 5 - 1
                    continue

        # GET AVERAGES PERFORMANCE EVALUATION
        average_max_wait = mean(wait_total)
        average_count = mean(count_total)
        # CLOSE SUMO FOR NEXT EPOCH
        traci.close()

        # APPEND AVERAGES FOR PLOT
        waiting_time.append(average_max_wait)
        waiting_ammt.append(average_count)
        print(f"Average waiting time: {average_max_wait} | Average vehicle count: {average_count}")
        print("\n")

    # PLOT RESULTS
    fig, ax = plt.subplots(2)
    x = range(1, len(waiting_time) +1)

    ax[0].plot(x, waiting_time)
    ax[0].set_xlabel("Epoch Number")
    ax[0].set_ylabel("Avg. Waiting Time")
    ax[0].set_title(f"Total Waiting Time Over {epochs} Epochs")

    ax[1].plot(x, waiting_ammt)
    ax[1].set_xlabel("Epoch Number")
    ax[1].set_ylabel("Total Waiting Amount")
    ax[1].set_title(f"Total Waiting Amount Over {epochs} Epochs")

    fig.suptitle("Traffic Light Control with Neural Networks")
    # plt.savefig('Data/Figures/First_NN_Test_100_Epochs.png')
    plt.subplots_adjust(hspace=0.5)
    mplcursors.cursor(hover=True)
    plt.show()

# RUN MAIN FUNCTION
if __name__ == "__main__":
    main()