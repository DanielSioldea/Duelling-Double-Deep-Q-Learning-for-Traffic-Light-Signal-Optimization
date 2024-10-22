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
    # print(f"The incoming lanes for light {light} are {incoming_lanes}")
    incoming_edges = {lane.split('_')[0] for lane in incoming_lanes if traci.lane.getLinks(lane)}
    # print(f"The incoming edges for light {light} are {incoming_edges}")
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
    # print(f"Light position: {light_pos}") 
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

# NEURAL NETWORK
class TrafficController(nn.Module):
    def __init__(self, lr, input_size, hidden1_size, hidden2_size, output_size, chkpt_dir, name):
        super(TrafficController, self).__init__()
        self.lr = lr
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.hidden1 = nn.Linear(self.input_size, self.hidden1_size)
        self.hidden2 = nn.Linear(self.hidden1_size, self.hidden2_size)                             
        self.output_A = nn.Linear(self.hidden2_size, self.output_size)   

        self.output_V = nn.Linear(self.hidden1_size, 1)
                              

        # OPTIMIZER AND LOSS FUNCTION (TEST DIFFERENT OPTIONS)
        self.optim = optim.Adam(self.parameters(), lr = self.lr)
        self.loss = nn.MSELoss()
        # SELECT GPU OR CPU DEPENDING ON WHAT IS AVAILABLE
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    # DEFINE HOW NN FEEDS FORWARD INTO LAYERS
    def forward(self, x):
        flat1 = F.relu(self.hidden1(x))
        flat2 = F.relu(self.hidden2(flat1))

        V = self.output_V(flat2)
        A = self.output_A(flat2)
        return V, A
    
    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))
    
## DEFINE AGENT CLASS FOR LEARNING
class TrafficAgent:
    def __init__(self, gamma, epsilon, lr, input_size, hidden1_size, hidden2_size, output_size,
                 batch_size, lights, max_memory_size=100000, eps_end=0.01, eps_dec=5e-4,
                 chkpt_dir = 'tmp/traffic_controller', replace_target_net = 1000):
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
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.chkpt_dir = chkpt_dir
        self.replace_target__net = replace_target_net

        self.q_eval = TrafficController(self.lr, self.input_size, self.hidden1_size, self.hidden2_size,
                                        self.output_size, chkpt_dir=self.chkpt_dir, name='q_eval')
        self.q_next = TrafficController(self.lr, self.input_size, self.hidden1_size, self.hidden2_size,
                                        self.output_size, chkpt_dir=self.chkpt_dir, name='q_next')

        self.memory = dict()
        for light in lights:
            self.memory[light] = {
                'state': np.zeros((self.mem_size, self.input_size), dtype=np.float32),
                'new_state': np.zeros((self.mem_size, self.input_size), dtype=np.float32),
                'action': np.zeros(self.mem_size, dtype=np.int64),
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

    def sample_buffer(self, batch_size, light):
        max_mem = min(self.memory[light]['mem_cntr'], self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.memory[light]['state'][batch]
        actions = self.memory[light]['action'][batch]
        rewards = self.memory[light]['reward'][batch]
        new_states = self.memory[light]['new_state'][batch]
        dones = self.memory[light]['terminal'][batch]

        return states, actions, rewards, new_states, dones 

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float32).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def replace_target_network(self):
        if self.replace_target__net is not None and \
           self.learn_step_counter % self.replace_target__net == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def save_checkpoint(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_checkpoint(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def save_model(self, model_name):
        torch.save(self.q_eval.state_dict(),f'Models/{model_name}.bin')

    def learn(self, light):
        self.q_eval.optim.zero_grad()
        if self.memory[light]['mem_cntr'] < self.batch_size:
            return
        
        self.replace_target_network()

        state, action, reward, new_state, done = self.sample_buffer(self.batch_size, light)

        states_T = torch.tensor(state).to(self.q_eval.device)
        rewards_T = torch.tensor(reward).to(self.q_eval.device)
        dones_T = torch.tensor(done).to(self.q_eval.device)
        actions_T = torch.tensor(action).to(self.q_eval.device)
        new_states_T = torch.tensor(new_state).to(self.q_eval.device)

        indicies = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states_T)
        V_s_, A_s_ = self.q_next.forward(new_states_T)
        V_s_eval, A_s_eval = self.q_eval.forward(new_states_T)

        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indicies, actions_T]
        q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = torch.argmax(q_eval, dim=1)

        q_next[dones_T] = 0.0
        q_target = rewards_T + self.gamma * q_next[indicies, max_actions]
        

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optim.step()
        self.learn_step_counter += 1

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        return loss

# MAIN FUNCTION
def main(train=True, model_name="model", epochs=50):
    # START SUMO FOR INITIAL CONFIGURATION
    sumoBinary = checkBinary('sumo')
    # PICK CONFIGURATION FILE
    # traci.start([sumoBinary, "-c", "Data\Test2\SmallGrid.sumocfg", "--no-warnings"])
    # traci.start([sumoBinary, "-c", "Data\Test4\BigGridTest.sumocfg", "--no-warnings"])
    # traci.start([sumoBinary, "-c", "Data\Test5\Rymal-upperRedHill.sumocfg", "--no-warnings"])
    # traci.start([sumoBinary, "-c", "Data\Test6\StoneChurch.sumocfg", "--no-warnings"])
    # traci.start([sumoBinary, "-c", "Data\TestCARLA\RymalRoadSimplified.sumocfg", "--no-warnings"])
    # traci.start([sumoBinary, "-c", "Data\TestCARLA\TwoIntersectionTest.sumocfg", "--no-warnings"])
    traci.start([sumoBinary, "-c", "Data\ThesisMapFull\ThesisMapFull_Draft2.sumocfg", "--no-warnings"])

    # DEFINE NETWORK PARAMETERS
    waiting_time = list()
    waiting_ammt = list()
    loss_total = list()
    epochs = epochs
    load_checkpoint = False
    closest_loops_dict = {}
    init_action = {}
    best_time = np.inf

    print("can something print here?")
    lights = traci.trafficlight.getIDList() 
    specific_lights = {'26', '85', '142', '68', '42'}
    # print(f"Light IDs: {lights}")
    print("can something print here too?")

    print(f"Light IDs: {lights}")
    get_phase = traci.trafficlight.getPhase(lights[1])
    print(f"Phase: {get_phase}")
    logics = traci.trafficlight.getAllProgramLogics(lights[0])
    print(f"Logics: {logics}")
    light_num = list(range(len(lights)))
    print(f"Light Numbers: {light_num}")
    # end_time = traci.simulation.getEndTime()/0.25
    end_time = traci.simulation.getEndTime()
    print(f"End time: {end_time}")
    test_phase = traci.trafficlight.getRedYellowGreenState(lights[0])
    print(f"Test phase: {test_phase}")
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
    
    for light in specific_lights:
        print(f"Specific lights are {light}")
        # GET CLOSEST INDUCTION LOOPS
        close_loop = closest_loops(light)
        closest_loops_dict[light] = close_loop[light]
        print(f"The closest loops for light {light} are {closest_loops_dict[light]}")

    if not train:
        agent.q_eval.load_state_dict(torch.load(f'Models/{model_name}.bin', map_location=agent.q_eval.device))

        
    traci.close()

    if load_checkpoint:
        agent.load_checkpoint()

    # TRAIN MODEL
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # OPEN GUI FOR TESTING
        if train:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')

        # SELECT CONFIGURATION FILE FOR SIMULATION
        # traci.start([sumoBinary, "-c", "Data\Test2\SmallGrid.sumocfg", "--no-warnings"])
        # traci.start([sumoBinary, "-c", "Data\Test4\BigGridTest.sumocfg", "--no-warnings"])
        # traci.start([sumoBinary, "-c", "Data\Test5\Rymal-upperRedHill.sumocfg", "--no-warnings"])
        # traci.start([sumoBinary, "-c", "Data\Test6\StoneChurch.sumocfg", "--no-warnings"])
        # traci.start([sumoBinary, "-c", "Data\TestCARLA\RymalRoadSimplified.sumocfg", "--no-warnings"])
        # traci.start([sumoBinary, "-c", "Data\TestCARLA\TwoIntersectionTest.sumocfg", "--no-warnings"])
        traci.start([sumoBinary, "-c", "Data\ThesisMapFull\ThesisMapFull_Draft2.sumocfg", "--no-warnings"])

        # INITIALIZE VARIABLES AND LISTS
        step = 0
        wait_total = []
        count_total = []
        total_loss = []
        prev_state = dict()
        light_times = dict()
        current_duration = dict()
        current_phase = dict()
        initial_yellow_phase = dict()
        prev_action = dict()
        target_edges = dict()
        surrounding_edges = dict()
        actions = dict()
        eastWest = dict()
        northSouth = dict()
        flow_rate_EW = {light: [] for light in lights}
        flow_rate_NS = {light: [] for light in lights}
        counted_vehicles_EW = set()
        counted_vehicles_NS = set()

        # specific_lights = {'26', '85', '142', '68', '42'}
        for light_id, light in enumerate(specific_lights):
            print(f'Getting info for light {light}')
            # GET INITIAL PHASE DURATION; SET STATES TO 0 BEFORE SIMULATION
            light_times[light] = traci.trafficlight.getPhaseDuration(light)
            # light_times[light] = traci.trafficlight.getPhaseDuration(light) / 0.25
            prev_state[light_id] = 0
            prev_action[light_id] = 0

            # SET INITIAL ACTION
            init_action[light] = 1

            # GET CURRENT PHASE DURATION AND STATE
            current_duration[light] = traci.trafficlight.getPhaseDuration(light)
            # current_duration[light] = traci.trafficlight.getPhaseDuration(light) / 0.25
            current_phase[light] = traci.trafficlight.getRedYellowGreenState(light)

            # DEFINE INITIAL YELLOW PHASE 
            initial_yellow_phase[light] = current_phase[light].replace('G', 'y').replace('g', 'y')

            # GET MAIN SIGNALIZED EDGES
            target_edges[light] = incoming_cont_edges(light)

            # GET SURROUNDING SIGNALIZED EDGES
            surrounding_edges[light] = surrounding_cont_edges(light, lights)

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

        while step <= end_time:
            
            # SIMULATION STEP
            traci.simulationStep()
            step += 1
            # print(f"Step: {step}")
            

            for light_id, light in enumerate(specific_lights):
                # TARGET EDGE QUEUE INFORMATION
                vehicles_per_edge_NS, vehicles_per_edge_EW, max_wait_time_NS, max_wait_time_EW = queue_info(target_edges[light])

                # GET TOTAL VEHICLES AND MAX WAIT TIME
                vehicle_total = vehicles_per_edge_NS + vehicles_per_edge_EW
                count_total.append(vehicle_total)

                max_wait = max_wait_time_NS + max_wait_time_EW
                wait_total.append(max_wait)

                # # SURROUNDING EDGE QUEUE INFORMATION
                S_vehicles_per_edge_NS, S_vehicles_per_edge_EW, S_max_wait_time_NS, S_max_wait_time_EW = queue_info(surrounding_edges[light])

                light_times[light] -= 1

                # CHECK IF ANY VEHICLES ARE PRESENT IN OPPOSITE DIRECTION; IF NO ADD MORE TIME TO CURRENT DIRECTION; IF YES CONTINUE
                if current_phase[light][0] == 'G' and light_times[light] == 0:
                    if max_wait_time_NS <= 5:
                        light_times[light] += 3
                if current_phase[light][0] == 'r' and current_phase[light][5] == 'G' and light_times[light] == 0:
                    if max_wait_time_EW <= 5:
                        light_times[light] += 3

                # IF LIGHT IS YELLOW AND TIME IS UP, SELECT ACTION
                if light_times[light] == 0 and 'y' in current_phase[light]:
                    # GET STATE VALUES IN FORM [Edge1_value, Edge2_value, ...]
                    # Look into normalizing the state values [0, 1]
                    # Look into getting traffic flow for state value - possible average of the flow in both directions
                    state_ = [vehicles_per_edge_NS, max_wait_time_NS, vehicles_per_edge_EW,  max_wait_time_EW, \
                              S_vehicles_per_edge_NS, S_max_wait_time_NS, S_vehicles_per_edge_EW, S_max_wait_time_EW]
                    state_ += [0] * (max_state_size - len(state_))
                    state = prev_state[light_id]
                    prev_state[light_id] = state_

                    # REWARD FUNCTION WITH VARYING WEIGHTS ON EACH VALUE
                    reward = -1 * (1*vehicles_per_edge_NS + 0.6*max_wait_time_NS + 1*vehicles_per_edge_EW + 0.6*max_wait_time_EW \
                                + 0.05*S_vehicles_per_edge_NS + 0.05*S_max_wait_time_NS + 0.05*S_vehicles_per_edge_EW + 0.05*S_max_wait_time_EW)

                    # Look into normalizing the reward [-1, 0]
                    # Look into getting traffic flow for reward value - possible average of the flow in both directions
                    # penalty if signals change too often: average taffic flow - gamma*low_time_penalty(time)
                    # print(f"Reward: {reward}")

                    # STORE TRANSITION
                    agent.store_transition(state, prev_action[light_id], reward, state_, (step==end_time), light_id)

                    # CHOOSE ACTION
                    action = agent.choose_action(state_)
                    # print(f"Action: {action}")
                    prev_action[light_id] = action

                    # ADJUST TRAFFIC LIGHTS
                    if current_phase[light][0] == 'y':
                        actions[light] = northSouth[light]
                    else:
                        actions[light] = eastWest[light]
                    adjust_traffic_light(light, actions[light][action][0], actions[light][action][1])
                    current_duration[light] = actions[light][action][0]
                    current_phase[light] = actions[light][action][1]
                    light_times[light] = (current_duration[light]) - 1
                    # light_times[light] = (current_duration[light] / 0.25) - 1

                    # LEARN
                    if train:
                        agent.learn(light_id)
                        loss = agent.learn(light_id)
                        if loss == None:
                            loss = 0.0
                        # print(f"Loss: {loss}")
                        total_loss.append(loss)
                        continue

                # IF LIGHT IS GREEN AND TIME IS UP AND NOT IN INITIAL STATE, SELECT CORRESPONDING YELLOW LIGHT ACTION   
                elif 'G' in current_phase[light] and light_times[light] == 0 and init_action[light] != 1:

                    adjust_traffic_light(light, actions[light][action][2], actions[light][action][3])  
                    current_phase[light] = actions[light][action][3]
                    light_times[light] = (actions[light][action][2]) - 1
                    # light_times[light] = (actions[light][action][2] / 0.25) - 1
                    continue
                
                # IF LIGHT IS GREEN AND TIME IS UP AND IN INITIAL STATE, SELECT INITIAL YELLOW LIGHT ACTION
                elif ('G' in current_phase[light] and light_times[light] == 0 and init_action[light] == 1):
                    # print("we got here")
                    adjust_traffic_light(light, 5, initial_yellow_phase[light])
                    # print("we got past here")
                    current_phase[light] = initial_yellow_phase[light]
                    light_times[light] = (5) - 1
                    # light_times[light] = (5 / 0.25) - 1
                    init_action[light] = 0
                    continue

        
        if epoch > 0 and epoch % 10 == 0:
            agent.save_checkpoint()

        # GET AVERAGES PERFORMANCE EVALUATION
        if train:
            if wait_total is None or len(wait_total) == 0:
                average_max_wait = 0
                average_count = 0
                average_loss = 0
            else:
                average_max_wait = mean(wait_total)
                average_count = mean(count_total)
            # average_loss = mean(total_loss)
                average_loss = mean([loss.item() if torch.is_tensor(loss) else loss for loss in total_loss])

        # SAVING BEST MODEL
        if train:
            if average_max_wait < best_time:
                best_time = average_max_wait
                agent.save_model(model_name)
                print("Model file overwritted with new best model.")

        # CLOSE SUMO FOR NEXT EPOCH
        traci.close()
        if not train:
            break



        # APPEND AVERAGES FOR PLOT
        if train:
            waiting_time.append(average_max_wait)
            waiting_ammt.append(average_count)
            loss_total.append(average_loss)
            print(f"Average waiting time: {round(average_max_wait,2)} | Average vehicle count: {round(average_count)}")
            print(f"Average loss: {round(average_loss, 2)}")
            print("\n")

    # PLOT RESULTS
    if train:
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
        plt.show()

        # PLOTTING LOSS OVER EPOCHS
        plt.figure()
        plt.plot(range(1, len(loss_total)+1), loss_total)
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss")
        plt.title(f"Loss Over {epochs} Epochs")
        plt.show()

        # PLOTTING TRAFFIC FLOW RATES FOR EACH LIGHT
        # for light in lights:
        #     plt.figure(figsize=(10, 5))
        #     # Convert the flow rates to pandas Series
        #     flow_rate_EW_series = pd.Series(flow_rate_EW[light])
        #     flow_rate_NS_series = pd.Series(flow_rate_NS[light])
            
        #     # Calculate the rolling averages
        #     rolling_avg_EW = flow_rate_EW_series.rolling(window=38).mean()
        #     rolling_avg_NS = flow_rate_NS_series.rolling(window=37).mean()
            
        #     plt.stackplot(range(1, step+1), rolling_avg_EW, rolling_avg_NS, labels=['East-West', 'North-South'])
        #     plt.title(f'Rolling average traffic flow for light {light}')
        #     plt.xlabel('Time step')
        #     plt.ylabel('Flow rate')
        #     plt.legend(loc='upper left')

        # plt.show() 

# OPTIONS (CUS WHY NOT)
def get_options():
    optParser = optparse.OptionParser() 
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="model",
        help="name of model",
    )
    optParser.add_option(
        "--train",
        action = 'store_true',
        default=False,
        help="training or testing",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=50,
        help="Number of epochs",
    )
    options, args = optParser.parse_args()
    return options

# RUN MAIN FUNCTION
if __name__ == "__main__":
    options = get_options()
    model_name = options.model_name
    train = options.train
    epochs = options.epochs
    main(train=train, model_name=model_name, epochs=epochs)