from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import random
import serial
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# we need to import python modules from the $SUMO_HOME/tools directory
# if "SUMO_HOME" in os.environ:
#     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

# from sumolib import checkBinary  # noqa
# import traci  # noqa

def flip_bit(a):
    if a == 0:
        return 1
    else:
        return 0

def right_or_left(a, b):
    """
    Check if vector b is on the right of the vector a or not
    1: on the right
    -1: on the left
    0: keep direction as a
    """
    dot = a[0]*-b[1] + a[1]*b[0]
    if(dot > 0):
        return 1
    elif (dot < 0):
        return 2
    else:
        return 0
def op_tuple(a, b):
    return (a[0] - b[0], a[1] - b[1])

def get_vehicle_numbers(lanes, olds):
    vehicle_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = [0, 0, 0]
        for k in traci.lane.getLastStepVehicleIDs(l):
            old_pos = olds.get(k, traci.lane.getShape(l)[0])
            next_pos = traci.vehicle.getPosition(k)
            vec_pos_vh = op_tuple(next_pos, old_pos)
            shape_lane = traci.lane.getShape(l)
            pos_lane_dir = [op_tuple(shape_lane[0], shape_lane[1]), op_tuple(shape_lane[1], shape_lane[0])]
            lane_dir = None
            for i in range(2):
                # print(pos_lane_dir[i], vec_pos_vh)
                if pos_lane_dir[i][0] * np.array(vec_pos_vh)[0] + pos_lane_dir[i][1] * np.array(vec_pos_vh)[1] >= 0:
                    lane_dir = pos_lane_dir[i]
                    break
            if traci.vehicle.getLanePosition(k) > 10:
                flow = right_or_left(lane_dir, vec_pos_vh)
                vehicle_per_lane[l][flow] += 1
            olds[k] = next_pos
    return vehicle_per_lane


def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time


def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)

class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        batch_size,
        n_actions,
        junctions,
        Q_eval = 0,
        max_memory_size=100000,
        epsilon_dec=5e-4,
        epsilon_end=0.05,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100
        if Q_eval != 0:
            self.Q_eval = Q_eval
        else:
            self.Q_eval = Model(
                self.lr, self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions
        )
        self.memory = dict()
        for junction in junctions:
            self.memory[junction] = {
                "state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "new_state_memory": np.zeros(
                    (self.max_mem, self.input_dims), dtype=np.float32
                ),
                "reward_memory":np.zeros(self.max_mem, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem, dtype=np.bool),
                "mem_cntr": 0,
                "iter_cntr": 0,
            }


    def store_transition(self, state, state_, action,reward, done,junction):
        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = state_
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def reset(self,junction_numbers):
        for junction_number in junction_numbers:
            self.memory[junction_number]['mem_cntr'] = 0

    def save(self,model_name):
        torch.save(self.Q_eval.state_dict(), f'models/{model_name}.bin')

    def learn(self, junction):
        self.Q_eval.optimizer.zero_grad()

        batch = np.arange(self.memory[junction]['mem_cntr'], dtype=np.int32)

        state_batch = torch.tensor(self.memory[junction]["state_memory"][batch]).to(
            self.Q_eval.device
        )
        new_state_batch = torch.tensor(
            self.memory[junction]["new_state_memory"][batch]
        ).to(self.Q_eval.device)
        reward_batch = torch.tensor(
            self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device)
        action_batch = self.memory[junction]["action_memory"][batch]

        q_eval = self.Q_eval.forward(state_batch)[batch, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = (
            self.epsilon - self.epsilon_dec
            if self.epsilon > self.epsilon_end
            else self.epsilon_end
        )


def run(train=True,model_name="model",epochs=50,steps=500,ard=False):
    if ard:
        arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
        def write_read(x):
            arduino.write(bytes(x, 'utf-8'))
            time.sleep(0.05)
            data = arduino.readline()
            return data
    """execute the TraCI control loop"""
    epochs = epochs
    steps = steps
    best_time = np.inf
    total_time_list = list()
    traci.start(
        [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"]
    )
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))

    #  n_actions is the chosed time 
    brain = Agent(
        gamma=0.99,
        epsilon=0.0,
        lr=0.1,
        input_dims = 12,
        # input_dims = len(all_junctions) * 12,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=1024,
        n_actions=10,
        junctions=junction_numbers,
    )

    if not train:
        brain.Q_eval.load_state_dict(torch.load(f'models/{model_name}.bin',map_location=brain.Q_eval.device))

    traci.close()
    olds = dict()
    for e in range(epochs):
        if train:
            traci.start(
            [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )
        else:
            traci.start(
            [checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )

        print(f"epoch: {e}")
        select_lane = [
            ["yyyrrryyyrrr", "GGGrrrGGGrrr"],
            ["rrryyyrrryyy", "rrrGGGrrrGGG"]
        ]

        step = 0
        total_time = 0
        min_duration = 5
        
        traffic_lights_time = dict()
        prev_wait_time = dict()
        prev_vehicles_per_lane = dict()
        prev_action = dict()
        all_lanes = list()
        olds = dict()

        cur_direction = 0
        # This only applied for one lane only
        # We have to set which lane is next to be choosed
        # Since this is the single junction with 4 lanes: 2 lanes are in the same axis and opposite direction and two other lanes are on the right and left
        for junction_number, junction in enumerate(all_junctions):
            prev_wait_time[junction] = 0
            prev_action[junction_number] = 0
            traffic_lights_time[junction] = 0
            prev_vehicles_per_lane[junction_number] = [0, 0, 0] * 4
            # prev_vehicles_per_lane[junction_number] = [0] * (len(all_junctions) * 4) 
            controled_lanes = traci.trafficlight.getControlledLanes(junction)
            all_lanes.extend(list(controled_lanes))

            for l in controled_lanes:
                for k in traci.lane.getLastStepVehicleIDs(l):
                    olds[k] = (0, 0)

        while step <= steps:
            traci.simulationStep()
            for junction_number, junction in enumerate(all_junctions):
                controled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(controled_lanes)

                total_time += waiting_time
                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(controled_lanes, olds)
                    # vehicles_per_lane is a dict with key is lane and value as a 3-sized list

                    #storing previous state and current state
                    reward = -1 *  waiting_time
                    state_ = [flow for l in vehicles_per_lane.values() for flow in l]
                    state = prev_vehicles_per_lane[junction_number]
                    prev_vehicles_per_lane[junction_number] = state_
                    brain.store_transition(state, state_, prev_action[junction_number],reward,(step==steps), junction_number)

                    #selecting new action based on current state
                    phase_time = brain.choose_action(state_)
                    prev_action[junction_number] = phase_time
                    phaseDuration(junction, 1, select_lane[cur_direction][0])
                    phaseDuration(junction, min_duration + phase_time, select_lane[cur_direction][1])

                    if ard:
                        ph = str(traci.trafficlight.getPhase("0"))
                        value = write_read(ph)

                    traffic_lights_time[junction] = min_duration + phase_time
                    if train:
                        brain.learn(junction_number)
                    cur_direction = flip_bit(cur_direction)
                else:
                    traffic_lights_time[junction] -= 1
            step += 1
        print("total_time",total_time)
        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                brain.save(model_name)

        traci.close()
        sys.stdout.flush()
        if not train:
            break
    if train:
        plt.plot(list(range(len(total_time_list))),total_time_list)
        plt.xlabel("epochs")
        plt.ylabel("total time")
        plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
        plt.show()

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
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=500,
        help="Number of steps",
    )
    optParser.add_option(
       "--ard",
        action='store_true',
        default=False,
        help="Connect Arduino", 
    )
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    model_name = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    ard = options.ard
    run(train=train,model_name=model_name,epochs=epochs,steps=steps,ard=ard)
