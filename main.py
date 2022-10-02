from __future__ import absolute_import
from __future__ import print_function

from joblib import Parallel, delayed
import subprocess
import pd
import os
import sys
import time
import math
import optparse
import torch
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
# import arduino
import serial.tools.list_ports as ports
import serial

sys.path.insert(1, "./controlling")
sys.path.insert(2, "./detection")

from controlling.train import Agent

def write_read(arduino, x):
	arduino.write(bytes(x, 'utf-8'))
	time.sleep(0.05)
	data = arduino.readline()
	return data

def flip_bit(a):
	if a==0:
		return 1
	else:
		return 0

def countCarState(fn, max_frame):
	state = pd.read_csv(fn, header = None, sep = ' ')
	dict_input = {}
	for row in range(state.shape[0]):
		dict_input[state.iloc[row, 1]] = [state.iloc[row, 2] - 1, state.iloc[row, 3]]

	dict_state = {}
	dict_state[0] = [0] * 12
	for it in range(1, max_frame):
		row = dict_input.get(it, [0, 0])
		dict_state[it] = dict_state[it-1] 
		if row != [0, 0]:
			dict_state[it][row[0]] += row[1]
	return dict_state

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

def run_process(args):
	# We only have on junction
	all_junctions = [0]
	brain = Agent(
    gamma=0.99,
    epsilon=0.0,
		lr=0.1,
		input_dims = 12,
		fc1_dims=256,
		fc2_dims=256,
		batch_size=1024,
		n_actions=10,
		junctions=all_junctions,
		Q_eval = Model(0.1, 12, 256, 256, 10)
  )
	if args.ard:
		com_ports = list(ports.comports())
		if len(com_ports) != 0:
			print(com_ports[0])
			arduino = serial.Serial(port=com_ports[0], baudrate=9600, timeout=.1)
		else:
			print("There is no port openning")

	brain.Q_eval.load_state_dict(torch.load(args.model_path, map_location=brain.Q_eval.device))

	step = 1
	min_duration = 5
	cur_direction = 0
	traffic_lights_time = dict()
	state_dict = countCarState(args.state_path, args.n_steps)
	
	for junction in all_junctions:
		traffic_lights_time[junction] = 0

	while step <= min(args.n_steps, len(state_dict)):
		for junction in all_junctions:
			# lane ID
			controlled_lanes = [0, 1, 2, 3]
			if abs(traffic_lights_time[junction]) < 0.0001:
				# Get the state at current frame
				vehicles_per_lane = state_dict[step] 
				state = vehicles_per_lane
				phase_time = brain.choose_action(state)
				traffic_lights_time[junction] = min_duration + phase_time
				if args.ard:
					ph = str('%d%d' % (cur_direction, traffic_lights_time[junction]))
					value = write_read(arduino, ph)
				cur_direction = flip_bit(cur_direction)
				print(cur_direction, " ", traffic_lights_time[junction])
			else:
				traffic_lights_time[junction] -= 1/15
		step += 1
		time.sleep(1/15)
	# print(traffic_lights_time[0])


def get_options():
	optParser = optparse.OptionParser()
	optParser.add_option(
		'--model_path',
		type='string',
		default='/Users/johntoro/Documents/Projects/TrafficLight/TrafficLightSystemRL-DLW2022-VAlly/controlling/models/1st_test.bin',
		help='load model path'
	)
	optParser.add_option(
		'--n_steps',
		type='int',
		default=500,
		help='time for rendering'
	)
	optParser.add_option(
		'--state_path',
		type='string',
		default='./detection/experiments/cam4/output.txt',
		help='time frame input state'
	)
	optParser.add_option(
		'--ard',
		action='store_true',
		default=False,
		help='if we are output to the arduino or not'
	)
	options, args = optParser.parse_args()
	return options

if __name__ == "__main__":
	options = get_options()
	run_process(options)
	# tasks = {"sim": args1, "vis": args2}
	# Parallel(n_jobs=2)(delayed(run(task, args)) for task, args in tasks.items())