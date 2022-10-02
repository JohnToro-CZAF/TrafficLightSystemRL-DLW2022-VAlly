from __future__ import absolute_import
from __future__ import print_function

#from joblib import Parallel, delayed
import subprocess
#import pd
import os
import sys
import time
import math
import optparse
from collections import defaultdict
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

def countCarState(fn, max_frame, state_dict):
	state = pd.read_csv(fn, header = None, sep = ' ')
	dict_input = {}
	for row in range(state.shape[0]):
		dict_input[state.iloc[row, 1]] = [state.iloc[row, 2] - 1, state.iloc[row, 3]]
	lst = [0] * 12
	state_dict.append(lst)
	for it in range(max_frame):
		row = dict_input.get(it, [0, 0]) 
		element = [i for i in lst]
		if row != [0, 0]:
			element[row[0]] += row[1]
		lst = [i for i in element]
		state_dict.append(element)

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
	ser = serial.Serial(port='/dev/tty.usbserial-110', baudrate=9600, timeout=.1)

	brain.Q_eval.load_state_dict(torch.load(args.model_path, map_location=brain.Q_eval.device))

	step = 1
	min_duration = 3
	cur_direction = 0
	traffic_lights_time = dict()
	state_dict = list()
	countCarState(args.state_path, args.n_steps, state_dict)

	for junction in all_junctions:
		traffic_lights_time[junction] = 0
	lastStep = 0
	while step <= min(args.n_steps, len(state_dict)):
		# print(state_dict[step])
		for junction in all_junctions:
			# lane ID
			controlled_lanes = [0, 1, 2, 3]
			if step > 150:
				if abs(traffic_lights_time[junction]) < 0.0001:
					# Get the state at current frame
					vehicles_per_lane = [a_i-b_i for a_i, b_i in zip(state_dict[step], state_dict[lastStep])]
					state = vehicles_per_lane
					phase_time = brain.choose_action(state)
					traffic_lights_time[junction] = min_duration + phase_time
					ph = str('%d %d\n' % (cur_direction, traffic_lights_time[junction]))
					ser.write(bytes(ph, 'utf-8'))
					cur_direction = flip_bit(cur_direction)
					print(cur_direction, " ", traffic_lights_time[junction])
					lastStep = step
				else:
					traffic_lights_time[junction] -= 1/15
		step += 1
		time.sleep(1/15)


def get_options():
	optParser = optparse.OptionParser()
	optParser.add_option(
		'--model_path',
		type='string',
		default='./controlling/models/1st_test.bin',
		help='load model path'
	)
	optParser.add_option(
		'--n_steps',
		type='int',
		default=300,
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