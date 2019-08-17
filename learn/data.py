import os
import pickle
import random
import pdb
import multiprocessing
import time

import numpy as np

from problem_environments.mover_env import Mover
from mover_library.utils import get_body_xytheta

from trajectory_representation.predicates.is_reachable import IsReachable
from trajectory_representation.predicates.in_way import InWay
from trajectory_representation.predicates.in_region import InRegion
from trajectory_representation.predicates.blocks_key_configs import BlocksKeyConfigs


def node_features(entity, mover, goal, is_reachable, blocks_key_configs):
	isobj = entity not in mover.regions

	obj = mover.env.GetKinBody(entity) if isobj else None

	pose = get_body_xytheta(obj)[0] if isobj else None

	return [
		0, # l
		0, # w
		0, # h
		pose[0] if isobj else 0, # x
		pose[1] if isobj else 0, # y
		pose[2] if isobj else 0, # theta
		entity not in mover.regions,  # IsObj
		entity in mover.regions,      # IsRoom
		entity in goal, 			  # IsGoal
		is_reachable(entity, goal),
		blocks_key_configs(entity, goal),
	]


def edge_features(a, b, goal, in_way, in_region):
	return [
		in_way(a, b, goal),
		in_region(a, b, goal),
	]


def parse_example(example, mover=None):
	state, goal, action, cost = example

	mover_was_none = False
	if state is not None:
		seed, robot_transform, object_transforms = state

		if mover is None:
			mover_was_none = True

			np.random.seed(seed)
			random.seed(seed)

			mover = Mover()

		mover.robot.SetTransform(robot_transform)

		for object_name, object_transform in object_transforms.items():
			mover.env.GetKinBody(object_name).SetTransform(object_transform)

	entity_names = [obj.GetName() for obj in mover.objects] + list(mover.regions)
	entity_idx = {name: idx for idx, name in enumerate(entity_names)}

	#is_reachable = IsReachable(mover)
	is_reachable = lambda *x: True
	#blocks_key_configs = BlocksKeyConfigs(mover)
	blocks_key_configs = lambda *x: False
	#in_way = InWay(mover)
	#in_way = InWay(mover, ['entire_region'])
	in_way = lambda *x: False
	in_region = InRegion(mover)

	nodes = np.array([
		node_features(entity, mover, goal, is_reachable, blocks_key_configs)
		for entity in entity_names
	])

	edges = np.array([[
			edge_features(a, b, goal, in_way, in_region)
			for b in entity_names
		] for a in entity_names
	])

	action_params = [
		entity_idx[param]
		for param in action
	]
	# This is rather strange?

	if mover_was_none:
		mover.problem_config['env'].Destroy()

	return nodes, edges, action_params, cost

def try_parse_example(example, output):
	try:
		print('a')
		out = parse_example(example)
		print('b')
		output.send(out)
		print('c')
	except:
		output.send(None)
		pass
	output.close()
	#print('d')
	#done.put_nowait(True)
	#print('e')
	#output.close()
	#done.close()
	#print('f')

def load_data(filename):
	cachefile = filename + '.cache.pkl'

	if os.path.isfile(cachefile):
		return pickle.load(open(cachefile, 'rb'))

	examples = pickle.load(open(filename, 'rb'))[:]
	print(len(examples))
	#output = multiprocessing.Queue(len(examples))
	#done = multiprocessing.Queue(len(examples))

	output_list = []

	for i,example in enumerate(examples):
		print("processing example {}".format(i))
		c1, c2 = multiprocessing.Pipe(False)
		process = multiprocessing.Process(target=try_parse_example, args=(example, c2))
		print('aa')
		process.start()
		print('bb')
		#process.join(60)
		result = c1.recv()
		print('cc')
		if result is not None:
			print('dd')
			output_list.append(result)
		print('ee')
		#if process.is_alive():
		#	process.terminate()
		#	print("ran out of time for example {}".format(i))
		#print('ff')
		#try:
		#	done.get(True, 10)
		#except:
		#	print(done.empty())
		#	print("ran out of time for example {}".format(i))
		#print('cc')
		#if process.is_alive():
		#	print('dd')
		#	process.terminate()
		#	#print("ran out of time for example {}".format(i))
		#print('ee')

	#while not output.empty():
	#	output_list.append(output.get())

	data = [np.stack(component) for component in zip(*output_list)]

	pickle.dump(data, open(cachefile, 'wb'))

	return data

if __name__ == '__main__':
	data = load_data('./training_data_541.pkl')
	pdb.set_trace()

