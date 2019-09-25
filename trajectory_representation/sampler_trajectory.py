from gtamp_problem_environments.mover_env import Mover
from gtamp_utils import utils
from trajectory_representation.concrete_node_state import ConcreteNodeState

import copy
import openravepy
import numpy as np
import random


class SamplerTrajectory:
    def __init__(self, problem_idx, key_configs):
        self.problem_idx = problem_idx
        self.paps_used = None
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_prime = []
        self.seed = None  # this defines the initial state
        self.key_configs = key_configs
        self.problem_env = None

    def compute_state(self, obj, region):
        # todo the state of a concrete node consists of the object, region, and the collision vector.
        if 'two_arm_mover' in self.problem_env.name:
            goal_entities = ['square_packing_box1', 'home_region']
        else:
            raise NotImplementedError
        return ConcreteNodeState(self.problem_env, obj, region, goal_entities, self.key_configs)

    def add_state_prime(self):
        self.state_prime = self.states[1:]

    def add_sar_tuples(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def create_environment(self):
        problem_env = Mover(self.problem_idx)
        openrave_env = problem_env.env
        return problem_env, openrave_env

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)

    def add_trajectory(self, plan):
        print "Problem idx", self.problem_idx
        self.set_seed(self.problem_idx)
        problem_env, openrave_env = self.create_environment()
        self.problem_env = problem_env

        state = None
        for action_idx, action in enumerate(plan):
            if 'pick' in action.type:
                associated_place = plan[action_idx+1]
                state = self.compute_state(action.discrete_parameters['object'],
                                           associated_place.discrete_parameters['region'])
                #incollision_configs = self.key_configs[state.state_vec[:-2,0]==1]
                action.execute()
                pick_rel_pose = utils.get_relative_base_pose_from_absolute_base_pose(
                    action.discrete_parameters['object'])
                #pick_base_pose = action.continuous_parameters['q_goal']
                # I need this to be relative. How do I do that?
            else:
                if action == plan[-1]:
                    reward = 0
                else:
                    reward = -1
                action.execute()
                place_base_pose = action.continuous_parameters['q_goal']
                cont_pap_params = np.hstack([pick_rel_pose, place_base_pose])
                self.add_sar_tuples(state, cont_pap_params, reward)

        self.add_state_prime()
        openrave_env.Destroy()
        openravepy.RaveDestroy()
