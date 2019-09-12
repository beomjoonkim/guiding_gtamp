from mcts_tree_continuous_node import ContinuousTreeNode
from mcts_tree_discrete_node import DiscreteTreeNode
from mcts_tree_discrete_pap_node import PaPDiscreteTreeNodeWithPriorQ
from mcts import MCTS

from generators.uniform import UniformPaPGenerator
from generators.voo import PaPVOOGenerator

from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from trajectory_representation.state import StateWithoutCspacePredicates
from trajectory_representation.one_arm_pap_state import OneArmPaPState

## openrave helper libraries
from gtamp_utils import utils


import numpy as np
import sys
import socket
import pickle
import time
import os

sys.setrecursionlimit(15000)
DEBUG = False

hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560':
    from mcts_graphics import write_dot_file


# todo
#   create MCTS for each environment. Each one will have different compute_state functions
class MCTSWithLeafStrategy(MCTS):
    def __init__(self, parameters, problem_env, goal_entities, v_fcn, learned_q):
        MCTS.__init__(self, parameters, problem_env, goal_entities, v_fcn, learned_q)

    def simulate(self, curr_node, node_to_search_from, depth, new_traj):
        if self.problem_env.reward_function.is_goal_reached():
            if not curr_node.is_goal_and_already_visited:
                self.found_solution = True
                curr_node.is_goal_node = True
                print "Solution found, returning the goal reward", self.problem_env.reward_function.goal_reward
                self.update_goal_node_statistics(curr_node, self.problem_env.reward_function.goal_reward)
            return self.problem_env.reward_function.goal_reward

        if depth == self.planning_horizon:
            # would it ever get here? why does it not satisfy the goal?
            print "Depth limit reached"
            return 0

        if DEBUG:
            print "At depth ", depth
            print "Is it time to pick?", self.problem_env.is_pick_time()

        action = self.choose_action(curr_node)
        is_action_feasible = self.apply_action(curr_node, action)

        is_tree_action = curr_node.is_action_tried(action)
        if is_tree_action:
            next_node = curr_node.children[action]
            reward = next_node.parent_action_reward
        else:
            next_node = self.create_node(action, depth + 1, curr_node, not is_action_feasible) # expansion
            self.tree.add_node(next_node, action, curr_node)
            reward = self.problem_env.reward_function(curr_node.state, next_node.state, action, depth)
            next_node.parent_action_reward = reward
            next_node.sum_ancestor_action_rewards = next_node.parent.sum_ancestor_action_rewards + reward

        print "Reward", reward

        print "=============================================================================="
        if not is_action_feasible:
            # this (s,a) is a dead-end
            print "Infeasible action"
            sum_rewards = reward
        else:
            if is_tree_action or curr_node.is_operator_skeleton_node:
                sum_rewards = reward + self.discount_rate * self.simulate(next_node, node_to_search_from, depth + 1,
                                                                          new_traj)
            else:
                next_state_value = self.v_fcn(next_node.state)
                print "Next state value", next_state_value
                sum_rewards = reward + next_state_value

        curr_node.update_node_statistics(action, sum_rewards, reward)
        if curr_node == node_to_search_from and curr_node.parent is not None:
            self.update_ancestor_node_statistics(curr_node.parent, curr_node.parent_action, sum_rewards)

        # todo return a plan
        return sum_rewards
