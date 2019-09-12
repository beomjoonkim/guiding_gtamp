import numpy as np
from planners.flat_mcts.mcts_tree_node import TreeNode
import openravepy
from manipulation.bodies.bodies import set_color
from gtamp_utils.utils import visualize_path
from planners.heuristics import get_objects_to_move


class DiscreteTreeNode(TreeNode):
    def __init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node, actions, learned_q):
        self.learned_q = learned_q
        TreeNode.__init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node)
        self.add_actions(actions)

    def add_actions(self, actions):
        if self.is_operator_skeleton_node:
            for action in actions:
                self.A.append(action)
                self.N[action] = 0

    def update_node_statistics(self, action, sum_rewards, reward):
        self.Nvisited += 1

        is_action_never_tried = self.N[action] == 0
        if is_action_never_tried:
            self.reward_history[action] = [reward]
            self.Q[action] = sum_rewards
            self.N[action] += 1
        else:
            # todo don't use averaging
            self.reward_history[action].append(reward)
            self.N[action] += 1
            if sum_rewards > self.Q[action]:
                self.Q[action] = sum_rewards

    def perform_ucb_over_actions(self):
        assert self.is_operator_skeleton_node
        actions = self.A
        q_values = [self.Q[a] for a in self.A]
        best_action = self.get_action_with_highest_ucb_value(actions, q_values)
        return best_action



