import numpy as np
from planners.flat_mcts.mcts_tree_node import TreeNode
import openravepy
from manipulation.bodies.bodies import set_color
from gtamp_utils.utils import visualize_path


class DiscreteTreeNode(TreeNode):
    def __init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node, actions):
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

    def perform_ucb_over_actions(self, learned_q_functions=None):
        never_executed_actions_exist = len(self.Q) != len(self.A)

        if never_executed_actions_exist and learned_q_functions is None:
            best_action = self.get_never_evaluated_action()
        else:
            # why is it never coming down here? Because there are actions that have not been tried.
            if self.is_operator_skeleton_node:
                feasible_actions = self.A
            else:
                feasible_actions = [a for a in self.A if a.continuous_parameters['is_feasible']]

            feasible_q_values = [self.Q[a] for a in feasible_actions]

            if not self.is_operator_skeleton_node:
                assert (len(feasible_actions) > 1)

            for a, q in zip(feasible_actions, feasible_q_values):
                print a.discrete_parameters, q
            try:
                best_action = self.get_action_with_highest_ucb_value(feasible_actions, feasible_q_values)
            except:
                import pdb;pdb.set_trace()

        return best_action

