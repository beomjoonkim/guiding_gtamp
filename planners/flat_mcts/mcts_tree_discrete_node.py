import numpy as np
from planners.flat_mcts.mcts_tree_node import TreeNode
import openravepy
from manipulation.bodies.bodies import set_color
from gtamp_utils.utils import visualize_path
from planners.heuristics import get_objects_to_move


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

        if never_executed_actions_exist:
            best_action = self.get_never_evaluated_action()
            print "Executing the never executed action"
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
                obj_name = a.discrete_parameters['object']
                region_name = a.discrete_parameters['region']
                obj_a_reachable = self.state.is_entity_reachable(obj_name)
                a_r_manip_free = self.state.binary_edges[(obj_name, region_name)][-1]
                psa = obj_a_reachable and a_r_manip_free
                print "%30s %30s Reachable? %d  ManipFree? %d IsGoal? %d Q? %.5f Q+UCB? %.5f" \
                      % (obj_name, region_name, self.state.is_entity_reachable(obj_name),
                         self.state.binary_edges[(obj_name, region_name)][-1],
                         obj_name in self.state.goal_entities, self.Q[a], self.compute_ucb_value(self.Q[a], a))

                #print a.discrete_parameters['region'], a.discrete_parameters['object'], q,  self.compute_ucb_value(q, a)
            best_action = self.get_action_with_highest_ucb_value(feasible_actions, feasible_q_values)

        return best_action


class DiscreteTreeNodeWithPsa(DiscreteTreeNode):
    def __init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node, actions):
        DiscreteTreeNode.__init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node, actions)
        for a in self.A:
            self.Q[a] = 0

    def perform_ucb_over_actions(self, learned_q_functions=None):
        # why is it never coming down here? Because there are actions that have not been tried.
        assert self.is_operator_skeleton_node
        feasible_actions = self.A
        feasible_q_values = [self.Q[a] for a in feasible_actions]

        if not self.is_operator_skeleton_node:
            assert (len(feasible_actions) > 1)

        for a, q in zip(feasible_actions, feasible_q_values):
            obj_name = a.discrete_parameters['object']
            region_name = a.discrete_parameters['region']
            o_reachable = self.state.is_entity_reachable(obj_name)
            o_r_manip_free = self.state.binary_edges[(obj_name, region_name)][-1]
            o_needs_to_be_moved = obj_name in get_objects_to_move(self.state, self.state.problem_env)
            # and it is in M
            psa = o_reachable and o_r_manip_free and o_needs_to_be_moved
            print "%30s %30s Reachable? %d  ManipFree? %d IsGoal? %d Q? %.5f Q+UCB? %.5f" \
                  % (obj_name, region_name, self.state.is_entity_reachable(obj_name),
                     self.state.binary_edges[(obj_name, region_name)][-1],
                     obj_name in self.state.goal_entities, self.Q[a], self.compute_ucb_value(self.Q[a], a, psa))

        best_action = self.get_action_with_highest_ucb_value(feasible_actions, feasible_q_values)
        return best_action

    def get_action_with_highest_ucb_value(self, feasible_actions, feasible_q_values):
        best_value = -np.inf
        best_action = feasible_actions[0]
        ucb_values = {}
        for a, value in zip(feasible_actions, feasible_q_values):
            obj_name = a.discrete_parameters['object']
            region_name = a.discrete_parameters['region']
            o_reachable = self.state.is_entity_reachable(obj_name)
            o_r_manip_free = self.state.binary_edges[(obj_name, region_name)][-1]
            o_needs_to_be_moved = obj_name in get_objects_to_move(self.state, self.state.problem_env)
            # and it is in M
            psa = o_reachable and o_r_manip_free and o_needs_to_be_moved

            ucb_value = self.compute_ucb_value(value, a, psa)

            if ucb_value > best_value:
                best_value = ucb_value

            ucb_values[a] = ucb_value
        best_actions = []
        for action, value in zip(feasible_actions, feasible_q_values):
            ucb_value = ucb_values[action]
            if ucb_value == best_value:
                best_actions.append(action)
        return best_actions[np.random.randint(len(best_actions))]


