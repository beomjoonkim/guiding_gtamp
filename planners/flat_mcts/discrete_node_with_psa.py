from mcts_tree_discrete_node import DiscreteTreeNode
from planners.heuristics import get_objects_to_move

import numpy as np


def alpha_zero_ucb(n, n_sa, psa):
    return psa * np.sqrt(n + 1) / float(n_sa + 1)


class DiscreteTreeNodeWithPsa(DiscreteTreeNode):
    def __init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node, actions):
        # psa is based on the number of objs to move
        DiscreteTreeNode.__init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node,
                                  is_init_node, actions)
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
            objects_to_move = get_objects_to_move(self.state, self.state.problem_env)
            o_needs_to_be_moved = obj_name in objects_to_move
            # and it is in M
            psa = o_reachable and o_r_manip_free and o_needs_to_be_moved
            print "%30s %30s Reachable? %d  ManipFree? %d IsGoal? %d Q? %.5f Q+UCB? %.5f" \
                  % (obj_name, region_name, self.state.is_entity_reachable(obj_name),
                     self.state.binary_edges[(obj_name, region_name)][-1],
                     obj_name in self.state.goal_entities, self.Q[a],
                     self.compute_ucb_value(self.Q[a], a, objects_to_move))

        best_action = self.get_action_with_highest_ucb_value(feasible_actions, feasible_q_values)
        return best_action

    def compute_psa(self, action, objects_to_move):
        obj_name = action.discrete_parameters['object']
        region_name = action.discrete_parameters['region']
        o_reachable = self.state.is_entity_reachable(obj_name)
        o_r_manip_free = self.state.binary_edges[(obj_name, region_name)][-1]
        o_needs_to_be_moved = obj_name in objects_to_move
        if o_reachable and o_r_manip_free and o_needs_to_be_moved:
            psa = 1
        else:
            psa = 0.1
        return psa

    def compute_ucb_value(self, value, action, objects_to_move):
        psa = self.compute_psa(action, objects_to_move)
        return value + self.ucb_parameter * alpha_zero_ucb(self.Nvisited, self.N[action], psa)

    def get_action_with_highest_ucb_value(self, feasible_actions, feasible_q_values):
        best_value = -np.inf
        best_action = feasible_actions[0]
        ucb_values = {}
        objects_to_move = get_objects_to_move(self.state, self.state.problem_env)
        for a, value in zip(feasible_actions, feasible_q_values):
            ucb_value = self.compute_ucb_value(value, a, objects_to_move)

            if ucb_value > best_value:
                best_value = ucb_value

            ucb_values[a] = ucb_value

        best_actions = []
        for action, value in zip(feasible_actions, feasible_q_values):
            ucb_value = ucb_values[action]
            if ucb_value == best_value:
                best_actions.append(action)
        return best_actions[np.random.randint(len(best_actions))]
