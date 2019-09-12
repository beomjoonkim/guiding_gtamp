from mcts_tree_discrete_node import DiscreteTreeNode
from planners.heuristics import get_objects_to_move

import numpy as np


def alpha_zero_ucb(n, n_sa):
    return np.sqrt(n + 1) / float(n_sa + 1)


class DiscreteTreeNodeWithPriorQ(DiscreteTreeNode):
    def __init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node, actions,
                 learned_q):
        # psa is based on the number of objs to move
        DiscreteTreeNode.__init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node,
                                  is_init_node, actions, learned_q)
        is_infeasible_state = self.state is None
        if is_infeasible_state:
            for a in self.A:
                self.Q[a] = 0
        else:
            objs_to_move = get_objects_to_move(self.state, self.state.problem_env)
            for a in self.A:
                self.Q[a] = -len(objs_to_move)

    def perform_ucb_over_actions(self):
        # todo this function is to be deleted once everything has been implemented
        assert self.is_operator_skeleton_node
        actions = self.A
        q_values = [self.Q[a] for a in self.A]

        """
        for a, q in zip(actions, q_values):
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
        """
        best_action = self.get_action_with_highest_ucb_value(actions, q_values)
        return best_action

    def get_action_with_highest_ucb_value(self, actions, q_values):
        best_value = -np.inf

        if self.learned_q is not None:
            # todo make this more efficient by calling predict_with_raw_*
            init_q_values = [self.learned_q.predict(self.state, a) for a in actions]
            exp_sum = np.sum([np.exp(q) for q in init_q_values])
        else:
            objects_to_move = get_objects_to_move(self.state, self.state.problem_env)
            init_q_values = []
            for a in actions:
                obj_name = a.discrete_parameters['object']
                region_name = a.discrete_parameters['region']
                o_reachable = self.state.is_entity_reachable(obj_name)
                o_r_manip_free = self.state.binary_edges[(obj_name, region_name)][-1]
                o_needs_to_be_moved = obj_name in objects_to_move
                if o_reachable and o_r_manip_free and o_needs_to_be_moved:
                    val = 1
                else:
                    val = 0
                init_q_values.append(val)
            exp_sum = np.sum([np.exp(q) for q in init_q_values])

        action_ucb_values = []

        for action, value, learned_value in zip(actions, q_values, init_q_values):
            q_bonus = np.exp(learned_value) / float(exp_sum)
            ucb_value = value + q_bonus + self.compute_ucb_value(action)

            obj_name = action.discrete_parameters['object']
            region_name = action.discrete_parameters['region']
            print "%30s %30s Reachable? %d  ManipFree? %d IsGoal? %d Q? %.5f QBonus? %.5f Q+UCB? %.5f" \
                  % (obj_name, region_name, self.state.is_entity_reachable(obj_name),
                     self.state.binary_edges[(obj_name, region_name)][-1],
                     obj_name in self.state.goal_entities, self.Q[action], q_bonus,
                     ucb_value)

            action_ucb_values.append(ucb_value)
            if ucb_value > best_value:
                best_value = ucb_value

        boolean_idxs_with_highest_ucb = (np.max(action_ucb_values) == action_ucb_values).squeeze()
        best_action_idx = np.random.randint(np.sum(boolean_idxs_with_highest_ucb))
        best_action = np.array(actions)[boolean_idxs_with_highest_ucb][best_action_idx]
        return best_action

    def compute_ucb_value(self, action):
        return self.ucb_parameter * alpha_zero_ucb(self.Nvisited, self.N[action])
