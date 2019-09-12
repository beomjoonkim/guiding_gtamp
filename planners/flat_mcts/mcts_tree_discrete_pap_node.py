from mcts_tree_discrete_node import DiscreteTreeNode
import numpy as np
import openravepy
from manipulation.bodies.bodies import set_color
from gtamp_utils.utils import visualize_path, set_color, viewer
from gtamp_utils import utils


class PaPDiscreteTreeNodeWithPriorQ(DiscreteTreeNode):
    def __init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node, prior_q,
                 actions, is_goal_reached=False):
        DiscreteTreeNode.__init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node,
                                  is_init_node, actions)
        self.prior_q = {}
        self.q_function = prior_q

        is_infeasible_state = state is None
        if not is_infeasible_state and not is_goal_reached:
            self.initialize_mixed_q_values()
        self.mix_weight = 0.99
        self.max_sum_rewards = {}

    def visualize_values_in_two_arm_domains(self, entity_values, entity_names):
        is_place_node = entity_names[0].find('region') != -1
        if is_place_node:
            return

        max_val = np.max(entity_values)
        min_val = np.min(entity_values)
        for entity_name, entity_value in zip(entity_names, entity_values):
            entity = openravepy.RaveGetEnvironments()[0].GetKinBody(entity_name)
            set_color(entity, [0, (entity_value - min_val) / (max_val - min_val), 0])

    def initialize_mixed_q_values(self):
        for a in self.A:
            self.Q[a] = self.q_function(self.state, a)
            self.prior_q[a] = self.q_function(self.state, a)

    def update_node_statistics(self, action, sum_rewards, reward):
        is_action_never_tried = self.N[action] == 0
        if is_action_never_tried:
            self.max_sum_rewards[action] = sum_rewards
            self.reward_history[action] = [reward]
        else:
            if sum_rewards > self.max_sum_rewards[action]:
                self.max_sum_rewards[action] = sum_rewards
            self.reward_history[action].append(reward)

        self.Nvisited += 1
        self.N[action] += 1
        temperature_on_action = np.power(self.mix_weight, self.N[action])
        self.Q[action] = temperature_on_action*self.Q[action] + (1 - temperature_on_action)*self.max_sum_rewards[action]

    def perform_ucb_over_actions(self, prior_q_functions=None):
        # why does this get called before initializing mixed q values
        # todo why does it ever have key error here?
        # it performs ucb_over_actions in an infeasible state?
        qucb_vals = []
        for a in self.A:
            qucb_vals.append(self.compute_ucb_value(self.Q[a], a))
            obj_name = a.discrete_parameters['object']
            region_name = a.discrete_parameters['region']
            print "%30s %30s Reachable? %d  ManipFree? %d IsGoal? %d Q? %.5f Q+UCB? %.5f" \
                  % (obj_name, region_name, self.state.is_entity_reachable(obj_name),
                     self.state.binary_edges[(obj_name, region_name)][-1],

                     obj_name in self.state.goal_entities, self.Q[a], self.compute_ucb_value(self.Q[a], a))

        best_action = self.get_action_with_highest_ucb_value(self.A, qucb_vals)  # but your Nsa are all zero?
        idx = -2
        """
        while self.is_action_redundant(best_action):
            ucb_values = self.compute_ucb_values(self.A, qucb_vals)
            ucb_actions = ucb_values.keys()
            ucb_values = ucb_values.values()
            best_action = np.array(ucb_actions)[np.argsort(ucb_values)][idx]
            idx -= -1
            print 'Redundant action detected'
        """

        print "Chosen action", best_action.discrete_parameters['object'], best_action.discrete_parameters['region']

        print self.state.get_entities_in_pick_way('square_packing_box1')
        print self.state.get_entities_in_place_way('square_packing_box1', 'home_region')
        #print self.state.get_entities_in_pick_way('rectangular_packing_box1')
        #print self.state.get_entities_in_place_way('rectangular_packing_box1', 'home_region')
        return best_action

    def is_obj_currently_in_goal_region(self, obj):
        curr_obj_region = self.state.problem_env.get_region_containing(obj)
        return curr_obj_region.name in self.state.goal_entities

    def not_in_way_of_anything(self, obj_name):
        return len(self.state.get_entities_occluded_by(obj_name)) == 0

    def is_in_goal_region(self, obj_name):
        goal_region = [o for o in self.state.goal_entities if o.find('region') != -1][0]
        curr_in = self.state.problem_env.get_region_containing(obj_name)
        return curr_in.name == goal_region

    def is_action_redundant(self, a):
        obj_name = a.discrete_parameters['object']
        region_name = a.discrete_parameters['region']

        goal_achieved_and_not_in_way = self.state.is_goal_entity(obj_name) and self.is_in_goal_region(obj_name) \
                                       and self.not_in_way_of_anything(obj_name)
        non_goal_not_in_way = not self.state.is_goal_entity(obj_name) and self.not_in_way_of_anything(obj_name)

        goal_region = [o for o in self.state.goal_entities if o.find('region') != -1][0]
        not_goal_entity_to_goal_region = not self.state.is_goal_entity(obj_name) and region_name == goal_region

        return goal_achieved_and_not_in_way or non_goal_not_in_way or not_goal_entity_to_goal_region




