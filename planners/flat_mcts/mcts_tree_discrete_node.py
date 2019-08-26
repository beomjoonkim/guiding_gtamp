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


class DiscreteTreeNodeWithLearnedQ(DiscreteTreeNode):
    def __init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node, learned_q,
                 actions):
        DiscreteTreeNode.__init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node,
                                  is_init_node, actions)
        self.learned_q = learned_q
        self.q_function = self.learned_q
        is_pick_node = self.A[0].type.find('pick') != -1
        if is_pick_node:
            self.q_function = learned_q['pick']
        else:
            self.q_function = learned_q['place']
        self.mixedQ = {}
        self.learnedQ = {}

        is_infeasible_state = state is None
        if not is_infeasible_state:
            self.initialize_mixed_q_values()
        self.mix_weight = 0.99

    def visualize_values_in_two_arm_domains(self, entity_values, entity_names):
        is_place_node = entity_names[0].find('region') != -1
        if is_place_node:
            return

        max_val = np.max(entity_values)
        min_val = np.min(entity_values)
        for entity_name, entity_value in zip(entity_names, entity_values):
            entity = openravepy.RaveGetEnvironments()[0].GetKinBody(entity_name)
            set_color(entity, [0, (entity_value - min_val) / (max_val - min_val), 0])
        try:
            visualize_path(self.state.in_way.minimum_constraint_path_to_entity['home_region'])
        except:
            import pdb;
            pdb.set_trace()

    def visualize_values_in_one_arm_domains(self, entity_names, entity_values):
        max_val = np.max(entity_values)
        min_val = np.min(entity_values)
        argsorted = np.argsort(entity_values)
        sorted_entity_names = np.array(entity_names)[argsorted]
        sorted_entity_values = np.sort(entity_values)
        n_entities = len(np.unique(np.sort(entity_values)))
        idx = 0
        prev_value = sorted_entity_values[0]
        for entity_name, entity_value in zip(sorted_entity_names, sorted_entity_values):
            entity = openravepy.RaveGetEnvironments()[0].GetKinBody(entity_name)
            set_color(entity, [0, float(idx) / n_entities, 0])
            print entity_name
            if entity_value != prev_value:
                prev_value = entity_value
                idx += 2
        set_color(openravepy.RaveGetEnvironments()[0].GetKinBody('r_obst0'), [0, 0, 1])
        visualize_path(self.state.in_way.minimum_constraint_path_to_entity['r_obst0'])

    def initialize_mixed_q_values(self):
        is_goal_idx = -3
        is_reachable_idx = -2
        is_pick_node = self.A[0].type.find('pick') != -1

        entity_values = []
        entity_names = []
        for a in self.A:
            self.mixedQ[a] = self.q_function.predict(self.state, a)[0]
            self.learnedQ[a] = self.mixedQ[a]
            # print self.state.nodes[a.discrete_parameters['object'].GetName()][is_reachable_idx], \

            if is_pick_node:
                discrete_param = a.discrete_parameters['object'].GetName()
            else:
                discrete_param = a.discrete_parameters['region'].name

            discrete_param_node = self.state.nodes[discrete_param]
            entity_values.append(self.learnedQ[a])
            entity_names.append(discrete_param)
            # things to print:
            #   - is reachable
            #   - is goal idx
            #   - is in the goal region
            #   - is in the way to the goal region
            is_reachable = discrete_param_node[is_reachable_idx]
            is_goal_entity = discrete_param_node[is_goal_idx]
            is_in_goal_region = self.state.edges[(discrete_param, 'home_region')][1]
            is_in_way_to_goal_region = self.state.edges[(discrete_param, 'home_region')][0]

            literal = "is_reachable %r is_goal %r is_in_goal_region %r is_in_way_to_goal_region %r" \
                      % (is_reachable, is_goal_entity, is_in_goal_region, is_in_way_to_goal_region)
            print literal, discrete_param, self.mixedQ[a]

            # print discrete_param_node[is_reachable_idx], discrete_param_node[is_goal_idx], discrete_param, \
            #    self.mixedQ[a]

        # self.visualize_values_in_two_arm_domains(entity_values, entity_names)
        # self.visualize_values_in_one_arm_domains(entity_names, entity_values)
        import pdb;
        pdb.set_trace()

        # if a.type.find('place') != -1:
        #    import pdb;pdb.set_trace()

        """
        if a.type.find('pick') != -1:
            goal_obj = 'square_packing_box1'
            reachable_obj = 'square_packing_box2'
            goal_idx = self.q_function.entity_name_to_idx[goal_obj]
            reachable_idx = self.q_function.entity_name_to_idx[reachable_obj]
            nodes, edges, actions = self.q_function.make_raw_format(self.state, a)
            n_entities = len(nodes)
            nodes = nodes[:, 6:]
            nodes = nodes.reshape((1, n_entities, 5))
            edges = edges.reshape((1, n_entities, n_entities, 2))
            actions = actions.reshape((1, 1))
            import pickle
            dest_vals = self.q_function.dest_model.predict(nodes).squeeze()
            sender_vals = self.q_function.sender_model.predict(nodes).squeeze()
            #one_arm_sender, one_arm_dest_vals = pickle.load(open('tmp.pkl', 'r'))
            edge_vals = self.q_function.edge_model.predict(edges).squeeze()
            msgs = self.q_function.msg_model.predict([nodes, edges, actions]).squeeze()
            values = self.q_function.value_model.predict([nodes, edges, actions]).squeeze()
            print "All edge values the same", np.all(edges[0, reachable_idx, :] == edges[0, goal_idx, :]), np.all(edges[0, :, reachable_idx] == edges[0, :, goal_idx])
        import pdb;pdb.set_trace()
        """

    def update_node_statistics(self, action, sum_rewards, reward):
        DiscreteTreeNode.update_node_statistics(self, action, sum_rewards, reward)
        self.update_mixed_q_value(action)

    def update_mixed_q_value(self, action):
        weight_on_learned_q = np.power(self.mix_weight, self.N[action])
        self.mixedQ[action] = weight_on_learned_q * self.learnedQ[action] + (1 - weight_on_learned_q) * self.Q[action]

    def perform_ucb_over_actions(self, learned_q_functions=None):
        q_vals = [self.mixedQ[a] for a in self.A]
        best_action = self.get_action_with_highest_ucb_value(self.A, q_vals)
        return best_action
