import numpy as np
from gtamp_utils.utils import get_body_xytheta, set_obj_xytheta, set_robot_config
from planners.heuristics import get_objects_to_move

class MCTSTree:
    def __init__(self, exploration_parameters):
        self.nodes = []
        self.exploration_parameters = exploration_parameters
        self.root = None

    def make_tree_picklable(self):
        for node in self.nodes:
            node.sampling_agent = None
            for a in node.A:
                if 'object' in a.discrete_parameters.keys() and type(a.discrete_parameters['object']) != str:
                    a.discrete_parameters['object'] = str(a.discrete_parameters['object'].GetName())

            if node.objects_in_collision is not None:
                for idx in range(len(node.objects_in_collision)):
                    if type(node.objects_in_collision[idx]) != str:
                        node.objects_in_collision[idx] = str(node.objects_in_collision[idx].GetName())

            for a in node.N.keys():
                if 'object' in a.discrete_parameters.keys() and type(a.discrete_parameters['object']) != str:
                    a.discrete_parameters['object'] = str(a.discrete_parameters['object'].GetName())

            node.state = None

    def set_root_node(self, root_node):
        self.root = root_node
        self.nodes.append(root_node)

    def has_state(self, state):
        return len([n for n in self.nodes if np.all(n.state == state)]) > 0

    def add_node(self, node, action, parent):
        node.parent = parent
        parent.children[action] = node
        """
        if is_action_hashable(action):
            parent.children[action] = node
        else:
            parent.children[make_action_hashable(action)] = node
        """
        if node not in self.nodes:
            self.nodes.append(node)
        node.idx = len(self.nodes)

    def is_node_just_added(self, node):
        if node == self.root:
            return False

        for action, resulting_child in zip(node.parent.children.keys(), node.parent.children.values()):
            if resulting_child == node:
                return not (action in node.parent.A)  # action that got to the node is not in parent's actions

    def get_leaf_nodes(self):
        return [n for n in self.nodes if len(n.children.keys()) == 0]

    def get_goal_nodes(self):
        return [n for n in self.nodes if len(n.children.keys()) == 0 and n.is_goal_node]

    def get_instance_nodes(self):
        return [n for n in self.nodes if not n.is_operator_skeleton_node]

    def get_discrete_nodes(self):
        return [n for n in self.nodes if n.is_operator_skeleton_node]

    def get_best_trajectory_sum_rewards_and_node(self, discount_factor):
        sumR_list = []
        leaf_nodes_for_curr_init_state = []
        leaf_nodes = self.get_leaf_nodes()
        reward_lists = []

        for n in leaf_nodes:
            curr_node = n

            reward_list = []
            while not curr_node.is_init_node and curr_node.parent is not None:
                reward_list.append(curr_node.parent.reward_history[curr_node.parent_action][0])
                curr_node = curr_node.parent

            if (curr_node.parent is None) and (not curr_node.is_init_node):
                continue

            reward_lists.append(np.sum(reward_list[::-1]))
            discount_rates = [np.power(discount_factor, i) for i in range(len(reward_list))]
            sumR = np.dot(discount_rates[::-1], reward_list)

            sumR_list.append(sumR)
            leaf_nodes_for_curr_init_state.append(n)

        #progress = self.compute_number_of_boxes_packed_in_mover_domain(leaf_nodes, sumR_list)
        best_node = leaf_nodes[np.argmax(sumR_list)]
        progress = np.min([self.get_node_hcount(n) for n in leaf_nodes])
        return np.max(sumR_list), progress, best_node

    def get_node_hcount(self, node):
        is_infeasible_parent_action = node.state is None
        if is_infeasible_parent_action:
            return len(get_objects_to_move(node.parent.state, node.parent.state.problem_env))
        else:
            return len(get_objects_to_move(node.state, node.state.problem_env))

    def compute_number_of_boxes_packed_in_mover_domain(self, leaf_nodes, sumR_list):
        best_leaf_node = leaf_nodes[np.argmax(sumR_list)]
        curr_node = best_leaf_node
        progress = 0
        while curr_node.parent is not None:
            if curr_node.parent_action_reward > 0:
                progress += 1
            elif curr_node.parent_action_reward < 0:
                progress -= 1
            curr_node = curr_node.parent

        return progress


