import numpy as np

def upper_confidence_bound(n, n_sa):
    return np.sqrt(2 * np.log(n + 1) / float(n_sa + 1))



class TreeNode:
    def __init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node):
        self.Nvisited = 0
        self.N = {}  # N(n,a)
        self.Q = {}  # Q(n,a)
        self.A = []  # traversed actions

        self.state = state
        self.parent = None
        self.children = {}
        self.parent_action = None
        self.sum_ancestor_action_rewards = 0  # for logging purpose
        self.sum_rewards_history = {}  # for debugging purpose
        self.reward_history = {}  # for debugging purpose
        self.ucb_parameter = ucb_parameter
        self.parent_motion = None
        self.is_init_node = False
        self.is_goal_node = False
        self.is_goal_and_already_visited = False
        self.depth = depth
        self.sum_rewards = 0
        self.operator = None
        self.sampling_agent = None  # todo define sampling agent
        self.is_operator_skeleton_node = is_operator_skeleton_node

        self.state_saver = state_saver
        self.is_init_node = is_init_node
        self.objects_in_collision = None
        self.n_ucb_iterations = 0
        self.idx = 1
        self.state = state

    def set_objects_in_collision(self, objects_in_collision):
        self.objects_in_collision = objects_in_collision

    def perform_ucb_over_actions(self):
        raise NotImplementedError

    def add_actions(self, actions):
        raise NotImplementedError

    def get_never_evaluated_action(self):
        # get list of actions that do not have an associated Q values
        no_evaled = [a for a in self.A if a not in self.Q.keys()]
        return np.random.choice(no_evaled)

    def is_descendent_of(self, node):
        ancestor = self
        while not (ancestor is None):
            if ancestor == node:
                return True
            ancestor = ancestor.parent
        return False

    def get_child_with_max_value(self):
        max_q = -np.inf
        max_child = None

        for child_action, child in self.children.items():
            child_value = self.Q[child_action]
            if child_value > max_q:
                max_q = child_value
                max_child = child

        return max_child

    def choose_new_arm(self):
        new_arm = self.A[-1]  # what to do if the new action is not a feasible one?
        is_new_arm_feasible = new_arm.continuous_parameters['is_feasible']
        try:
            assert is_new_arm_feasible
        except:
            # todo it runs in here. Figure out why.
            import pdb;
            pdb.set_trace()
        return new_arm

    def compute_ucb_value(self, value, action):
        return value + self.ucb_parameter * upper_confidence_bound(self.Nvisited, self.N[action])

    def get_action_with_highest_ucb_value(self, feasible_actions, feasible_q_values):
        best_value = -np.inf
        best_action = feasible_actions[0]

        for action, value in zip(feasible_actions, feasible_q_values):
            ucb_value = self.compute_ucb_value(value, action)

            if ucb_value > best_value:
                best_action = action
                best_value = ucb_value

        best_actions = []
        for action, value in zip(feasible_actions, feasible_q_values):
            ucb_value = self.compute_ucb_value(value, action)
            if ucb_value == best_value:
                best_actions.append(action)
        return best_actions[np.random.randint(len(best_actions))]

    def compute_ucb_values(self, feasible_actions, feasible_q_values):
        ucb_values = {action: self.compute_ucb_value(value, action)
                      for action, value in zip(feasible_actions, feasible_q_values)}
        return ucb_values

    def is_action_tried(self, action):
        return action in self.children
        # return action in self.Q.keys()

    def update_node_statistics(self, action, sum_rewards, reward):
        self.Nvisited += 1

        is_action_never_tried = self.N[action] == 0
        if is_action_never_tried:
            self.reward_history[action] = [reward]
            self.Q[action] = sum_rewards
            self.N[action] += 1
        else:
            self.reward_history[action].append(reward)
            self.N[action] += 1
            if sum_rewards > self.Q[action]:
                self.Q[action] = sum_rewards
