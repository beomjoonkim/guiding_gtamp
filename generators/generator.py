from gtamp_utils.samplers import *
from gtamp_utils.utils import get_pick_domain, get_place_domain

from feasibility_checkers.two_arm_pick_feasibility_checker import TwoArmPickFeasibilityChecker
from feasibility_checkers.two_arm_place_feasibility_checker import TwoArmPlaceFeasibilityChecker
from feasibility_checkers.one_arm_pick_feasibility_checker import OneArmPickFeasibilityChecker
from feasibility_checkers.one_arm_place_feasibility_checker import OneArmPlaceFeasibilityChecker
from feasibility_checkers.two_arm_pap_feasiblity_checker import TwoArmPaPFeasibilityChecker
from planners.mcts_utils import make_action_executable


class Generator:
    def __init__(self, operator_skeleton, problem_env, swept_volume_constraint):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.evaled_actions = []
        self.evaled_q_values = []
        self.swept_volume_constraint = swept_volume_constraint
        self.objects_to_check_collision = None
        operator_type = operator_skeleton.type

        target_region = None
        if 'region' in operator_skeleton.discrete_parameters:
            target_region = operator_skeleton.discrete_parameters['region']
            if type(target_region) == str:
                target_region = self.problem_env.regions[target_region]
        if 'two_arm_place_region' in operator_skeleton.discrete_parameters:
            target_region = operator_skeleton.discrete_parameters['two_arm_place_region']
            if type(target_region) == str:
                target_region = self.problem_env.regions[target_region]

        if operator_type == 'two_arm_pick':
            self.domain = get_pick_domain()
            self.op_feasibility_checker = TwoArmPickFeasibilityChecker(problem_env)
        elif operator_type == 'one_arm_pick':
            self.domain = get_pick_domain()
            self.op_feasibility_checker = OneArmPickFeasibilityChecker(problem_env)
        elif operator_type == 'two_arm_place':
            self.domain = get_place_domain(target_region)
            self.op_feasibility_checker = TwoArmPlaceFeasibilityChecker(problem_env)
        elif operator_type == 'one_arm_place':
            self.domain = get_place_domain(target_region)
            self.op_feasibility_checker = OneArmPlaceFeasibilityChecker(problem_env)
        elif operator_type == 'two_arm_pick_two_arm_place':
            # used by MCTS
            pick_min = get_pick_domain()[0]
            pick_max = get_pick_domain()[1]
            place_min = get_place_domain(target_region)[0]
            place_max = get_place_domain(target_region)[1]
            mins = np.hstack([pick_min, place_min])
            maxes = np.hstack([pick_max, place_max])
            self.domain = np.vstack([mins, maxes])
            self.op_feasibility_checker = TwoArmPaPFeasibilityChecker(problem_env)
        elif operator_type == 'one_arm_pick_one_arm_place':
            self.pick_feasibility_checker = OneArmPickFeasibilityChecker(problem_env)
            self.place_feasibility_checker = OneArmPlaceFeasibilityChecker(problem_env)
            pick_min = get_pick_domain()[0]
            pick_max = get_pick_domain()[1]
            place_min = get_place_domain(target_region)[0]
            place_max = get_place_domain(target_region)[1]
            self.pick_domain = np.vstack([pick_min, pick_max])
            self.place_domain = np.vstack([place_min, place_max])
        else:
            raise ValueError

    def update_evaled_values(self, node):
        executed_actions_in_node = node.Q.keys()
        executed_action_values_in_node = node.Q.values()

        for action, q_value in zip(executed_actions_in_node, executed_action_values_in_node):
            executable_action = make_action_executable(action)
            is_in_array = [np.array_equal(executable_action['action_parameters'], a) for a in self.evaled_actions]
            is_action_included = np.any(is_in_array)

            if not is_action_included:
                self.evaled_actions.append(executable_action['action_parameters'])
                self.evaled_q_values.append(q_value)
            else:
                # update the value if the action is included
                self.evaled_q_values[np.where(is_in_array)[0][0]] = q_value

    def sample_next_point(self, node, n_iter):
        raise NotImplementedError

    def sample_from_uniform(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()

