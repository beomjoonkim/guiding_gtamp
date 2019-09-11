from gtamp_utils.samplers import *
from gtamp_utils.utils import get_pick_domain, get_place_domain
from gtamp_utils import utils

from feasibility_checkers.two_arm_pick_feasibility_checker import TwoArmPickFeasibilityChecker
from feasibility_checkers.two_arm_place_feasibility_checker import TwoArmPlaceFeasibilityChecker
from feasibility_checkers.one_arm_pick_feasibility_checker import OneArmPickFeasibilityChecker
from feasibility_checkers.one_arm_place_feasibility_checker import OneArmPlaceFeasibilityChecker
from feasibility_checkers.two_arm_pap_feasiblity_checker import TwoArmPaPFeasibilityChecker
from planners.mcts_utils import make_action_executable


class Generator:
    def __init__(self, node, operator_skeleton, problem_env, swept_volume_constraint,
                 total_number_of_feasibility_checks, n_candidate_params_to_smpl, dont_check_motion_existence):
        self.total_number_of_feasibility_checks = total_number_of_feasibility_checks
        self.n_candidate_params_to_smpl = n_candidate_params_to_smpl
        self.node = node

        self.problem_env = problem_env
        self.env = problem_env.env
        self.evaled_actions = []
        self.evaled_q_values = []
        self.swept_volume_constraint = swept_volume_constraint
        self.objects_to_check_collision = None
        operator_type = operator_skeleton.type
        self.operator_skeleton = operator_skeleton
        self.dont_check_motion_existence = dont_check_motion_existence

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

    def get_op_param_with_feasible_motion_plan(self, feasible_op_params, cached_collisions):
        # from the multiple operator continuous parameters, return the one that has the feasible motion
        motion_plan_goals = [op['q_goal'] for op in feasible_op_params]
        motion, status = self.problem_env.motion_planner.get_motion_plan(motion_plan_goals,
                                                                         cached_collisions=cached_collisions)
        found_feasible_motion_plan = status == "HasSolution"
        if found_feasible_motion_plan:
            which_op_param = np.argmin(np.linalg.norm(motion[-1] - motion_plan_goals, axis=-1))
            chosen_op_param = feasible_op_params[which_op_param]
            chosen_op_param['motion'] = motion
            chosen_op_param['is_feasible'] = True
        else:
            chosen_op_param = feasible_op_params[0]
            chosen_op_param['is_feasible'] = False

        return chosen_op_param

    @staticmethod
    def choose_one_of_params(params, status):
        sampled_feasible_parameters = status == "HasSolution"

        chosen_op_param = params[0]
        if sampled_feasible_parameters:
            chosen_op_param['motion'] = [chosen_op_param['q_goal']]
            chosen_op_param['is_feasible'] = True
        else:
            chosen_op_param['is_feasible'] = False

        return chosen_op_param

    def update_evaled_values(self):
        executed_actions_in_node = self.node.Q.keys()
        executed_action_values_in_node = self.node.Q.values()

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

    def sample_next_point(self, n_iter):
        raise NotImplementedError

    def sample_from_uniform(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()


class PaPGenerator(Generator):
    def __init__(self, node, operator_skeleton, problem_env, swept_volume_constraint,
                 total_number_of_feasibility_checks, n_candidate_params_to_smpl, dont_check_motion_existence):
        Generator.__init__(self, node, operator_skeleton, problem_env, swept_volume_constraint,
                           total_number_of_feasibility_checks, n_candidate_params_to_smpl, dont_check_motion_existence)
        self.motion_verified_pick_params = {}

    def get_place_param_with_feasible_motion_plan(self, chosen_pick_param, candidate_pap_parameters,
                                                  cached_holding_collisions):
        original_config = utils.get_body_xytheta(self.problem_env.robot).squeeze()
        utils.two_arm_pick_object(self.operator_skeleton.discrete_parameters['object'], chosen_pick_param)
        place_op_params = [op['place'] for op in candidate_pap_parameters]
        chosen_place_param = self.get_op_param_with_feasible_motion_plan(place_op_params, cached_holding_collisions)
        utils.two_arm_place_object(chosen_pick_param)
        utils.set_robot_config(original_config)

        return chosen_place_param

    def get_pick_param_with_feasible_motion_plan(self, candidate_pap_parameters, cached_collisions):
        pick_op_params = [op['pick'] for op in candidate_pap_parameters]
        chosen_pick_param = self.get_op_param_with_feasible_motion_plan(pick_op_params, cached_collisions)
        return chosen_pick_param

    def save_feasible_pick_params(self, chosen_pick_param):
        # For keeping pick params if we have sampled feasible pick parameters but not place parameters
        target_obj = self.operator_skeleton.discrete_parameters['object']
        if target_obj in self.motion_verified_pick_params:  # what is this object used for?
            self.motion_verified_pick_params[target_obj].append(chosen_pick_param)
        else:
            self.motion_verified_pick_params[target_obj] = [chosen_pick_param]

    def get_pap_param_with_feasible_motion_plan(self, candidate_pap_parameters,
                                                cached_collisions, cached_holding_collisions):
        chosen_pick_param = self.get_pick_param_with_feasible_motion_plan(candidate_pap_parameters, cached_collisions)
        if not chosen_pick_param['is_feasible']:
            return candidate_pap_parameters[0]

        self.save_feasible_pick_params(chosen_pick_param)
        chosen_place_param = self.get_place_param_with_feasible_motion_plan(chosen_pick_param,
                                                                            candidate_pap_parameters,
                                                                            cached_holding_collisions)
        if not chosen_place_param['is_feasible']:
            return candidate_pap_parameters[0]

        chosen_pap_action_parameters = np.hstack([chosen_pick_param['action_parameters'],
                                                  chosen_place_param['action_parameters']])
        chosen_pap_param = {'pick': chosen_pick_param,
                            'place': chosen_place_param,
                            'action_parameters': chosen_pap_action_parameters,
                            'is_feasible': True}
        return chosen_pap_param

    def sample_candidate_params_with_increasing_iteration_limit(self):
        status = "NoSolution"
        candidate_op_parameters = None
        for iter_limit in range(10, self.total_number_of_feasibility_checks, 10):
            candidate_op_parameters, status = self.sample_candidate_pap_parameters(iter_limit)
            if status == 'HasSolution' and len(candidate_op_parameters) >= self.n_candidate_params_to_smpl:
                break
        return candidate_op_parameters, status

    def sample_next_point(self, cached_collisions=None, cached_holding_collisions=None):
        target_obj = self.operator_skeleton.discrete_parameters['object']

        # For re-using pick params if we have sampled feasible pick parameters but not place parameters
        if target_obj in self.motion_verified_pick_params:
            self.op_feasibility_checker.feasible_pick = self.motion_verified_pick_params[target_obj]

        # sample parameters whose feasibility have been checked except the existence of collision-free motion
        candidate_op_parameters, status = self.sample_candidate_params_with_increasing_iteration_limit()
        if status == "NoSolution":
            candidate_op_parameters[0]['is_feasible'] = False
            return candidate_op_parameters[0]
        if self.dont_check_motion_existence:
            chosen_op_param = self.choose_one_of_params(candidate_op_parameters, status)
        else:
            chosen_op_param = self.get_pap_param_with_feasible_motion_plan(candidate_op_parameters,
                                                                           cached_collisions,
                                                                           cached_holding_collisions)
        return chosen_op_param

    def sample_candidate_pap_parameters(self, iter_limit):
        raise NotImplementedError





