from generator import PaPGenerator, Generator
from gtamp_utils import utils
from gtamp_utils.utils import get_pick_domain, get_place_domain
# from mover_library import utils
# from mover_library.utils import get_pick_domain, get_place_domain

from feasibility_checkers.two_arm_pick_feasibility_checker import TwoArmPickFeasibilityChecker
from feasibility_checkers.two_arm_place_feasibility_checker import TwoArmPlaceFeasibilityChecker
from feasibility_checkers.one_arm_pick_feasibility_checker import OneArmPickFeasibilityChecker
from feasibility_checkers.one_arm_place_feasibility_checker import OneArmPlaceFeasibilityChecker
from feasibility_checkers.two_arm_pap_feasiblity_checker import TwoArmPaPFeasibilityChecker

import numpy as np
import time


class UniformGenerator:
    def __init__(self, operator_skeleton, problem_env, max_n_iter, swept_volume_constraint=None):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.evaled_actions = []
        self.evaled_q_values = []
        self.swept_volume_constraint = swept_volume_constraint
        self.objects_to_check_collision = None
        self.tried_smpls = []
        self.smpling_time = []
        self.max_n_iter = max_n_iter
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

    def sample_feasible_op_parameters(self, operator_skeleton, n_iter, n_parameters_to_try_motion_planning):
        assert n_iter > 0
        feasible_op_parameters = []
        for i in range(n_iter):
            # print 'Sampling attempts %d/%d' % (i, n_iter)
            stime = time.time()
            op_parameters = self.sample_from_uniform()
            self.tried_smpls.append(op_parameters)
            # is this sampling the absolute or relative pick base config?
            # It is sampling the relative config, but q_goal is the absolute.
            op_parameters, status = self.op_feasibility_checker.check_feasibility(operator_skeleton,
                                                                                  op_parameters,
                                                                                  self.swept_volume_constraint)

            smpling_time = time.time() - stime
            self.smpling_time.append(smpling_time)
            if status == 'HasSolution':
                feasible_op_parameters.append(op_parameters)
                if len(feasible_op_parameters) >= n_parameters_to_try_motion_planning:
                    break

        if len(feasible_op_parameters) == 0:
            feasible_op_parameters.append(op_parameters)  # place holder
            status = "NoSolution"
        else:
            status = "HasSolution"

        return feasible_op_parameters, status

    @staticmethod
    def choose_one_of_params(params, status):
        sampled_feasible_parameters = status == "HasSolution"

        if sampled_feasible_parameters:
            chosen_op_param = params[0]
            chosen_op_param['motion'] = [chosen_op_param['q_goal']]
            chosen_op_param['is_feasible'] = True
        else:
            chosen_op_param = {'is_feasible': False}

        return chosen_op_param

    def sample_next_point(self, operator_skeleton, n_iter, n_parameters_to_try_motion_planning=1,
                          cached_collisions=None, dont_check_motion_existence=False):
        # Not yet motion-planning-feasible
        feasible_op_parameters, status = self.sample_feasible_op_parameters(operator_skeleton,
                                                                            n_iter,
                                                                            n_parameters_to_try_motion_planning)
        if dont_check_motion_existence:
            chosen_op_param = self.choose_one_of_params(feasible_op_parameters, status)
        else:
            chosen_op_param = self.get_op_param_with_feasible_motion_plan(feasible_op_parameters, cached_collisions)

        return chosen_op_param

    def get_op_param_with_feasible_motion_plan(self, feasible_op_params, cached_collisions):
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

    def sample_from_uniform(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()


class PaPUniformGenerator(UniformGenerator):
    def __init__(self, operator_skeleton, problem_env, max_n_iter, swept_volume_constraint=None):
        UniformGenerator.__init__(self, operator_skeleton, problem_env, max_n_iter, swept_volume_constraint)
        self.feasible_pick_params = {}

    def sample_next_point(self, operator_skeleton, n_parameters_to_try_motion_planning=1,
                          cached_collisions=None, cached_holding_collisions=None, dont_check_motion_existence=False):
        # Not yet motion-planning-feasible
        target_obj = operator_skeleton.discrete_parameters['object']
        if target_obj in self.feasible_pick_params:
            self.op_feasibility_checker.feasible_pick = self.feasible_pick_params[target_obj]

        status = "NoSolution"
        stime = time.time()
        for curr_n_iter in range(10, self.max_n_iter, 10):
            feasible_op_parameters, status = self.sample_feasible_op_parameters(operator_skeleton,
                                                                                curr_n_iter,
                                                                                n_parameters_to_try_motion_planning)

            if status == 'HasSolution':
                # Don't break here, but try to get more parameters
                break
                """
                if dont_check_motion_existence:
                    chosen_op_param = self.choose_one_of_params(feasible_op_parameters, status)
                    return chosen_op_param
                else:
                    chosen_op_param = self.get_pap_param_with_feasible_motion_plan(operator_skeleton,
                                                                                   feasible_op_parameters,
                                                                                   cached_collisions,
                                                                                   cached_holding_collisions)
                    if chosen_op_param['is_feasible']:
                        return chosen_op_param
                """
        print "Time taken", time.time() - stime, status

        if status == "NoSolution":
            return {'is_feasible': False}

        # We would have to move these to the loop in order to be fair
        if dont_check_motion_existence:
            chosen_op_param = self.choose_one_of_params(feasible_op_parameters, status)
        else:
            chosen_op_param = self.get_pap_param_with_feasible_motion_plan(operator_skeleton,
                                                                           feasible_op_parameters,
                                                                           cached_collisions,
                                                                           cached_holding_collisions)
        return chosen_op_param

    def get_pap_param_with_feasible_motion_plan(self, operator_skeleton, feasible_op_parameters,
                                                cached_collisions, cached_holding_collisions):
        # getting pick motion - I can still use the cached collisions from state computation
        pick_op_params = [op['pick'] for op in feasible_op_parameters]
        chosen_pick_param = self.get_op_param_with_feasible_motion_plan(pick_op_params, cached_collisions)
        if not chosen_pick_param['is_feasible']:
            return {'is_feasible': False}

        target_obj = operator_skeleton.discrete_parameters['object']
        if target_obj in self.feasible_pick_params:
            self.feasible_pick_params[target_obj].append(chosen_pick_param)
        else:
            self.feasible_pick_params[target_obj] = [chosen_pick_param]

        # self.feasible_pick_params[target_obj].append(pick_op_params)

        # getting place motion
        original_config = utils.get_body_xytheta(self.problem_env.robot).squeeze()
        utils.two_arm_pick_object(operator_skeleton.discrete_parameters['object'], chosen_pick_param)
        place_op_params = [op['place'] for op in feasible_op_parameters]
        chosen_place_param = self.get_op_param_with_feasible_motion_plan(place_op_params, cached_holding_collisions)
        utils.two_arm_place_object(chosen_pick_param)
        utils.set_robot_config(original_config)

        if not chosen_place_param['is_feasible']:
            return {'is_feasible': False}

        chosen_pap_param = {'pick': chosen_pick_param, 'place': chosen_place_param, 'is_feasible': True}
        return chosen_pap_param


# It is a mess at the moment: I am using the below for computing the state, but above for sampling in SAHS,
# to be consistent with the old performance. Will fix later.

class UniformPaPGenerator(PaPGenerator):
    def __init__(self, node, operator_skeleton, problem_env, swept_volume_constraint,
                 total_number_of_feasibility_checks, n_candidate_params_to_smpl, dont_check_motion_existence):
        PaPGenerator.__init__(self, node, operator_skeleton, problem_env, swept_volume_constraint,
                              total_number_of_feasibility_checks, n_candidate_params_to_smpl,
                              dont_check_motion_existence)

    def sample_candidate_pap_parameters(self, iter_limit):
        assert iter_limit > 0
        feasible_op_parameters = []
        for i in range(iter_limit):
            op_parameters = self.sample_from_uniform()
            op_parameters, status = self.op_feasibility_checker.check_feasibility(self.operator_skeleton,
                                                                                  op_parameters,
                                                                                  self.swept_volume_constraint)

            if status == 'HasSolution':
                feasible_op_parameters.append(op_parameters)
                if len(feasible_op_parameters) >= self.n_candidate_params_to_smpl:
                    break

        if len(feasible_op_parameters) == 0:
            feasible_op_parameters.append(op_parameters)  # place holder
            status = "NoSolution"
        else:
            status = "HasSolution"

        return feasible_op_parameters, status
