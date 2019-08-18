from generator import Generator
import numpy as np
import time

from gtamp_utils import utils
import cProfile


class UniformGenerator(Generator):
    def __init__(self, operator_skeleton, problem_env, swept_volume_constraint=None):
        Generator.__init__(self, operator_skeleton, problem_env, swept_volume_constraint)

    def sample_feasible_op_parameters(self, operator_skeleton, n_iter, n_parameters_to_try_motion_planning):
        assert n_iter > 0
        feasible_op_parameters = []
        for i in range(n_iter):
            op_parameters = self.sample_from_uniform()
            op_parameters, status = self.op_feasibility_checker.check_feasibility(operator_skeleton,
                                                                                  op_parameters,
                                                                                  self.swept_volume_constraint)

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

    def sample_next_point(self, node, n_iter, n_parameters_to_try_motion_planning=1,
                          cached_collisions=None, dont_check_motion_existence=False):

        operator_skeleton = node.operator_skeleton
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


class PaPUniformGenerator(UniformGenerator):
    def __init__(self, operator_skeleton, problem_env, swept_volume_constraint=None):
        UniformGenerator.__init__(self, operator_skeleton, problem_env, swept_volume_constraint)
        self.feasible_pick_params = {}

    def sample_next_point(self, node, n_iter, n_parameters_to_try_motion_planning=1,
                          cached_collisions=None, cached_holding_collisions=None, dont_check_motion_existence=False):
        # Not yet motion-planning-feasible
        operator_skeleton = node.operator_skeleton
        target_obj = operator_skeleton.discrete_parameters['object']
        if target_obj in self.feasible_pick_params:
            self.op_feasibility_checker.feasible_pick = self.feasible_pick_params[target_obj]

        status = "NoSolution"
        for n_iter in range(10, n_iter, 10):
            feasible_op_parameters, status = self.sample_feasible_op_parameters(operator_skeleton,
                                                                                n_iter,
                                                                                n_parameters_to_try_motion_planning)
            if status =='HasSolution':
                break
        if status == "NoSolution":
            return {'is_feasible': False}

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

        #self.feasible_pick_params[target_obj].append(pick_op_params)

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

