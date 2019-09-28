from uniform import PaPUniformGenerator

import numpy as np


class LearnedGenerator(PaPUniformGenerator):
    def __init__(self, operator_skeleton, problem_env, sampler, state, swept_volume_constraint=None):
        PaPUniformGenerator.__init__(self, operator_skeleton, problem_env, swept_volume_constraint)
        self.feasible_pick_params = {}
        self.sampler = sampler
        self.state = state

    def generate(self):
        if "Pose" in self.sampler.__module__:
            poses = np.array([np.hstack([self.state.obj_pose, self.state.robot_wrt_obj]).squeeze()])
            pick_place_base_poses = self.sampler.generate(self.state.state_vec, poses)  # I need grasp parameters;
        else:
            pick_place_base_poses = self.sampler.generate(self.state.state_vec)  # I need grasp parameters;
        return pick_place_base_poses

    def sample_feasible_op_parameters(self, operator_skeleton, n_iter, n_parameters_to_try_motion_planning):
        assert n_iter > 0
        feasible_op_parameters = []
        for i in range(n_iter):
            # fix it to take in the pose
            pick_place_base_poses = self.generate()
            grasp_parameters = self.sample_from_uniform()[0:3][None, :]
            op_parameters = np.hstack([grasp_parameters, pick_place_base_poses]).squeeze()
            op_parameters, status = self.op_feasibility_checker.check_feasibility(operator_skeleton, op_parameters,
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

    def sample_next_point(self, operator_skeleton, n_iter, n_parameters_to_try_motion_planning=1,
                          cached_collisions=None, cached_holding_collisions=None, dont_check_motion_existence=False):
        # Not yet motion-planning-feasible
        target_obj = operator_skeleton.discrete_parameters['object']
        if target_obj in self.feasible_pick_params:
            self.op_feasibility_checker.feasible_pick = self.feasible_pick_params[target_obj]

        status = "NoSolution"
        for n_iter in range(10, n_iter, 10):
            feasible_op_parameters, status = self.sample_feasible_op_parameters(operator_skeleton,
                                                                                n_iter,
                                                                                n_parameters_to_try_motion_planning)
            if status == 'HasSolution':
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
