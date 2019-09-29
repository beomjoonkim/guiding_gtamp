from uniform import PaPUniformGenerator

from generators.learning.train_sampler import get_processed_poses_from_state
from gtamp_utils import utils
import numpy as np


class LearnedGenerator(PaPUniformGenerator):
    def __init__(self, operator_skeleton, problem_env, sampler, state, swept_volume_constraint=None):
        PaPUniformGenerator.__init__(self, operator_skeleton, problem_env, swept_volume_constraint)
        self.feasible_pick_params = {}
        self.sampler = sampler
        self.state = state

    def generate(self, operator_skeleton, action_data_mode='pick_relative_place_relative_to_region'):
        if "Pose" in self.sampler.__module__:
            poses = get_processed_poses_from_state(self.state, 'robot_rel_to_obj').reshape((1, 8))
            pick_place_base_poses = self.sampler.generate(self.state.state_vec, poses)  # I need grasp parameters;
        else:
            pick_place_base_poses = self.sampler.generate(self.state.state_vec)  # I need grasp parameters;
        pick_place_base_poses = pick_place_base_poses.squeeze()

        if action_data_mode == 'pick_relative_place_relative_to_region':
            relative_pick_pose_wrt_obj = utils.decode_pose_with_sin_and_cos_angle(pick_place_base_poses[:4])
            pick_pose = utils.get_global_pose_from_relative_pose_to_body(operator_skeleton.discrete_parameters['object'],
                                                                         relative_pick_pose_wrt_obj)
            place_pose = utils.decode_pose_with_sin_and_cos_angle(pick_place_base_poses[4:])
            if operator_skeleton.discrete_parameters['region'] == 'home_region':
                place_pose[0:2] += [-1.75, 5.25]
            elif operator_skeleton.discrete_parameters['region'] == 'loading_region':
                place_pose[0:2] += [-0.7, 4.3]
        else:
            raise NotImplementedError

        return pick_pose, place_pose

    def sample_feasible_op_parameters(self, operator_skeleton, n_iter, n_parameters_to_try_motion_planning):
        assert n_iter > 0
        feasible_op_parameters = []
        for i in range(n_iter):
            # fix it to take in the pose
            pick_place_base_poses = self.generate(operator_skeleton)
            import pdb;pdb.set_trace()
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
