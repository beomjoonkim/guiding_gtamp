from uniform import PaPUniformGenerator
from generators.learning.utils.data_processing_utils import action_data_mode
from generators.learning.utils.sampler_utils import generate_smpls
from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.learning.utils import data_processing_utils

from gtamp_utils import utils
from generators.learning.utils.sampler_utils import generate_policy_smpl_batch
from generators.learning.RelKonfIMLE import noise
import time
import numpy as np
import pickle
import os


class LearnedGenerator(PaPUniformGenerator):
    def __init__(self, operator_skeleton, problem_env, sampler, abstract_state, max_n_iter,
                 swept_volume_constraint=None):
        PaPUniformGenerator.__init__(self, operator_skeleton, problem_env, max_n_iter, swept_volume_constraint)
        self.feasible_pick_params = {}
        self.sampler = sampler
        self.abstract_state = abstract_state
        self.obj = operator_skeleton.discrete_parameters['object']
        self.region = operator_skeleton.discrete_parameters['region']

        goal_entities = self.abstract_state.goal_entities
        self.smpler_state = ConcreteNodeState(self.problem_env, self.obj, self.region,
                                              goal_entities,
                                              collision_vector=abstract_state.key_config_obstacles)
        self.noises_used = []
        self.tried_smpls = []

        # to do generate 1000 smpls here
        n_total_iters = sum(range(10, self.max_n_iter, 10))

        # generate n_total_iters number of samples -  can I be systematic about this, instead of random smpling?
        # I guess they will have to live in the same space; but I need to increase the variance

        """
        z_smpl_fname = 'z_smpls.pkl'
        if os.path.isfile(z_smpl_fname):
            z_smpls = pickle.load(open(z_smpl_fname, 'r'))
        else:
            z_smpls = []
            i = 0
            for _ in range(n_total_iters):  # even pre-store these
                if 0 < len(z_smpls) < 50:
                    min_dist = np.min(np.linalg.norm(new_z - np.array(z_smpls), axis=-1))
                    while min_dist < 1:
                        new_z = i * np.random.normal(size=(1, 4)).astype('float32')
                        min_dist = np.min(np.linalg.norm(new_z - np.array(z_smpls), axis=-1))
                        i += 1
                else:
                    new_z = np.random.normal(size=(1, 4)).astype('float32')
                z_smpls.append(new_z)
            pickle.dump(z_smpls, open(z_smpl_fname, 'wb'))
        """
        z_smpls = noise(z_size=(1900, 4))
        z_smpls = np.vstack([np.array([0, 0, 0, 0]), z_smpls])
        self.policy_smpl_batch = generate_policy_smpl_batch(self.smpler_state, self.sampler, z_smpls)
        self.policy_smpl_idx = 0

    def generate(self):
        if action_data_mode == 'pick_parameters_place_relative_to_object':
            # place_smpls, noises_used = generate_smpls(self.smpler_state, self.sampler, 1, self.noises_used)
            # place_smpls = place_smpls[0].squeeze()
            place_smpl = self.policy_smpl_batch[self.policy_smpl_idx]
            place_smpl = data_processing_utils.get_unprocessed_placement(place_smpl, self.smpler_state.abs_obj_pose)
            #place_smpls = [data_processing_utils.get_unprocessed_placement(s, self.smpler_state.abs_obj_pose) for s in self.policy_smpl_batch]
            self.policy_smpl_idx += 1
        else:
            raise NotImplementedError
        self.tried_smpls.append(place_smpl)
        # self.noises_used = noises_used
        # if self.smpler_state.obj == 'square_packing_box2':
        #    import pdb;
        #    pdb.set_trace()
        parameters = self.sample_from_uniform()
        parameters[6:] = place_smpl
        return parameters

    def sample_feasible_op_parameters(self, operator_skeleton, n_iter, n_parameters_to_try_motion_planning):
        assert n_iter > 0
        feasible_op_parameters = []
        obj = operator_skeleton.discrete_parameters['object']

        orig_color = utils.get_color_of(obj)
        #utils.set_color(obj, [1, 0, 0])
        #utils.viewer()
        for i in range(n_iter):
            # print 'Sampling attempts %d/%d' %(i,n_iter)
            # fix it to take in the pose
            stime = time.time()
            op_parameters = self.generate()
            op_parameters, status = self.op_feasibility_checker.check_feasibility(operator_skeleton, op_parameters,
                                                                                  self.swept_volume_constraint,
                                                                                  parameter_mode='robot_base_pose')
            # if we have sampled the feasible place, then can we keep that?
            # if we have infeasible pick, then we cannot determine that.
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
            # import pdb;pdb.set_trace()
            status = "HasSolution"
        utils.set_color(obj, orig_color)
        return feasible_op_parameters, status

    """
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
        """
