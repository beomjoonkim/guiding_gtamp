from uniform import PaPUniformGenerator
from generators.learning.utils.data_processing_utils import action_data_mode
from generators.learning.utils.sampler_utils import generate_smpls
from trajectory_representation.concrete_node_state import ConcreteNodeState

from gtamp_utils import utils


class LearnedGenerator(PaPUniformGenerator):
    def __init__(self, operator_skeleton, problem_env, sampler, abstract_state, swept_volume_constraint=None):
        PaPUniformGenerator.__init__(self, operator_skeleton, problem_env, swept_volume_constraint)
        self.feasible_pick_params = {}
        self.sampler = sampler
        self.abstract_state = abstract_state
        self.obj = operator_skeleton.discrete_parameters['object']
        self.region = operator_skeleton.discrete_parameters['region']

        # todo make the concrete state to be used to generate samples
        goal_entities = self.abstract_state.goal_entities
        self.smpler_state = ConcreteNodeState(self.problem_env, self.obj, self.region,
                                              goal_entities,
                                              collision_vector=abstract_state.key_config_obstacles)

    def generate(self):
        if action_data_mode == 'pick_parameters_place_relative_to_object':
            place_smpls = generate_smpls(self.smpler_state, self.sampler, 1)[0].squeeze()
        else:
            raise NotImplementedError
        if self.smpler_state.obj == 'square_packing_box2':
            import pdb;pdb.set_trace()
        parameters = self.sample_from_uniform()
        parameters[6:] = place_smpls
        # todo I am supposed to predict the object pose...
        return parameters

    def sample_feasible_op_parameters(self, operator_skeleton, n_iter, n_parameters_to_try_motion_planning):
        assert n_iter > 0
        feasible_op_parameters = []
        for i in range(n_iter):
            # fix it to take in the pose
            op_parameters = self.generate()
            op_parameters, status = self.op_feasibility_checker.check_feasibility(operator_skeleton, op_parameters,
                                                                                  self.swept_volume_constraint,
                                                                                  parameter_mode='robot_base_pose')

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
