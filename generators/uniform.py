from generator import PaPGenerator, Generator
from gtamp_utils import utils



class UniformGenerator(Generator):
    def __init__(self, operator_skeleton, problem_env, swept_volume_constraint=None):
        Generator.__init__(self, operator_skeleton, problem_env, swept_volume_constraint, 1, 1, False)


class UniformPaPGenerator(PaPGenerator):
    def __init__(self, operator_skeleton, problem_env, swept_volume_constraint,
                 total_number_of_feasibility_checks, n_candidate_params_to_smpl, dont_check_motion_existence):
        PaPGenerator.__init__(self, operator_skeleton, problem_env, swept_volume_constraint,
                              total_number_of_feasibility_checks, n_candidate_params_to_smpl, dont_check_motion_existence)

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


