from gtamp_utils.utils import set_robot_config, \
    two_arm_pick_object, two_arm_place_object
from gtamp_utils.operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp
from generators.feasibility_checkers.two_arm_pick_feasibility_checker import TwoArmPickFeasibilityChecker
from generators.feasibility_checkers.two_arm_place_feasibility_checker import TwoArmPlaceFeasibilityChecker
from trajectory_representation.operator import Operator
from gtamp_utils import utils


class TwoArmPaPFeasibilityChecker(TwoArmPickFeasibilityChecker, TwoArmPlaceFeasibilityChecker):
    def __init__(self, problem_env):
        TwoArmPickFeasibilityChecker.__init__(self, problem_env)
        TwoArmPlaceFeasibilityChecker.__init__(self, problem_env)
        self.feasible_pick = [] # todo this needs to be a set rather than a list

    def check_place_feasible(self, pick_parameters, place_parameters, operator_skeleton):
        pick_op = Operator('two_arm_pick', operator_skeleton.discrete_parameters)
        pick_op.continuous_parameters = pick_parameters

        # todo remove the CustomStateSaver
        #saver = utils.CustomStateSaver(self.problem_env.env)
        original_config = utils.get_body_xytheta(self.problem_env.robot).squeeze()
        pick_op.execute()
        place_op = Operator('two_arm_place', operator_skeleton.discrete_parameters)
        place_cont_params, place_status = TwoArmPlaceFeasibilityChecker.check_feasibility(self,
                                                                                          place_op,
                                                                                          place_parameters)
        utils.two_arm_place_object(pick_op.continuous_parameters)
        utils.set_robot_config(original_config)

        #saver.Restore()
        return place_cont_params, place_status

    def check_pick_feasible(self, pick_parameters, operator_skeleton):
        pick_op = Operator('two_arm_pick', operator_skeleton.discrete_parameters)
        params, status = TwoArmPickFeasibilityChecker.check_feasibility(self, pick_op, pick_parameters)
        return params, status

    def check_feasibility(self, operator_skeleton, parameters, swept_volume_to_avoid=None):
        pick_parameters = parameters[:6]
        place_parameters = parameters[-3:]
        pap_continuous_parameters = {'action_parameters': parameters, 'is_feasible': False}

        we_already_have_feasible_pick = len(self.feasible_pick) > 0
        if we_already_have_feasible_pick:
            pick_parameters = self.feasible_pick[0]
        else:
            pick_parameters, pick_status = self.check_pick_feasible(pick_parameters, operator_skeleton)

            if pick_status != 'HasSolution':
                return pap_continuous_parameters, pick_status
            else:
                self.feasible_pick.append(pick_parameters)

        place_parameters, place_status = self.check_place_feasible(pick_parameters, place_parameters, operator_skeleton)

        if place_status != 'HasSolution':
            return pap_continuous_parameters, "NoSolution"
        else:
            pap_continuous_parameters['pick'] = pick_parameters
            pap_continuous_parameters['place'] = place_parameters
            return pap_continuous_parameters, 'HasSolution'


