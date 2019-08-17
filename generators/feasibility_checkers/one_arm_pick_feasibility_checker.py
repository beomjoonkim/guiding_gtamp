from gtamp_utils.utils import set_robot_config, open_gripper
from gtamp_utils.operator_utils.grasp_utils import compute_one_arm_grasp, solveIKs
from manipulation.bodies.bodies import set_config
from generators.feasibility_checkers.pick_feasibility_checker import PickFeasibilityChecker
from gtamp_utils.utils import get_pick_base_pose_and_grasp_from_pick_parameters, set_active_config
import numpy as np
import time


class OneArmPickFeasibilityChecker(PickFeasibilityChecker):
    def __init__(self, problem_env):
        PickFeasibilityChecker.__init__(self, problem_env)

    def check_feasibility(self, operator_skeleton, pick_parameters, swept_volume_to_avoid=None):
        # This function checks if the base pose is not in collision and if there is a feasible pick
        obj = operator_skeleton.discrete_parameters['object']
        if type(obj) == str:
            obj = self.problem_env.env.GetKinBody(obj)
        grasp_params, pick_base_pose = get_pick_base_pose_and_grasp_from_pick_parameters(obj, pick_parameters)
        grasp_config = self.compute_feasible_grasp_config(obj, pick_base_pose, grasp_params)

        if grasp_config is not None:
            pick_action = {'operator_name': operator_skeleton.type, 'q_goal': np.hstack([grasp_config, pick_base_pose]),
                           'grasp_params': grasp_params, 'g_config': grasp_config, 'action_parameters': pick_parameters}
            return pick_action, 'HasSolution'
        else:
            pick_action = {'operator_name': operator_skeleton.type, 'q_goal': None, 'grasp_params': None,
                           'g_config': None, 'action_parameters': pick_parameters}
            return pick_action, "NoSolution"

    def compute_grasp_config(self, obj, pick_base_pose, grasp_params, from_place=False):
        set_robot_config(pick_base_pose, self.robot)

        if not from_place:
            # checks the base feasibility
            no_collision = not self.env.CheckCollision(self.robot)
            if not no_collision:
                return None
            # TODO: for some reason this is really slow, is it actually necessary?
            inside_region = self.problem_env.regions['home_region'].contains(self.robot.ComputeAABB()) or \
                            self.problem_env.regions['loading_region'].contains(self.robot.ComputeAABB())
            if not inside_region:
                return None

        #stime = time.time()
        open_gripper()
        grasps = compute_one_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj,
                                       robot=self.robot)
        #print 'grasp_computation', time.time()-stime
        grasp_config, grasp = solveIKs(self.env, self.robot, grasps)
        #print 'Succeed?', grasp_config is not None
        #print 'IK computation', time.time()-stime

        return grasp_config

    def is_grasp_config_feasible(self, obj, pick_base_pose, grasp_params, grasp_config):
        # checks the feasibility of the pick configuration
        rightarm_torso_manip = self.robot.GetManipulator('rightarm_torso')
        set_config(self.robot, grasp_config, rightarm_torso_manip.GetArmIndices())
        inside_region = self.problem_env.regions['home_region'].contains(self.robot.ComputeAABB()) or \
                        self.problem_env.regions['loading_region'].contains(self.robot.ComputeAABB())
        if inside_region:
            return True
        else:
            return False
