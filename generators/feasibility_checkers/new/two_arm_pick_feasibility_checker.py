from gtamp_utils.utils import set_robot_config, \
    two_arm_pick_object, two_arm_place_object
from gtamp_utils.operator_utils.grasp_utils import solveTwoArmIKs, compute_two_arm_grasp
from gtamp_utils import utils
from generators.feasibility_checkers.pick_feasibility_checker import PickFeasibilityChecker


class TwoArmPickFeasibilityChecker(PickFeasibilityChecker):
    def __init__(self, problem_env):
        PickFeasibilityChecker.__init__(self, problem_env)

    def compute_grasp_config(self, obj, pick_base_pose, grasp_params):
        set_robot_config(pick_base_pose, self.robot)

        were_objects_enabled = [o.IsEnabled() for o in self.problem_env.objects]
        self.problem_env.disable_objects_in_region('entire_region')
        obj.Enable(True)
        if self.env.CheckCollision(self.robot):
            for enabled, o in zip(were_objects_enabled, self.problem_env.objects):
                if enabled:
                    o.Enable(True)
                else:
                    o.Enable(False)
            return None

        grasps = compute_two_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj,
                                       robot=self.robot)

        g_config = solveTwoArmIKs(self.env, self.robot, obj, grasps)
        for enabled, o in zip(were_objects_enabled, self.problem_env.objects):
            if enabled:
                o.Enable(True)
            else:
                o.Enable(False)
        return g_config

    def is_grasp_config_feasible(self, obj, pick_base_pose, grasp_params, grasp_config):
        pick_action = {'operator_name': 'two_arm_pick', 'q_goal': pick_base_pose,
                       'grasp_params': grasp_params, 'g_config': grasp_config}
        two_arm_pick_object(obj, pick_action)
        no_collision = not self.env.CheckCollision(self.robot)

        inside_region = self.problem_env.regions['home_region'].contains(self.robot.ComputeAABB()) or \
                        self.problem_env.regions['loading_region'].contains(self.robot.ComputeAABB())

        # Why did I have the below? I cannot plan to some of the spots in entire region
        # inside_region = self.problem_env.regions['entire_region'].contains(self.robot.ComputeAABB())
        two_arm_place_object(pick_action)

        return no_collision and inside_region
