from gtamp_utils.samplers import *
from gtamp_utils.utils import set_robot_config, grab_obj, release_obj, set_config
from generators.feasibility_checkers.place_feasibility_checker import PlaceFeasibilityChecker
from generators.feasibility_checkers.one_arm_pick_feasibility_checker import OneArmPickFeasibilityChecker
from gtamp_utils.operator_utils.grasp_utils import compute_one_arm_grasp, solveIKs


class OneArmPlaceFeasibilityChecker(PlaceFeasibilityChecker, OneArmPickFeasibilityChecker):
    def __init__(self, problem_env):
        PlaceFeasibilityChecker.__init__(self, problem_env)

    def place_object_and_robot_at_new_pose(self, obj, obj_pose, obj_region):
        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())
        if len(self.robot.GetGrabbed()) > 0:
            release_obj()
        set_obj_xytheta(obj_pose, obj)
        new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
        self.robot.SetTransform(new_T_robot)
        new_base_pose = get_body_xytheta(self.robot)
        set_robot_config(new_base_pose, self.robot)
        fold_arms()
        set_point(obj, np.hstack([obj_pose[0:2], obj_region.z + 0.001]))
        return new_base_pose

    def solve_ik_from_grasp_params(self, obj, grasp_params):
        open_gripper()
        grasps = compute_one_arm_grasp(depth_portion=grasp_params[2],
                                       height_portion=grasp_params[1],
                                       theta=grasp_params[0],
                                       obj=obj,
                                       robot=self.robot)
        # todo place the tool at grasp
        grasp_config, grasp = solveIKs(self.env, self.robot, grasps)
        return grasp_config

    def check_feasibility(self, operator_skeleton, place_parameters, swept_volume_to_avoid=None):
        obj = self.robot.GetGrabbed()[0]
        obj_original_pose = obj.GetTransform()
        obj_pose = place_parameters

        robot_original_xytheta = get_body_xytheta(self.robot)
        robot_original_config = self.robot.GetDOFValues()
        grasp_params = operator_skeleton.continuous_parameters['grasp_params']

        obj_region = operator_skeleton.discrete_parameters['region']
        if type(obj_region) == str:
            obj_region = self.problem_env.regions[obj_region]
        target_robot_region = self.problem_env.regions['home_region']
        target_obj_region = obj_region

        new_base_pose = self.place_object_and_robot_at_new_pose(obj, obj_pose, obj_region)

        is_object_pose_infeasible = self.env.CheckCollision(obj) or \
                                    (not target_obj_region.contains(obj.ComputeAABB()))

        if not is_object_pose_infeasible:
            if swept_volume_to_avoid is not None:
                is_object_pose_infeasible = not swept_volume_to_avoid.is_swept_volume_cleared(obj)

        if is_object_pose_infeasible:
            action = {'operator_name': 'one_arm_place', 'q_goal': None, 'base_pose': None, 'object_pose': None,
                      'action_parameters': obj_pose, 'grasp_params': grasp_params}
            set_robot_config(robot_original_xytheta)
            self.robot.SetDOFValues(robot_original_config)
            obj.SetTransform(obj_original_pose)
            grab_obj(obj)
            return action, 'InfeasibleBase'

        is_base_pose_infeasible = self.env.CheckCollision(self.robot) or \
                                  (not target_robot_region.contains(self.robot.ComputeAABB()))
        if is_base_pose_infeasible:
            for i in range(3):
                obj_pose[-1] += 90 * np.pi / 180.0
                new_base_pose = self.place_object_and_robot_at_new_pose(obj, obj_pose, obj_region)
                # is_object_pose_infeasible = self.env.CheckCollision(obj) or \
                #                            (not target_obj_region.contains(obj.ComputeAABB()))
                is_base_pose_infeasible = self.env.CheckCollision(self.robot) or \
                                          (not target_robot_region.contains(self.robot.ComputeAABB()))
                if not (is_base_pose_infeasible or is_object_pose_infeasible):
                    break

        if is_base_pose_infeasible or is_object_pose_infeasible:
            action = {'operator_name': 'one_arm_place', 'q_goal': None, 'base_pose': None, 'object_pose': None,
                      'action_parameters': obj_pose, 'grasp_params': grasp_params}
            set_robot_config(robot_original_xytheta)
            self.robot.SetDOFValues(robot_original_config)
            obj.SetTransform(obj_original_pose)
            grab_obj(obj)
            return action, 'InfeasibleBase'

        grasp_config = self.solve_ik_from_grasp_params(obj, grasp_params)

        #self.problem_env.enable_objects_in_region('entire_region')
        #[o.Enable(True) for o in self.problem_env.boxes]

        if grasp_config is None:
            action = {'operator_name': 'one_arm_place', 'base_pose': None, 'object_pose': None,
                      'q_goal': None, 'action_parameters': obj_pose, 'g_config': grasp_config,
                      'grasp_params': grasp_params}
            set_robot_config(robot_original_xytheta)
            self.robot.SetDOFValues(robot_original_config)
            obj.SetTransform(obj_original_pose)
            grab_obj(obj)
            return action, 'InfeasibleIK'
        else:
            grasp_config = grasp_config.squeeze()
            new_base_pose = new_base_pose.squeeze()
            action = {'operator_name': 'one_arm_place', 'q_goal': np.hstack([grasp_config, new_base_pose]),
                      'base_pose': new_base_pose, 'object_pose': place_parameters,
                      'action_parameters': obj_pose, 'g_config': grasp_config, 'grasp_params': grasp_params}
            set_robot_config(robot_original_xytheta)
            self.robot.SetDOFValues(robot_original_config)
            obj.SetTransform(obj_original_pose)
            grab_obj(obj)
            return action, 'HasSolution'
