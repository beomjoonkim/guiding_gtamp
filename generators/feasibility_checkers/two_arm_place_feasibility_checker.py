from mover_library.samplers import *
from mover_library.utils import set_robot_config, grab_obj, release_obj, get_body_xytheta


class TwoArmPlaceFeasibilityChecker:
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]
        self.robot_region = self.problem_env.regions['entire_region']
        self.objects_to_check_collision = []

    def check_feasibility(self, operator_skeleton, place_parameters, swept_volume_to_avoid=None,
                          parameter_mode='obj_pose'):
        # Note:
        #    this function checks if the target region contains the robot when we place object at place_parameters
        #    and whether the robot will be in collision
        obj_region = operator_skeleton.discrete_parameters['region']
        if parameter_mode == 'obj_pose':
            return self.check_place_at_obj_pose_feasible(obj_region, place_parameters, swept_volume_to_avoid)
        elif parameter_mode == 'robot_base_pose':
            return self.check_place_at_base_pose_feasible(obj_region, place_parameters, swept_volume_to_avoid)
        else:
            raise NotImplementedError

    def is_collision_and_region_constraints_satisfied(self, target_robot_region1, target_robot_region2,
                                                      target_obj_region):
        target_region_contains = target_robot_region1.contains(self.robot.ComputeAABB()) or \
                                 target_robot_region2.contains(self.robot.ComputeAABB())
        if not target_region_contains:
            return False

        obj = self.robot.GetGrabbed()[0]
        if not target_obj_region.contains(obj.ComputeAABB()):
            return False

        is_base_pose_infeasible = self.env.CheckCollision(self.robot)
        if is_base_pose_infeasible:
            return False

        is_object_pose_infeasible = self.env.CheckCollision(obj)
        if is_object_pose_infeasible:
            return False

        return True

    def check_place_at_base_pose_feasible(self, obj_region, place_base_pose, swept_volume_to_avoid):
        if type(obj_region) == str:
            obj_region = self.problem_env.regions[obj_region]
        target_obj_region = obj_region  # for fetching, you want to move it around
        target_robot_region1 = self.problem_env.regions['home_region']
        target_robot_region2 = self.problem_env.regions['loading_region']
        set_robot_config(place_base_pose, self.robot)
        is_feasible = self.is_collision_and_region_constraints_satisfied(target_robot_region1, target_robot_region2,
                                                                         target_obj_region)
        obj = self.robot.GetGrabbed()[0]
        original_trans = self.robot.GetTransform()
        original_obj_trans = obj.GetTransform()
        if not is_feasible:
            action = {'operator_name': 'two_arm_place', 'q_goal': None, 'object_pose': None,
                      'action_parameters': place_base_pose}
            self.robot.SetTransform(original_trans)
            obj.SetTransform(original_obj_trans)
            return action, 'NoSolution'
        else:
            release_obj()
            obj_pose = get_body_xytheta(obj)
            if swept_volume_to_avoid is not None:
                # release the object
                if not swept_volume_to_avoid.is_swept_volume_cleared(obj):
                    self.robot.SetTransform(original_trans)
                    obj.SetTransform(original_obj_trans)
                    grab_obj(obj)
                    action = {'operator_name': 'two_arm_place', 'q_goal': None, 'object_pose': None,
                              'action_parameters': place_base_pose}
                    return action, 'NoSolution'
            self.robot.SetTransform(original_trans)
            obj.SetTransform(original_obj_trans)
            grab_obj(obj)
            action = {'operator_name': 'two_arm_place', 'q_goal': place_base_pose, 'object_pose': obj_pose,
                      'action_parameters': place_base_pose}
            return action, 'HasSolution'

    def check_place_at_obj_pose_feasible(self, obj_region, obj_pose, swept_volume_to_avoid):
        obj = self.robot.GetGrabbed()[0]
        if type(obj_region) == str:
            obj_region = self.problem_env.regions[obj_region]

        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())
        original_trans = self.robot.GetTransform()
        original_obj_trans = obj.GetTransform()

        target_robot_region1 = self.problem_env.regions['home_region']
        target_robot_region2 = self.problem_env.regions['loading_region']
        target_obj_region = obj_region  # for fetching, you want to move it around

        robot_xytheta = self.compute_robot_base_pose_given_object_pose(obj, self.robot, obj_pose, T_r_wrt_o)
        set_robot_config(robot_xytheta, self.robot)

        is_feasible = self.is_collision_and_region_constraints_satisfied(target_robot_region1, target_robot_region2,
                                                                         target_obj_region)
        if not is_feasible:
            action = {'operator_name': 'two_arm_place', 'q_goal': None, 'object_pose': None,
                      'action_parameters': obj_pose}
            self.robot.SetTransform(original_trans)
            obj.SetTransform(original_obj_trans)
            return action, 'NoSolution'
        else:
            release_obj()
            if swept_volume_to_avoid is not None:
                # release the object
                if not swept_volume_to_avoid.is_swept_volume_cleared(obj):
                    self.robot.SetTransform(original_trans)
                    obj.SetTransform(original_obj_trans)
                    grab_obj(obj)
                    action = {'operator_name': 'two_arm_place', 'q_goal': None, 'object_pose': None,
                              'action_parameters': obj_pose}
                    return action, 'NoSolution'
            self.robot.SetTransform(original_trans)
            obj.SetTransform(original_obj_trans)
            grab_obj(obj)
            action = {'operator_name': 'two_arm_place', 'q_goal': robot_xytheta, 'object_pose': obj_pose,
                      'action_parameters': obj_pose}
            return action, 'HasSolution'

    @staticmethod
    def compute_robot_base_pose_given_object_pose(obj, robot, obj_pose, T_r_wrt_o):
        original_robot_T = robot.GetTransform()
        release_obj()
        set_obj_xytheta(obj_pose, obj)
        new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
        robot.SetTransform(new_T_robot)
        robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        robot_xytheta = robot.GetActiveDOFValues()
        grab_obj(obj)
        robot.SetTransform(original_robot_T)
        return robot_xytheta

    def get_placement(self, obj, target_obj_region, T_r_wrt_o):
        original_trans = self.robot.GetTransform()
        original_config = self.robot.GetDOFValues()
        self.robot.SetTransform(original_trans)
        self.robot.SetDOFValues(original_config)

        release_obj()
        with self.robot:
            # print target_obj_region
            obj_pose = randomly_place_in_region(self.env, obj, target_obj_region)  # randomly place obj
            obj_pose = obj_pose.squeeze()

            # compute the resulting robot transform
            new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
            self.robot.SetTransform(new_T_robot)
            self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
            robot_xytheta = self.robot.GetActiveDOFValues()
            set_robot_config(robot_xytheta, self.robot)
            grab_obj(obj)
        return obj_pose, robot_xytheta
