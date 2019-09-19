from mover_library.samplers import *
from mover_library.utils import set_robot_config, grab_obj, release_obj


class PlaceFeasibilityChecker:
    def __init__(self, problem_env):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]
        self.robot_region = self.problem_env.regions['entire_region']
        self.objects_to_check_collision = []

    def check_feasibility(self, operator_skeleton, place_parameters, swept_volume_to_avoid=None):
        # Note:
        #    this function checks if the target region contains the robot when we place object at place_parameters
        #    and whether the robot will be in collision
        obj = self.robot.GetGrabbed()[0]
        obj_region = operator_skeleton.discrete_parameters['region']
        if type(obj_region) == str:
            obj_region = self.problem_env.regions[obj_region]
        obj_pose = place_parameters

        T_r_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), self.robot.GetTransform())
        original_trans = self.robot.GetTransform()
        original_obj_trans = obj.GetTransform()

        target_robot_region1 = self.problem_env.regions['home_region']
        target_robot_region2 = self.problem_env.regions['loading_region']
        target_obj_region = obj_region  # for fetching, you want to move it around

        robot_xytheta = self.compute_robot_base_pose_given_object_pose(obj, self.robot, obj_pose, T_r_wrt_o)
        set_robot_config(robot_xytheta, self.robot)

        # why do I disable objects in region? Most likely this is for minimum constraint computation
        #self.problem_env.disable_objects_in_region('entire_region')
        target_region_contains = target_robot_region1.contains(self.robot.ComputeAABB()) or \
                                 target_robot_region2.contains(self.robot.ComputeAABB())
        is_base_pose_infeasible = self.env.CheckCollision(self.robot) or \
                                  (not target_region_contains)
        is_object_pose_infeasible = self.env.CheckCollision(obj) or \
                                    (not target_obj_region.contains(obj.ComputeAABB()))
        #self.problem_env.enable_objects_in_region('entire_region')

        if is_base_pose_infeasible or is_object_pose_infeasible:
            action = {'operator_name': 'two_arm_place', 'base_pose': None, 'object_pose': None,
                      'action_parameters': place_parameters}
            self.robot.SetTransform(original_trans)
            obj.SetTransform(original_obj_trans)
            return action, 'NoSolution'
        else:
            release_obj(self.robot, obj)
            if swept_volume_to_avoid is not None:
                # release the object
                if not swept_volume_to_avoid.is_swept_volume_cleared(obj):
                    self.robot.SetTransform(original_trans)
                    obj.SetTransform(original_obj_trans)
                    grab_obj(self.robot, obj)
                    action = {'operator_name': 'two_arm_place', 'base_pose': None, 'object_pose': None,
                              'action_parameters': place_parameters}
                    return action, 'NoSolution'

                """
                for config in swept_volume_to_avoid:
                    set_robot_config(config, self.robot)
                    if self.env.CheckCollision(self.robot, obj):
                        self.robot.SetTransform(original_trans)
                        obj.SetTransform(original_obj_trans)
                        grab_obj(self.robot, obj)
                        action = {'operator_name': 'two_arm_place', 'base_pose': None, 'object_pose': None,
                                  'action_parameters': place_parameters}
                        return action, 'NoSolution'
                """

            self.robot.SetTransform(original_trans)
            obj.SetTransform(original_obj_trans)
            grab_obj(self.robot, obj)
            action = {'operator_name': 'two_arm_place', 'base_pose': robot_xytheta, 'object_pose': obj_pose,
                      'action_parameters': place_parameters}
            return action, 'HasSolution'

    @staticmethod
    def compute_robot_base_pose_given_object_pose(obj, robot, obj_pose, T_r_wrt_o):
        original_robot_T = robot.GetTransform()
        release_obj(robot, obj)
        set_obj_xytheta(obj_pose, obj)
        new_T_robot = np.dot(obj.GetTransform(), T_r_wrt_o)
        robot.SetTransform(new_T_robot)
        robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        robot_xytheta = robot.GetActiveDOFValues()
        grab_obj(robot, obj)
        robot.SetTransform(original_robot_T)
        return robot_xytheta

    def get_placement(self, obj, target_obj_region, T_r_wrt_o):
        original_trans = self.robot.GetTransform()
        original_config = self.robot.GetDOFValues()
        self.robot.SetTransform(original_trans)
        self.robot.SetDOFValues(original_config)

        release_obj(self.robot, obj)
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
            grab_obj(self.robot, obj)
        return obj_pose, robot_xytheta

