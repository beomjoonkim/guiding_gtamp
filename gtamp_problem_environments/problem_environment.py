import copy
import numpy as np
import openravepy
import random

from openravepy import DOFAffine, Environment
from gtamp_utils.utils import grab_obj, release_obj, set_robot_config, check_collision_except, set_active_config
from gtamp_utils import utils


class ProblemEnvironment:
    def __init__(self, problem_idx):
        np.random.seed(problem_idx)
        random.seed(problem_idx)
        self.env = Environment()
        collisionChecker = openravepy.RaveCreateCollisionChecker(self.env, 'fcl_')
        self.env.SetCollisionChecker(collisionChecker)
        self.problem_idx = problem_idx

        self.initial_placements = []
        self.placements = []
        self.robot = None
        self.objects = None
        self.curr_state = None
        self.curr_obj = None
        self.init_saver = None
        self.init_which_opreator = None
        self.v = False
        self.robot_region = None
        self.obj_region = None
        self.objs_to_move = None
        self.problem_config = None
        self.init_objs_to_move = None
        self.optimal_score = None
        self.name = None

        self.is_solving_packing = False
        self.is_solving_namo = False
        self.is_solving_fetching = False

        self.high_level_planner = None
        self.namo_planner = None
        self.fetch_planner = None
        self.env.StopSimulation()  # openrave crashes with physics engine on
        self.motion_planner = None

    def set_body_poses(self, poses):
        for body_name, body_pose in zip(poses.keys(), poses.values()):
            utils.set_obj_xytheta(body_pose, self.env.GetKinBody(body_name))

    def set_motion_planner(self, motion_planner):
        self.motion_planner = motion_planner

    def make_config_from_op_instance(self, op_instance):
        if op_instance['operator'].find('one_arm') != -1:
            g_config = op_instance['action']['g_config']
            base_pose = op_instance['action']['base_pose']
            config = np.hstack([g_config, base_pose.squeeze()])
        else:
            config = op_instance['action']['base_pose']

        return config.squeeze()

    def reset_to_init_state_stripstream(self):
        # todo check if this works
        self.init_saver.Restore()
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def enable_movable_objects(self):
        for obj in self.objects:
            obj.Enable(True)

    def disable_movable_objects(self):
        if len(self.robot.GetGrabbed()) > 0:
            held_obj = self.robot.GetGrabbed()[0]
        else:
            held_obj = None

        for obj in self.objects:
            if obj == held_obj:
                continue
            obj.Enable(False)

    def get_curr_object(self):
        return self.curr_obj

    def get_placements(self):
        return copy.deepcopy(self.placements)

    def get_state(self):
        return 1

    def is_pick_time(self):
        return len(self.robot.GetGrabbed()) == 0

    def check_action_feasible(self, action, do_check_reachability=True, region_name=None):
        action = action.reshape((1, action.shape[-1]))
        place_robot_pose = action[0, 0:3]

        if not self.is_collision_at_base_pose(place_robot_pose):
            if do_check_reachability:
                # define the region to stay in?
                path, status = self.check_reachability(place_robot_pose, region_name)
                if status == "HasSolution":
                    return path, True
                else:
                    return None, False
            else:
                return None, True
        else:
            return None, False

    def is_collision_at_base_pose(self, base_pose, obj=None):
        robot = self.robot
        env = self.env
        if obj is None:
            obj_holding = self.curr_obj
        else:
            obj_holding = obj
        with robot:
            set_robot_config(base_pose, robot)
            in_collision = check_collision_except(obj_holding, env)
        if in_collision:
            return True
        return False

    def is_in_region_at_base_pose(self, base_pose, obj, robot_region, obj_region):
        robot = self.robot
        if obj is None:
            obj_holding = self.curr_obj
        else:
            obj_holding = obj

        with robot:
            set_robot_config(base_pose, robot)
            in_region = (robot_region.contains(robot.ComputeAABB())) and \
                        (obj_region.contains(obj_holding.ComputeAABB()))
        return in_region

    def apply_operator_instance(self, plan, check_feasibility=True):
        raise NotImplementedError

    def get_region_containing(self, obj):
        containing_regions = []
        for r in self.regions.values():
            if r.name == 'entire_region':
                continue
            if r.contains(obj.ComputeAABB()):
                containing_regions.append(r)

        if len(containing_regions) == 0:
            return None
        elif len(containing_regions) == 1:
            return containing_regions[0]
        else:
            region_with_smallest_area = containing_regions[0]
            for r in containing_regions:
                if r.area() < region_with_smallest_area.area():
                    region_with_smallest_area = r
            return region_with_smallest_area

    def apply_pick_constraint(self, obj_name, pick_config, pick_base_pose=None):
        # todo I think this function can be removed?
        obj = self.env.GetKinBody(obj_name)
        if pick_base_pose is not None:
            set_robot_config(pick_base_pose, self.robot)
        self.robot.SetDOFValues(pick_config)
        grab_obj(self.robot, obj)

    def is_region_contains_all_objects(self, region, objects):
        return np.all([region.contains(obj.ComputeAABB()) for obj in objects])

    def get_objs_in_collision(self, path, region_name):
        if path is None:
            return []
        if len(path) == 0:
            return []
        if not isinstance(path, list):
            path = [path]
        if len(path[0]) == 3:
            self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
        elif len(path[0]) == 11:
            manip = self.robot.GetManipulator('rightarm_torso')
            self.robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis,
                                     [0, 0, 1])
        assert len(path[0]) == self.robot.GetActiveDOF(), 'Robot active dof should match the path'
        objs = self.get_objs_in_region(region_name)
        in_collision = []
        with self.robot:
            for conf in path:
                set_active_config(conf, self.robot)
                if self.env.CheckCollision(self.robot):
                    for obj in objs:
                        if self.env.CheckCollision(self.robot, obj) and obj not in in_collision:
                            in_collision.append(obj)
        return in_collision

    def disable_objects_in_region(self, region_name):
        raise NotImplementedError

    def enable_objects_in_region(self, region_name):
        raise NotImplementedError

    def get_applicable_ops(self):
        raise NotImplementedError

    def apply_pick_action(self, action, obj=None):
        raise NotImplementedError

    def update_next_obj_to_pick(self, place_action):
        raise NotImplementedError

    def apply_place_action(self, action, do_check_reachability=True):
        raise NotImplementedError

    def remove_all_obstacles(self):
        raise NotImplementedError

    def is_goal_reached(self):
        raise NotImplementedError

    def set_init_state(self, saver):
        raise NotImplementedError

    def replay_plan(self, plan):
        raise NotImplementedError

    def which_operator(self, obj):
        raise NotImplementedError

    def restore(self, state_saver):
        raise NotImplementedError


class TwoArmProblemEnvironment(ProblemEnvironment):
    def __init__(self):
        ProblemEnvironment.__init__(self)

    def restore(self, state_saver):
        curr_obj = state_saver.curr_obj
        which_operator = state_saver.which_operator
        if not which_operator == 'two_arm_pick':
            grab_obj(self.robot, curr_obj)
        else:
            if len(self.robot.GetGrabbed()) > 0:
                release_obj(self.robot, self.robot.GetGrabbed()[0])
        state_saver.Restore()
        self.robot.SetActiveDOFs([], DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    def which_operator(self, obj=None):
        if self.is_pick_time():
            return 'two_arm_pick'
        else:
            return 'two_arm_place'
