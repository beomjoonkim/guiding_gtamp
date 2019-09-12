from planners.subplanners.motion_planner import BaseMotionPlanner, ArmBaseMotionPlanner
import time


class MinimumConstraintPlanner(BaseMotionPlanner, ArmBaseMotionPlanner):
    def __init__(self, problem_env, target_object, planning_algorithm):
        BaseMotionPlanner.__init__(self, problem_env, planning_algorithm)
        if type(target_object) == str:
            self.target_object = self.problem_env.env.GetKinBody(target_object)
        else:
            self.target_object = target_object

    def approximate_minimal_collision_path(self, goal_configuration, path_ignoring_all_objects,
                                           collisions_in_path_ignoring_all_objects, cached_collisions):
        enabled_objects = {obj.GetName() for obj in self.problem_env.objects}
        enabled_objects -= {obj.GetName() for obj in collisions_in_path_ignoring_all_objects}

        [o.Enable(False) for o in collisions_in_path_ignoring_all_objects]
        minimal_objects_in_way = []
        minimal_collision_path = path_ignoring_all_objects
        for obj in collisions_in_path_ignoring_all_objects:
            obj.Enable(True)
            [o.Enable(False) for o in minimal_objects_in_way]
            enabled_objects.add(obj.GetName())
            enabled_objects -= {obj.GetName() for obj in minimal_objects_in_way}
            if self.problem_env.name.find('one_arm') != -1:
                path, status = ArmBaseMotionPlanner.get_motion_plan(self, goal_configuration,
                                                                    cached_collisions=cached_collisions)
            else:
                path, status = BaseMotionPlanner.get_motion_plan(self,
                                                                 goal_configuration,
                                                                 cached_collisions=cached_collisions,
                                                                 n_iterations=[20, 50, 100])
            if status != 'HasSolution':
                minimal_objects_in_way.append(obj)
            else:
                minimal_collision_path = path
        self.problem_env.enable_objects_in_region('entire_region')
        return minimal_collision_path

    def compute_path_ignoring_obstacles(self, goal_configuration):
        self.problem_env.disable_objects_in_region('entire_region')
        if self.target_object is not None:
            self.target_object.Enable(True)
        if self.problem_env.name.find('one_arm') != -1:
            path, status = ArmBaseMotionPlanner.get_motion_plan(self, goal_configuration)
        else:
            stime = time.time()
            path, status = BaseMotionPlanner.get_motion_plan(self, goal_configuration)
            print "Motion plan time", time.time()-stime
        self.problem_env.enable_objects_in_region('entire_region')
        if path is None:
            import pdb; pdb.set_trace()
        return path

    def get_motion_plan(self, goal_configuration, region_name='entire_region', n_iterations=None,
                        cached_collisions=None):
        path_ignoring_obstacles = self.compute_path_ignoring_obstacles(goal_configuration)

        naive_path_collisions = self.problem_env.get_objs_in_collision(path_ignoring_obstacles, 'entire_region')
        assert not (self.target_object in naive_path_collisions)

        no_obstacle = len(naive_path_collisions) == 0
        if no_obstacle:
            return path_ignoring_obstacles, 'HasSolution'

        minimal_collision_path = self.approximate_minimal_collision_path(goal_configuration, path_ignoring_obstacles,
                                                                         naive_path_collisions, cached_collisions)
        self.problem_env.enable_objects_in_region('entire_region')
        return minimal_collision_path, "HasSolution"

"""
class OperatorMinimumConstraintPlanner(MinimumConstraintPlanner, OperatorBaseMotionPlanner):
    def __init__(self, problem_env, target_object, objects_moved_in_higher_level_plan, planning_algorithm,
                 parent_pick_op=None, is_last_object_to_clear=False):
        MinimumConstraintPlanner.__init__(self, problem_env, target_object, planning_algorithm)
        self.operator_instance = None
        self.robot = self.problem_env.robot
        self.objects_moved_in_higher_level_plan = objects_moved_in_higher_level_plan

        self.parent_pick_op = parent_pick_op
        self.is_last_object_to_clear = is_last_object_to_clear

    def plan_to_preplace(self, place_path):
        preplace_config = place_path[0]
        held_object = self.problem_env.robot.GetGrabbed()[0]
        held_object_original_transform = held_object.GetTransform()

        collisions = self.problem_env.get_objs_in_collision(place_path, 'entire_region')
        [o.Enable(False) for o in collisions]
        with self.problem_env.robot:
            # tentatively place to verify the existence of a path
            two_arm_place_object(self.operator_instance.continuous_parameters)
            preplace_path, status = BaseMotionPlanner.get_motion_plan(self, preplace_config)

        # go back to original pick pose
        held_object.SetTransform(held_object_original_transform)
        grab_obj(held_object)

        self.problem_env.enable_objects_in_region('entire_region')
        return preplace_path, status

    def compute_path_ignoring_obstacles(self, goal_configuration):
        self.problem_env.disable_objects_in_region('entire_region')
        [o.Enable(True) for o in self.objects_moved_in_higher_level_plan]

        self.target_object.Enable(True)
        path, status = BaseMotionPlanner.get_motion_plan(self, goal_configuration)
        self.problem_env.enable_objects_in_region('entire_region')
        return path

    def get_motion_plan(self, goal_configuration, region_name='entire_region', n_iterations=None,
                        cached_collisions=None):
        assert self.operator_instance is not None
        path_ignoring_all_objects = self.compute_path_ignoring_obstacles(goal_configuration)
        if path_ignoring_all_objects is None:
            print "No path ignoring all obstacles"
            return None, "NoSolution"

        collisions_in_path_ignoring_all_objects = \
            self.problem_env.get_objs_in_collision(path_ignoring_all_objects, 'entire_region')
        assert not (self.target_object in collisions_in_path_ignoring_all_objects)

        no_obstacle = len(path_ignoring_all_objects) == 0
        if no_obstacle:
            return path_ignoring_all_objects, 'HasSolution'

        minimal_collision_path = self.approximate_minimal_collision_path(goal_configuration,
                                                                         path_ignoring_all_objects,
                                                                         collisions_in_path_ignoring_all_objects)
        self.problem_env.enable_objects_in_region('entire_region')

        if self.operator_instance.type == 'two_arm_place' and self.is_last_object_to_clear:
            parent_pick = self.parent_pick_op.low_level_motion[0]

            held_object = self.problem_env.robot.GetGrabbed()[0]
            state_saver = CustomStateSaver(self.problem_env.env)
            two_arm_place_object(self.operator_instance.continuous_parameters)
            path_to_parent_object, status = BaseMotionPlanner.get_motion_plan(self, parent_pick)

            state_saver.Restore()
            grab_obj(held_object)

            if status == "HasSolution":
                pass
            else:
                minimal_collision_path = None
        else:
            status = "HasSolution"

        return minimal_collision_path, status
"""
