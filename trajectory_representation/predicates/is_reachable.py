from planners.subplanners.motion_planner import BaseMotionPlanner, ArmBaseMotionPlanner
from trajectory_representation.predicates.predicate import UnaryPredicate
from generators.uniform import UniformPaPGenerator
from trajectory_representation.operator import Operator
from gtamp_utils.utils import get_place_domain, set_robot_config, CustomStateSaver, get_body_xytheta, visualize_path, \
    are_base_confs_close_enough
from gtamp_utils.motion_planner import find_prm_path, init_prm

import numpy as np


class IsReachable(UnaryPredicate):
    def __init__(self, problem_env, collides=None, pick_poses=None):
        UnaryPredicate.__init__(self, problem_env)
        self.evaluations = {}
        self.motion_plans = {}
        self.pick_used = {}
        self.reachable_entities = []
        self.unreachable_entities = []
        self.collides = collides
        self.sampled_pick_configs_for_objects = {}
        self.pick_poses = pick_poses

        self.precompute()

    def set_pick_used(self, pick_used):
        self.pick_used = pick_used

    def generate_potential_pick_configs(self, operator_skeleton, n_pick_configs):
        target_object = operator_skeleton.discrete_parameters['object']

        if target_object.GetName() in self.pick_poses:
            poses = self.pick_poses[target_object.GetName()]
            if len(poses) == 0:
                return None
            else:
                return poses

        self.problem_env.enable_objects_in_region('entire_region')
        generator = UniformPaPGenerator(operator_skeleton, self.problem_env, None, 100, n_pick_configs, False)
        print "Generating goals for ", target_object
        # todo fix this here
        op_cont_params, _ = generator.sample_params_with_feasible_motion_planning_goals(operator_skeleton, n_iter=100,
                                                                                        n_parameters_to_try_motion_planning=n_pick_configs)
        print "Done"
        self.problem_env.enable_objects_in_region('entire_region')
        motion_plan_goals = [op['q_goal'] for op in op_cont_params if op['q_goal'] is not None]
        is_op_skel_infeasible = len(motion_plan_goals) == 0
        if is_op_skel_infeasible:
            return None
        else:
            return motion_plan_goals

    def precompute(self):
        if self.problem_env.name.find('one_arm') != -1:
            return

        init_prm()
        from gtamp_utils.motion_planner import prm_vertices

        entities = [
                       obj.GetName()
                       for obj in self.problem_env.objects
                   ] + self.problem_env.regions.keys()

        baseconf, = get_body_xytheta(self.problem_env.robot)

        start = {
            i for i, q in enumerate(prm_vertices)
            if are_base_confs_close_enough(baseconf, q, xy_threshold=.8, th_threshold=50)
        }
        holding = len(self.problem_env.robot.GetGrabbed()) > 0
        goal_states = [
            None if entity in self.problem_env.regions else
            [] if holding else
            self.generate_potential_pick_configs(
                Operator('two_arm_pick', {'object': self.problem_env.env.GetKinBody(entity)}), n_pick_configs=10)
            for entity in entities
        ]
        goal_fns = [
            (lambda i: self.problem_env.regions[entity].contains_point(prm_vertices[i]))
            if entity in self.problem_env.regions else
            (lambda i: any(
                are_base_confs_close_enough(q, prm_vertices[i], xy_threshold=.8, th_threshold=50)
                for q in (goal_states[j] if goal_states[j] is not None else [])
            ))
            for j, entity in enumerate(entities)
        ]
        heuristic = lambda i: 0
        all_collides = {i for s in self.collides for i in s} if self.collides is not None else None

        def in_collision(q):
            old_q = get_body_xytheta(self.problem_env.robot)
            set_robot_config(q, self.problem_env.robot)
            col = self.problem_env.env.CheckCollision(self.problem_env.robot)
            set_robot_config(old_q, self.problem_env.robot)
            return col

        is_collision = (lambda i: i in all_collides) if all_collides is not None else in_collision
        paths = find_prm_path(start, goal_fns, heuristic, is_collision)

        self.reachable_entities = [entity for i, entity in enumerate(entities) if paths[i] is not None]
        self.unreachable_entities = [entity for i, entity in enumerate(entities) if paths[i] is None]

        self.evaluations = {
            entity: path is not None
            for entity, path in zip(entities, paths)
        }
        self.motion_plans = {
            entity: path
            for entity, path in zip(entities, paths)
        }

        return

    def check_obj_reachable(self, obj):
        if len(self.problem_env.robot.GetGrabbed()) > 0:
            return False

        obj = self.problem_env.env.GetKinBody(obj)
        obj_name = str(obj.GetName())
        if self.problem_env.name.find('one_arm') != -1:
            op = Operator('one_arm_pick', {'object': obj})
        else:
            op = Operator('two_arm_pick', {'object': obj})

        if obj_name in self.pick_used:
            motion_plan_goals = self.pick_used[obj_name].continuous_parameters['q_goal']
        else:
            motion_plan_goals = self.generate_potential_pick_configs(op, n_pick_configs=10)

        if motion_plan_goals is not None:
            motion_planner = BaseMotionPlanner(self.problem_env, 'prm')
            motion, status = motion_planner.get_motion_plan(motion_plan_goals)
            is_feasible_param = status == 'HasSolution'
        else:
            is_feasible_param = False

        if is_feasible_param:
            op.make_pklable()
            op.continuous_parameters = {'q_goal': motion[-1]}
            self.motion_plans[obj_name] = motion

            if obj_name not in self.pick_used:
                self.pick_used[obj_name] = op
            self.evaluations[obj_name] = True
            return True
        else:
            self.evaluations[obj_name] = False
            return False

    def check_region_reachable(self, region):
        if self.problem_env.regions[region].contains(self.problem_env.robot.ComputeAABB()):
            self.motion_plans[region] = []
            self.evaluations[region] = True
            return True

        if self.problem_env.name.find("one_arm") != -1:
            return self.one_arm_domain_region_reachability_check(region)
        else:
            return self.two_arm_domain_region_reachability_check(region)

    def one_arm_domain_region_reachability_check(self, region):
        is_packing_box_region = region.find('box') != -1
        if is_packing_box_region:
            is_holding = len(self.problem_env.robot.GetGrabbed()) > 0
            if is_holding:
                self.motion_plans[region] = []
                self.evaluations[region] = True
                return True
            else:
                self.motion_plans[region] = []
                self.evaluations[region] = True

                return True
        else:
            self.motion_plans[region] = []
            self.evaluations[region] = True

            return True

    def two_arm_domain_region_reachability_check(self, region):
        motion_planner = BaseMotionPlanner(self.problem_env, 'prm')
        print "Motion planning to ", region
        motion, status = motion_planner.get_motion_plan(self.problem_env.regions[region],
                                                        cached_collisions=self.collides)
        if status == 'HasSolution':
            self.motion_plans[region] = motion
            self.evaluations[region] = True
            return True
        else:
            self.evaluations[region] = False
            return False

    def sample_feasible_base_pose(self, region):
        saver = CustomStateSaver(self.problem_env.env)
        domain = get_place_domain(self.problem_env.regions[region])
        domain_min = domain[0]
        domain_max = domain[1]

        in_collision = True
        base_pose = None
        for i in range(10000):
            base_pose = np.random.uniform(domain_min, domain_max, (1, 3)).squeeze()
            set_robot_config(base_pose, self.problem_env.robot)
            in_collision = self.problem_env.env.CheckCollision(self.problem_env.robot)
            if not in_collision:
                break
        saver.Restore()

        if in_collision:
            return None
        else:
            return base_pose

    def __call__(self, entity):
        if entity is None:  # not an object
            return False

        if entity in self.evaluations.keys():
            return self.evaluations[entity]
        else:
            raise NotImplementedError
            """
            is_entity_obj = entity not in self.problem_env.regions
            if is_entity_obj:
                is_reachable = self.check_obj_reachable(entity)
            else:
                is_reachable = True
            return is_reachable
            """
