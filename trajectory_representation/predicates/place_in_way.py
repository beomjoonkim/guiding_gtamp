from predicate import TernaryPredicate
from trajectory_representation.predicates.in_way import InWay
from gtamp_utils.utils import two_arm_pick_object
from planners.subplanners.motion_planner import BaseMotionPlanner
from planners.subplanners.minimum_constraint_planner import MinimumConstraintPlanner
from trajectory_representation.operator import Operator
from generators.uniform import UniformPaPGenerator, UniformGenerator
import pickle

from gtamp_utils.utils import *


class PlaceInWay(TernaryPredicate, InWay):
    def __init__(self, problem_env, collides=None, pick_poses=None, use_shortest_path=True):
        TernaryPredicate.__init__(self, problem_env)
        InWay.__init__(self, problem_env, collides)
        self.mc_paths = {}
        self.reachable_obj_region_pairs = []
        self.pick_poses = pick_poses

        self.use_shortest_path = use_shortest_path

    def generate_potential_pick_configs(self, operator_skeleton, n_pick_configs):
        target_object = operator_skeleton.discrete_parameters['object']
        we_already_have_pick_config = target_object.GetName() in self.sampled_pick_configs_for_objects.keys()
        if we_already_have_pick_config:
            return self.sampled_pick_configs_for_objects[target_object.GetName()]

        self.problem_env.disable_objects_in_region('entire_region')
        target_object.Enable(True)
        generator = UniformPaPGenerator(None,
                                        operator_skeleton,
                                        self.problem_env,
                                        None,
                                        n_candidate_params_to_smpl=1,
                                        total_number_of_feasibility_checks=50,
                                        dont_check_motion_existence=True)

        print "Generating goals for ", target_object
        op_cont_params = []
        for _ in range(n_pick_configs):
            param = generator.sample_next_point(cached_collisions=self.collides)
            op_cont_params.append(param)
        print "Done"
        self.problem_env.enable_objects_in_region('entire_region')
        motion_plan_goals = [op['q_goal'] for op in op_cont_params if op['is_feasible'] and op['q_goal'] is not None]
        is_op_skel_infeasible = len(motion_plan_goals) == 0
        if is_op_skel_infeasible:
            return None
        else:
            return motion_plan_goals

    def pick_object(self, obj_name):
        assert type(obj_name) == str or type(obj_name) == unicode
        obj = self.problem_env.env.GetKinBody(obj_name)

        # this assumes we have pick
        if obj_name in self.pick_used: # where does self.pick_used get updated?
            # we cannot use the pick path used in data because q_init is different
            motion_plan_goals = self.pick_used[obj_name].continuous_parameters['q_goal']
            # todo check if pick_used is still valid
            #   during planning, we would never have pick_used. So we just have to make sure for the data processing
            #   it is valid if it didn't move
            operator = self.pick_used[obj_name]
        else:
            operator = Operator('two_arm_pick', {'object': obj})
            motion_plan_goals = self.generate_potential_pick_configs(operator, n_pick_configs=10)

        if motion_plan_goals is None:
            self.sampled_pick_configs_for_objects[obj_name] = None
            return None
        else:
            self.sampled_pick_configs_for_objects[obj_name] = motion_plan_goals

        motion_planner = MinimumConstraintPlanner(self.problem_env, obj, 'prm')
        motion, status = motion_planner.get_motion_plan(motion_plan_goals, cached_collisions=self.collides)
        # why does it ever enter here?
        try:
            assert motion is not None
        except:
            import pdb;pdb.set_trace()

        if obj.GetName() not in self.pick_used:
            # import pdb;pdb.set_trace()
            operator.continuous_parameters = {'q_goal': motion_plan_goals}
            self.pick_used[obj.GetName()] = operator

        operator.execute()

    def get_minimum_constraint_path_to(self, goal_config, target_obj):
        print "Planning to goal config:", goal_config
        if self.use_shortest_path:
            motion_planner = BaseMotionPlanner(self.problem_env, 'prm')
        else:
            motion_planner = MinimumConstraintPlanner(self.problem_env, target_obj, 'prm')
        motion, status = motion_planner.get_motion_plan(goal_config, cached_collisions=self.collides)

        if motion is None:
            return None, 'NoSolution'
        return motion, status

    def generate_potential_place_configs(self, operator_skeleton, n_pick_configs):
        target_object = operator_skeleton.discrete_parameters['object']

        """
        if target_object.GetName() in self.pick_poses:
            poses = self.pick_poses[target_object.GetName()]
            if len(poses) == 0:
                return None
            else:
                return poses
        """

        # self.problem_env.disable_objects_in_region('entire_region')
        target_object.Enable(True)
        generator = UniformGenerator(operator_skeleton, self.problem_env)  # Sample collision-free placements?
        print "Generating goals for ", target_object
        op_cont_params, _ = generator.sample_feasible_op_parameters(operator_skeleton,
                                                                    n_iter=100,
                                                                    n_parameters_to_try_motion_planning=n_pick_configs)
        print "Done"
        # self.problem_env.enable_objects_in_region('entire_region')

        potential_motion_plan_goals = [op['q_goal'] for op in op_cont_params if op['q_goal'] is not None]
        is_op_skel_infeasible = len(potential_motion_plan_goals) == 0
        if is_op_skel_infeasible:
            return None
        else:
            return potential_motion_plan_goals

    def plan_minimum_constraint_path_to_region(self, region):
        obj_holding = self.robot.GetGrabbed()[0]
        target_region = self.problem_env.regions[region]
        place_op = Operator(operator_type='two_arm_place', discrete_parameters={'object': obj_holding,
                                                                                'region': target_region})
        obj_region_key = (obj_holding.GetName(), region)
        if obj_region_key in self.mc_paths:
            motion = self.mc_paths[(obj_holding.GetName(), region)]
            place_op.low_level_motion = motion
            place_op.continuous_parameters = {'q_goal': motion[-1]}
            return motion, 'HasSolution', place_op

        if obj_region_key in self.place_used:
            print "Using the place data" # todo but this depends on which state...
            motion = self.place_used[obj_region_key].low_level_motion
            status = 'HasSolution'
        else:
            potential_motion_plan_goals = self.generate_potential_place_configs(place_op, n_pick_configs=10)

            if potential_motion_plan_goals is None:
                return None, "NoSolution", None
            self.mc_calls += 1
            motion, status = self.get_minimum_constraint_path_to(potential_motion_plan_goals, obj_holding)
            if motion is None:
                return None, "NoSolution", None

        place_op.low_level_motion = motion
        place_op.continuous_parameters = {'q_goal': motion[-1]}
        if obj_region_key not in self.place_used:
            self.place_used[obj_region_key] = place_op

        return motion, status, place_op

    def __call__(self, a, b, r, cached_path=None):
        assert r in self.problem_env.regions

        # While transferring "a" to region "r", is "b" in the way to region
        if (a, b, r) in self.evaluations.keys():
            return self.evaluations[(a, b, r)]
        else:
            # what happens a is already in r? We still plan a path, because we want to know if we can move object a
            # inside region r

            is_a_obj = a not in self.problem_env.region_names
            is_b_obj = b not in self.problem_env.region_names
            is_r_region = r in self.problem_env.region_names  # this is already defended

            if not is_a_obj or not is_b_obj:
                return False

            if a == b:
                return False

            if not is_r_region:
                return False

            min_constraint_path_already_computed = (a, r) in self.mc_to_entity.keys()
            if min_constraint_path_already_computed:
                objs_in_collision = self.mc_to_entity[(a, r)]
            else:
                saver = CustomStateSaver(self.problem_env.env)
                if cached_path is not None:
                    self.pick_used[a].execute()
                    path = cached_path
                    status = 'HasSolution'
                    place_op = Operator(operator_type='two_arm_place', discrete_parameters={
                        'object': self.problem_env.env.GetKinBody(a),
                        'region': self.problem_env.regions[r],
                    })
                    place_op.low_level_motion = path
                    place_op.continuous_parameters = {'q_goal': path[-1]}
                else:
                    self.pick_object(a)
                    path, status, place_op = self.plan_minimum_constraint_path_to_region(r)

                if status != 'HasSolution':
                    objs_in_collision = None
                else:
                    objs_in_collision = self.problem_env.get_objs_in_collision(path, 'entire_region')
                    objs_in_collision = [o.GetName() for o in objs_in_collision]
                    self.mc_to_entity[(a, r)] = objs_in_collision
                    self.mc_path_to_entity[(a, r)] = path
                    if len(objs_in_collision) == 0:
                        self.reachable_obj_region_pairs.append((a, r))
                saver.Restore()

            evaluation = objs_in_collision is not None and b in objs_in_collision
            self.evaluations[(a, b, r)] = evaluation
            return evaluation
