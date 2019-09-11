from trajectory_representation.predicates.predicate import BinaryPredicate
from trajectory_representation.predicates.is_reachable import IsReachable
from generators.uniform import UniformPaPGenerator
from trajectory_representation.operator import Operator
from planners.subplanners.motion_planner import BaseMotionPlanner
#from planners.subplanners.minimum_constraint_planner import MinimumConstraintPlanner
#from planners.subplanners.one_arm_minimum_constraint_planner import OneArmMinimumConstraintPlanner
from trajectory_representation.predicates.in_way import InWay

import numpy as np
from gtamp_utils.utils import visualize_path


class PickInWay(BinaryPredicate, InWay):
    def __init__(self, problem_env, collides=None, pick_poses=None, use_shortest_path=True):
        BinaryPredicate.__init__(self, problem_env)
        InWay.__init__(self, problem_env, collides)

        self.pick_poses = pick_poses

        self.use_shortest_path = use_shortest_path

    def generate_potential_pick_configs(self, operator_skeleton, n_pick_configs):
        import pdb;pdb.set_trace()
        target_object = operator_skeleton.discrete_parameters['object']

        if target_object.GetName() in self.pick_poses:
            poses = self.pick_poses[target_object.GetName()]
            if len(poses) == 0:
                return None
            else:
                return poses
        we_already_have_pick_config = target_object.GetName() in self.sampled_pick_configs_for_objects.keys()
        if we_already_have_pick_config:
            return self.sampled_pick_configs_for_objects[target_object.GetName()]

        self.problem_env.disable_objects_in_region('entire_region')
        target_object.Enable(True)
        generator = UniformPaPGenerator(None,
                                        operator_skeleton,
                                        self.problem_env,
                                        None,
                                        n_candidate_params_to_smpl=n_pick_configs,
                                        total_number_of_feasibility_checks=100,
                                        dont_check_motion_existence=False)
        print "Generating goals for ", target_object
        op_cont_params, _ = generator.sample_params_with_feasible_motion_planning_goals(operator_skeleton,
                                                                                        n_iter=100,
                                                                                        n_parameters_to_try_motion_planning=n_pick_configs)
        print "Done"
        self.problem_env.enable_objects_in_region('entire_region')
        motion_plan_goals = [op['q_goal'] for op in op_cont_params if op['q_goal'] is not None]
        is_op_skel_infeasible = len(motion_plan_goals) == 0
        if is_op_skel_infeasible:
            return None
        else:
            return motion_plan_goals

    def __call__(self, a, b, cached_path=None):
        if (a, b) in self.evaluations.keys():
            return self.evaluations[(a, b)]
        else:
            is_a_obj = a not in self.problem_env.region_names
            if not is_a_obj or a == b:
                self.evaluations[(a, b)] = False
                return False

            is_b_region = b in self.problem_env.region_names
            if is_b_region:
                self.reachable_entities.append(b)  # assume region is always reachable, because we will never try to reach it without object in hand
                self.evaluations[(a, b)] = False
                return False

            is_robot_holding = len(self.problem_env.robot.GetGrabbed()) > 0
            assert not is_robot_holding

            is_mc_path_to_b_already_computed = b in self.mc_to_entity.keys()
            if not is_mc_path_to_b_already_computed:
                #print 'Computing mc path for ', a, b
                self.mc_calls += 1
                #assert cached_path is not None # we assume we always have cached path, with caching-based states
                path = cached_path
                if path is None:
                    self.evaluations[(a, b)] = False
                    return False
                try:
                    objs_in_collision = self.problem_env.get_objs_in_collision(path, 'entire_region')
                except:
                    import pdb;pdb.set_trace()
                objs_in_collision = [o.GetName() for o in objs_in_collision]
                self.mc_to_entity[b] = objs_in_collision
                self.mc_path_to_entity[b] = path
                if len(objs_in_collision) == 0:
                    self.reachable_entities.append(b)
            else:
                objs_in_collision = self.mc_to_entity[b]

            evaluation = a in objs_in_collision
            self.evaluations[(a, b)] = evaluation
            return evaluation
