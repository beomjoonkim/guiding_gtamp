from trajectory_representation.operator import Operator
from generators.uniform import UniformGenerator
from planners.subplanners.motion_planner import BaseMotionPlanner
from gtamp_utils.utils import CustomStateSaver

from gtamp_utils import utils

import numpy as np


class PlannerWithoutReachability:
    def __init__(self, problem_env, goal_object_names, goal_region):
        self.problem_env = problem_env
        self.goal_objects = [problem_env.env.GetKinBody(o) for o in goal_object_names]
        self.goal_region = self.problem_env.regions[goal_region]

    def sample_cont_params(self, operator_skeleton, n_iter):
        target_object = operator_skeleton.discrete_parameters['object']
        self.problem_env.disable_objects_in_region('entire_region')
        generator = UniformGenerator(operator_skeleton, self.problem_env, None)
        target_object.Enable(True)
        print "Generating goals for ", target_object
        param = generator.sample_next_point(operator_skeleton,
                                            n_iter=n_iter,
                                            n_parameters_to_try_motion_planning=1,
                                            dont_check_motion_existence=True)
        self.problem_env.enable_objects_in_region('entire_region')
        return param

    def get_goal_config_used(self, motion_plan, potential_goal_configs):
        which_goal = np.argmin(np.linalg.norm(motion_plan[-1] - potential_goal_configs, axis=-1))
        return potential_goal_configs[which_goal]

    def find_pick(self, curr_obj):
        if self.problem_env.name.find("one_arm") != -1:
            pick_op = Operator(operator_type='one_arm_pick', discrete_parameters={'object': curr_obj})
        else:
            pick_op = Operator(operator_type='two_arm_pick', discrete_parameters={'object': curr_obj})
        params = self.sample_cont_params(pick_op, n_iter=500)
        if not params['is_feasible']:
            return None

        pick_op.continuous_parameters = params
        return pick_op

    def find_place(self, curr_obj, pick):
        if self.problem_env.name.find("one_arm") != -1:
            place_op = Operator(operator_type='one_arm_place',
                                discrete_parameters={'object': curr_obj, 'region': self.goal_region},
                                continuous_parameters=pick.continuous_parameters)
        else:
            place_op = Operator(operator_type='two_arm_place', discrete_parameters={'object': curr_obj,
                                                                                    'region': self.goal_region})
        # it must be because sampling a feasible pick can be done by trying as many as possible,
        # but placements cannot be made feasible  by sampling more
        # also, it takes longer to check feasibility on place?
        # I just have to check the IK solution once
        params = self.sample_cont_params(place_op, n_iter=500)
        if not params['is_feasible']:
            return None

        place_op.continuous_parameters = params
        return place_op

    def search(self):
        # returns the order of objects that respects collision at placements
        # todo if I cannot find a grasp or placement in the goal region, then I should declare infeasible problem

        init_state = CustomStateSaver(self.problem_env.env)
        # self.problem_env.set_exception_objs_when_disabling_objects_in_region(self.goal_objects)
        idx = 0
        plan = []
        goal_obj_move_plan = []

        while True:
            curr_obj = self.goal_objects[idx]

            self.problem_env.disable_objects_in_region('entire_region')
            print [o.IsEnabled() for o in self.problem_env.objects]
            curr_obj.Enable(True)
            pick = self.find_pick(curr_obj)
            if pick is None:
                plan = []  # reset the whole thing?
                goal_obj_move_plan = []
                idx += 1
                idx = idx % len(self.goal_objects)
                init_state.Restore()
                print "Pick sampling failed"
                continue
            pick.execute()

            self.problem_env.enable_objects_in_region('entire_region')
            place = self.find_place(curr_obj, pick)
            if place is None:
                plan = []
                goal_obj_move_plan = []
                idx += 1
                idx = idx % len(self.goal_objects)
                init_state.Restore()
                print "Place sampling failed"
                continue
            place.execute()

            plan.append(pick)
            plan.append(place)
            goal_obj_move_plan.append(curr_obj)

            idx += 1
            idx = idx % len(self.goal_objects)
            print "Plan length: ", len(plan)
            if len(plan) / 2.0 == len(self.goal_objects):
                break

        init_state.Restore()
        return goal_obj_move_plan, plan
