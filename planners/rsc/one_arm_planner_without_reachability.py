from trajectory_representation.operator import Operator
from gtamp_utils.utils import CustomStateSaver
from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator
from gtamp_utils import utils

import time


class OneArmPlannerWithoutReachability:
    def __init__(self, problem_env, goal_object_names, goal_region):
        self.problem_env = problem_env
        self.goal_objects = [problem_env.env.GetKinBody(o) for o in goal_object_names]

        self.goal_region = self.problem_env.regions[goal_region]

    def sample_op_instance(self, curr_obj, n_iter):
        op = Operator(operator_type='one_arm_pick_one_arm_place',
                      discrete_parameters={'object': curr_obj, 'region': self.goal_region})
        target_object = op.discrete_parameters['object']
        generator = OneArmPaPUniformGenerator(op, self.problem_env, None)
        print "Sampling paps for ", target_object
        pick_cont_param, place_cont_param, status = generator.sample_next_point(max_ik_attempts=n_iter)
        op.continuous_parameters = {'pick': pick_cont_param, 'place': place_cont_param}
        return op, status

    def find_pick_and_place(self, curr_obj):
        stime = time.time()
        op, status = self.sample_op_instance(curr_obj, 10)
        print time.time() - stime
        if status == 'HasSolution':
            return op, status
        else:
            return None, status

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
            curr_obj.Enable(True)
            pap, status = self.find_pick_and_place(curr_obj)

            if pap is None:
                plan = []  # reset the whole thing?
                goal_obj_move_plan = []
                idx += 1
                idx = idx % len(self.goal_objects)
                init_state.Restore()
                self.problem_env.objects_to_check_collision = None
                print "Pick sampling failed"
                continue

            pap.execute()
            self.problem_env.set_exception_objs_when_disabling_objects_in_region([curr_obj])
            print "Pap executed"

            plan.append(pap)
            goal_obj_move_plan.append(curr_obj)

            idx += 1
            idx = idx % len(self.goal_objects)
            print "Plan length: ", len(plan)
            if len(plan) == len(self.goal_objects):
                break
        init_state.Restore()
        self.problem_env.enable_objects_in_region('entire_region')
        return goal_obj_move_plan, plan
