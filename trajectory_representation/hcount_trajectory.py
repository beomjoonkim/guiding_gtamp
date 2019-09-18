from trajectory import Trajectory
from trajectory_representation.operator import Operator
from planners.subplanners.motion_planner import BaseMotionPlanner

import copy
import openravepy


class HCountExpTrajectory(Trajectory):
    def __init__(self, problem_idx, filename, statetype):
        Trajectory.__init__(self, problem_idx, filename, statetype)
        self.paps_used = None

    def get_pap_used_in_plan(self, plan):
        obj_to_pick = {op.discrete_parameters['object']: [] for op in plan}
        obj_to_place = {(op.discrete_parameters['object'], op.discrete_parameters['region']): [] for op in plan}
        for op in plan:
            # making obj_to_pick and obj_to_place used in the plan; this can be done once, not every time
            pick_cont_param = op.continuous_parameters['pick']
            pick_disc_param = {'object': op.discrete_parameters['object']}
            pick_op = Operator('two_arm_pick', pick_disc_param, pick_cont_param)
            pick_op.low_level_motion = pick_cont_param['motion']

            obj_to_pick[op.discrete_parameters['object']].append(pick_op)

            place_cont_param = op.continuous_parameters['place']
            place_disc_param = {'object': op.discrete_parameters['object'],
                                'region': op.discrete_parameters['region']}
            place_op = Operator('two_arm_pick', place_disc_param, place_cont_param)
            place_op.low_level_motion = place_cont_param['motion']
            obj_to_place[(op.discrete_parameters['object'], op.discrete_parameters['region'])].append(place_op)
        return [obj_to_pick, obj_to_place]

    def account_for_used_picks_and_places(self, n_times_objs_moved, n_times_obj_region_moved):
        # accounting for the moved objects
        obj_to_pick = copy.deepcopy(self.paps_used[0])
        obj_to_place = copy.deepcopy(self.paps_used[1])

        for moved in n_times_objs_moved:
            n_times_moved = n_times_objs_moved[moved]
            if moved in obj_to_pick:
                if n_times_moved >= len(obj_to_pick[moved]):
                    del obj_to_pick[moved]
                else:
                    obj_to_pick[moved] = obj_to_pick[moved][n_times_moved]

        for moved_obj_region in n_times_obj_region_moved:
            n_times_moved = n_times_obj_region_moved[moved_obj_region]
            if moved_obj_region in obj_to_place:
                if n_times_moved >= len(obj_to_place[moved_obj_region]):
                    del obj_to_place[moved_obj_region]
                else:
                    obj_to_place[moved_obj_region] = obj_to_place[moved_obj_region][n_times_moved]

        return [obj_to_pick, obj_to_place]

    def add_trajectory(self, plan, goal_entities):
        print "Problem idx", self.problem_idx
        self.set_seed(self.problem_idx)
        problem_env, openrave_env = self.create_environment()
        motion_planner = BaseMotionPlanner(problem_env, 'prm')
        problem_env.set_motion_planner(motion_planner)

        idx = 0
        parent_state = None
        parent_action = None

        moved_objs = {p.discrete_parameters['object'] for p in plan}
        moved_obj_regions = {(p.discrete_parameters['object'], p.discrete_parameters['region']) for p in plan}
        n_times_objs_moved = {obj_name: 0 for obj_name in moved_objs}
        n_times_obj_region_moved = {(obj_name, region_name): 0 for obj_name, region_name in moved_obj_regions}

        self.paps_used = self.get_pap_used_in_plan(plan)
        curr_paps_used = self.account_for_used_picks_and_places(n_times_objs_moved, n_times_obj_region_moved)
        state = self.compute_state(parent_state, parent_action, goal_entities, problem_env, curr_paps_used, 0)

        for action_idx, action in enumerate(plan):
            action.execute()

            # mark that a pick or place in the plan has been used
            target_obj_name = action.discrete_parameters['object']
            target_region_name = action.discrete_parameters['region']
            n_times_objs_moved[target_obj_name] += 1
            n_times_obj_region_moved[(target_obj_name, target_region_name)] += 1

            curr_paps_used = self.account_for_used_picks_and_places(n_times_objs_moved, n_times_obj_region_moved)
            parent_state = state
            parent_action = action
            state = self.compute_state(parent_state, parent_action, goal_entities, problem_env, curr_paps_used,
                                       action_idx)

            # execute the pap action
            if action == plan[-1]:
                reward = 0
            else:
                reward = -1

            print "The reward is ", reward

            self.add_sar_tuples(parent_state, action, reward)
            print "Executed", action.discrete_parameters

        self.add_state_prime()
        openrave_env.Destroy()
        openravepy.RaveDestroy()
