from gtamp_problem_environments.mover_env import Mover
from gtamp_utils.utils import visualize_path
from manipulation.bodies.bodies import set_color, get_color
from planners.subplanners.motion_planner import BaseMotionPlanner
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState

from gtamp_problem_environments.reward_functions.packing_problem.reward_function import ShapedRewardFunction

from trajectory_representation.minimum_constraint_pick_and_place_state import MinimiumConstraintPaPState
from gtamp_utils import utils

import openravepy
import numpy as np
import random
import pickle
import copy


class Trajectory:
    def __init__(self, problem_idx, filename, statetype):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_prime = []
        self.seed = None  # this defines the initial state
        self.problem_idx = problem_idx
        self.filename = filename
        self.statetype = statetype

    def add_sar_tuples(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def add_state_prime(self):
        self.state_prime = self.states[1:]

    def create_environment(self):
        problem_env = Mover(self.problem_idx)
        openrave_env = problem_env.env
        target_object = openrave_env.GetKinBody('square_packing_box1')
        set_color(target_object, [1, 0, 0])

        return problem_env, openrave_env

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)

    def get_pap_used_in_plan(self, plan):
        picks = plan[::2]
        places = plan[1::2]
        obj_to_pick = {p.discrete_parameters['object']: p for p in picks}
        obj_to_place = {(p.discrete_parameters['object'], p.discrete_parameters['region']): p for p in places}
        return [obj_to_pick, obj_to_place]

    def visualize_q_values(self, problem_env):
        traj = pickle.load(
            open('./test_results/mcts_results_on_mover_domain/widening_5/uct_1.0/raw_data/traj_pidx_0.pkl', 'r'))
        state = traj.states[0]
        action = traj.actions[0]
        print "Truth: ", action.discrete_parameters['object']
        q_function = self.load_qinit(problem_env.entity_idx)
        objects = problem_env.objects
        object_names = [str(p.GetName()) for p in objects]
        other_actions = [copy.deepcopy(action) for a in object_names]
        for a, obj_name in zip(other_actions, object_names):
            a.discrete_parameters['object'] = obj_name
            pred = q_function.predict(state, a)
            print pred, a.discrete_parameters
            obj = problem_env.env.GetKinBody(obj_name)
            set_color(obj, [0, 0, 1.0 / float(pred)])

    def plan_sanity_check(self, problem_env, plan):
        pass

    def verify_if_predicate_evaluations_match_with_old_file(self, idx, paps_used, state, problem_env):
        pidx = problem_env.problem_idx

        old_state = pickle.load(
            open('./test_results/hpn_results_on_mover_domain/1/trajectory_data//pap_traj_%d.pkl'
                 % (pidx), 'r')).states[idx]

        # check nodes
        for key in paps_used[1]:
            held_obj = key[0]

            region = key[1]
            for obj in problem_env.object_names:
                ternary_key = (held_obj, obj, region)
                path_used = state.place_in_way.mc_path_to_entity[key]
                path_used_in_plan = paps_used[1][key].low_level_motion
                assert np.all(np.isclose(path_used, path_used_in_plan))

                for t, z in zip(old_state.ternary_edges[ternary_key], state.ternary_edges[ternary_key]):
                    if not np.isclose(t, z):
                        assert np.all(np.isclose(state.place_in_way.mc_path_to_entity[key][-1],
                                                 paps_used[1][key].continuous_parameters['q_goal']))
                        try:
                            assert np.all(np.isclose(state.place_in_way.mc_path_to_entity[key][0],
                                                     paps_used[0][key[0]].continuous_parameters['q_goal']))
                        except:
                            import pdb;
                            pdb.set_trace()
                        # did I use the path given in the path?

    @staticmethod
    def is_same_motion(path1, path2):
        if len(path1) != len(path2):
            return False
        for c1, c2 in zip(path1, path2):
            if not np.all(c1 == c2):
                return False
        return True

    def compute_state(self, parent_state, parent_action, goal_entities, problem_env, paps_used, idx):
        # Debugging purpose
        fstate = './%s_pidx_%d_node_idx_%d_state.pkl' % (self.filename, self.problem_idx, idx)
        """
        if os.path.isfile(fstate):
            state = pickle.load(open(fstate, 'r'))
            state.problem_env = problem_env
            state.make_plannable(problem_env)
        else:
        """
        if parent_action is not None:
            parent_action.discrete_parameters['two_arm_place_object'] = parent_action.discrete_parameters['object']

        if self.statetype == 'shortest':
            state = ShortestPathPaPState(problem_env, goal_entities, parent_state, parent_action, 'irsc', paps_used)
        elif self.statetype == 'mc':
            state = MinimiumConstraintPaPState(problem_env, goal_entities, parent_state, parent_action, paps_used)
        # state.make_pklable()  # removing openrave files to pkl
        # pickle.dump(state, open(fstate, 'wb'))
        # state.make_plannable(problem_env)
        # End of debugging

        return state

    def add_trajectory(self, plan, goal_entities):
        print "Problem idx", self.problem_idx
        self.set_seed(self.problem_idx)
        problem_env, openrave_env = self.create_environment()
        motion_planner = BaseMotionPlanner(problem_env, 'prm')
        problem_env.set_motion_planner(motion_planner)

        idx = 0
        parent_state = None
        parent_action = None

        self.plan_sanity_check(problem_env, plan)
        paps_used = self.get_pap_used_in_plan(plan)
        pick_used = paps_used[0]
        place_used = paps_used[1]
        reward_function = ShapedRewardFunction(problem_env, ['square_packing_box1'], 'home_region', 3 * 8)
        # utils.viewer()
        state = self.compute_state(parent_state, parent_action, goal_entities, problem_env, paps_used, 0)
        for action_idx, action in enumerate(plan):
            if 'place' in action.type:
                continue

            target_obj = openrave_env.GetKinBody(action.discrete_parameters['object'])
            color_before = get_color(target_obj)
            set_color(target_obj, [1, 1, 1])

            pick_used, place_used = self.delete_moved_objects_from_pap_data(pick_used, place_used, target_obj)
            paps_used = [pick_used, place_used]

            action.is_skeleton = False
            pap_action = copy.deepcopy(action)
            pap_action = pap_action.merge_operators(plan[action_idx + 1])
            pap_action.is_skeleton = False
            pap_action.execute()
            # set_color(target_obj, color_before)

            parent_state = state
            parent_action = pap_action
            state = self.compute_state(parent_state, parent_action, goal_entities, problem_env, paps_used, action_idx)

            # execute the pap action
            reward = reward_function(parent_state, state, parent_action, action_idx)
            print "The reward is ", reward

            self.add_sar_tuples(parent_state, pap_action, reward)
            print "Executed", action.discrete_parameters

        self.add_state_prime()
        openrave_env.Destroy()
        openravepy.RaveDestroy()

    def delete_moved_objects_from_pap_data(self, pick_used, place_used, moved_obj):
        moved_obj_name = moved_obj.GetName()
        new_pick_used = {key: value for key, value in zip(pick_used.keys(), pick_used.values()) if
                         key != moved_obj_name}

        new_place_used = {}
        for key, value in zip(place_used.keys(), place_used.values()):
            if moved_obj_name == key[0]:
                continue
            new_place_used[key] = value
        return new_pick_used, new_place_used

    def visualize(self):
        self.set_seed(self.problem_idx)
        problem_env, openrave_env = self.create_environment()
        if openrave_env.GetViewer() is None:
            openrave_env.SetViewer('qtcoin')

        for a in self.actions:
            visualize_path(problem_env.robot, a.low_level_motion)
            a.execute()
