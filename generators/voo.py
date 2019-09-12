import sys
import numpy as np
from gtamp_utils.samplers import gaussian_randomly_place_in_region
from generator import PaPGenerator
from gtamp_utils.utils import pick_parameter_distance, place_parameter_distance, se2_distance, visualize_path
from gtamp_utils.utils import *
from gtamp_utils import utils
import time


class PaPVOOGenerator(PaPGenerator):
    def __init__(self, node, operator_skeleton, problem_env, swept_volume_constraint,
                 n_feasibility_checks, n_candidate_params_to_smpl, dont_check_motion_existence,
                 explr_p, c1, sampling_mode, counter_ratio):
        PaPGenerator.__init__(self, node, operator_skeleton, problem_env, swept_volume_constraint,
                              n_feasibility_checks, n_candidate_params_to_smpl, dont_check_motion_existence)
        self.node = node
        self.explr_p = explr_p
        self.evaled_actions = []
        self.evaled_q_values = []
        self.c1 = c1
        self.idx_to_update = None
        self.robot = self.problem_env.robot
        self.sampling_mode = sampling_mode
        self.counter_ratio = 1.0 / counter_ratio
        self.feasible_pick_params = {}

        if node.operator_skeleton.type == 'two_arm_pick':
            obj = self.node.operator_skeleton.discrete_parameters['object']

            def dist_fcn(x, y):
                return pick_parameter_distance(obj, x, y)
        elif node.operator_skeleton.type == 'two_arm_place':
            def dist_fcn(x, y):
                return place_parameter_distance(x, y, self.c1)
        elif 'pick' in node.operator_skeleton.type \
                and 'place' in node.operator_skeleton.type:
            obj_name = self.node.operator_skeleton.discrete_parameters['object']
            obj = utils.convert_to_kin_body(obj_name)

            def dist_fcn(x, y):
                x_pick = x[:6]
                x_place = x[-3:]
                y_pick = y[:6]
                y_place = y[-3:]
                dist = pick_parameter_distance(obj, x_pick, y_pick) \
                       + place_parameter_distance(x_place, y_place, self.c1)
                return dist
        else:
            raise NotImplementedError

        self.dist_fcn = dist_fcn

    def sample_candidate_pap_parameters(self, iter_limit):
        assert iter_limit > 0
        feasible_op_parameters = []
        for i in range(iter_limit):
            op_parameters = self.sample_using_voo()
            op_parameters, status = self.op_feasibility_checker.check_feasibility(self.operator_skeleton,
                                                                                  op_parameters,
                                                                                  self.swept_volume_constraint)

            if status == 'HasSolution':
                op_parameters['is_feasible'] = False
                feasible_op_parameters.append(op_parameters)
                if len(feasible_op_parameters) >= self.n_candidate_params_to_smpl:
                    break

        if len(feasible_op_parameters) == 0:
            feasible_op_parameters.append(op_parameters)  # place holder
            status = "NoSolution"
        else:
            status = "HasSolution"

        return feasible_op_parameters, status

    def sample_next_point(self, cached_collisions=None, cached_holding_collisions=None):
        self.update_evaled_values()
        chosen_op_param = PaPGenerator.sample_next_point(self, cached_collisions, cached_holding_collisions)
        if chosen_op_param['is_feasible']:
            self.evaled_actions.append(chosen_op_param['action_parameters'])
            self.evaled_q_values.append('update_me')
            self.idx_to_update = len(self.evaled_actions) - 1
        else:
            print self.node.operator_skeleton.type + " sampling failed"
            self.evaled_actions.append(chosen_op_param['action_parameters'])
            worst_possible_rwd = self.node.depth + self.problem_env.reward_function.worst_reward
            self.evaled_q_values.append(worst_possible_rwd)

        return chosen_op_param

    def update_evaled_values(self):
        executed_actions_in_node = self.node.Q.keys()
        executed_action_values_in_node = self.node.Q.values()
        if len(executed_action_values_in_node) == 0:
            return

        if self.idx_to_update is not None:
            found = False
            for a, q in zip(executed_actions_in_node, executed_action_values_in_node):
                if np.all(np.isclose(self.evaled_actions[self.idx_to_update],
                                     a.continuous_parameters['action_parameters'])):
                    found = True
                    break
            try:
                assert found
            except AssertionError:
                print "idx to update not found"
                import pdb;
                pdb.set_trace()

            self.evaled_q_values[self.idx_to_update] = q

        # What does the code snippet below do? Update the feasible operator instances? Why?
        # We need to assert that idxs other than self.idx_to_update has the same value
        assert np.array_equal(np.array(self.evaled_q_values).sort(),
                              np.array(executed_action_values_in_node).sort()), "Are you using N_r?"

    def sample_using_voo(self):
        is_sample_from_best_v_region = self.is_time_to_sample_from_best_v_region()
        if is_sample_from_best_v_region:
            stime = time.time()
            cont_parameters = self.sample_from_best_voronoi_region()
            print "Best V region sampling time", time.time() - stime
        else:
            cont_parameters = self.sample_from_uniform()

        return cont_parameters

    def is_time_to_sample_from_best_v_region(self):
        is_more_than_one_action_in_node = len(self.evaled_actions) > 1
        if is_more_than_one_action_in_node:
            feasible_actions = [a for a in self.node.A if a.continuous_parameters['is_feasible']]
            we_have_feasible_action = len(feasible_actions) > 0
        else:
            we_have_feasible_action = False

        rnd = np.random.random()
        is_sample_from_best_v_region = rnd < (1 - self.explr_p) and we_have_feasible_action

        return is_sample_from_best_v_region

    def get_best_evaled_action(self):
        DEBUG = True
        if DEBUG:
            if 'update_me' in self.evaled_q_values:
                try:
                    best_action_idxs = np.argwhere(self.evaled_q_values[:-1] == np.amax(self.evaled_q_values[:-1]))
                except:
                    import pdb;
                    pdb.set_trace()
            else:
                best_action_idxs = np.argwhere(self.evaled_q_values == np.amax(self.evaled_q_values))
            best_action_idxs = best_action_idxs.reshape((len(best_action_idxs, )))
            best_action_idx = np.random.choice(best_action_idxs)
        else:
            best_action_idxs = np.argwhere(self.evaled_q_values == np.amax(self.evaled_q_values))
            best_action_idxs = best_action_idxs.reshape((len(best_action_idxs, )))
            best_action_idx = np.random.choice(best_action_idxs)
        return self.evaled_actions[best_action_idx]

    def centered_uniform_sample_near_best_action(self, best_evaled_action, counter):
        dim_x = self.domain[1].shape[-1]
        possible_max = (self.domain[1] - best_evaled_action) / np.exp(self.counter_ratio * counter)
        possible_min = (self.domain[0] - best_evaled_action) / np.exp(self.counter_ratio * counter)

        possible_values = np.random.uniform(possible_min, possible_max, (dim_x,))
        new_parameters = best_evaled_action + possible_values
        while np.any(new_parameters > self.domain[1]) or np.any(new_parameters < self.domain[0]):
            possible_values = np.random.uniform(possible_min, possible_max, (dim_x,))
            new_parameters = best_evaled_action + possible_values
        return new_parameters

    def gaussian_sample_near_best_action(self, best_evaled_action, counter):
        variance = (self.domain[1] - self.domain[0]) / np.exp(self.counter_ratio * counter)
        new_parameters = np.random.normal(best_evaled_action, variance)
        new_parameters = np.clip(new_parameters, self.domain[0], self.domain[1])

        return new_parameters

    def uniform_sample_near_best_action(self, best_evaled_action):
        dim_x = self.domain[1].shape[-1]
        new_parameters = np.random.uniform(self.domain[0], self.domain[1], (dim_x,))
        return new_parameters

    def sample_from_best_voronoi_region(self):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 0

        best_evaled_action = self.get_best_evaled_action()
        other_actions = self.evaled_actions

        new_parameters = None
        closest_best_dist = np.inf
        #print "Q diff", np.max(self.node.Q.values()) - np.min(self.node.Q.values())
        max_counter = 1000  # 100 vs 1000 does not really make difference in MCD domain
        # todo I think I can squeeze out performance by using gaussian in higher dimension
        while np.any(best_dist > other_dists) and counter < max_counter:
            new_parameters = self.sample_near_best_action(best_evaled_action, counter)
            best_dist = self.dist_fcn(new_parameters, best_evaled_action)
            other_dists = np.array([self.dist_fcn(other, new_parameters) for other in other_actions])
            counter += 1

            if closest_best_dist > best_dist:
                closest_best_dist = best_dist
                best_other_dists = other_dists
                best_parameters = new_parameters

        print "Counter ", counter
        print "n actions = ", len(self.evaled_actions)
        if counter >= max_counter:
            self.sampling_mode = 'gaussian'
            print closest_best_dist, best_other_dists
            return best_parameters
        else:
            return new_parameters

    def sample_near_best_action(self, best_evaled_action, counter):
        if self.sampling_mode == 'gaussian':
            new_parameters = self.gaussian_sample_near_best_action(best_evaled_action, counter)
        elif self.sampling_mode == 'centered_uniform':
            new_parameters = self.centered_uniform_sample_near_best_action(best_evaled_action, counter)
        else:
            new_parameters = self.uniform_sample_near_best_action(best_evaled_action)
        return new_parameters
