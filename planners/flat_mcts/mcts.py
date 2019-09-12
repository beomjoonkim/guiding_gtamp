from mcts_tree_continuous_node import ContinuousTreeNode
from discrete_node_with_psa import DiscreteTreeNodeWithPsa
from mcts_tree_discrete_pap_node import PaPDiscreteTreeNodeWithPriorQ
from discrete_node_with_prior_q import DiscreteTreeNodeWithPriorQ
from mcts_tree import MCTSTree

from generators.uniform import UniformPaPGenerator
from generators.voo import PaPVOOGenerator

from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from trajectory_representation.state import StateWithoutCspacePredicates
from trajectory_representation.one_arm_pap_state import OneArmPaPState

## openrave helper libraries
from gtamp_utils import utils


import numpy as np
import sys
import socket
import pickle
import time
import os

sys.setrecursionlimit(15000)
DEBUG = False

hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560':
    from mcts_graphics import write_dot_file


# todo
#   create MCTS for each environment. Each one will have different compute_state functions
class MCTS:
    def __init__(self, parameters, problem_env, goal_entities, v_fcn, learned_q):
        # MCTS parameters
        self.widening_parameter = parameters.widening_parameter
        self.ucb_parameter = parameters.ucb_parameter
        self.time_limit = parameters.timelimit
        self.n_motion_plan_trials = parameters.n_motion_plan_trials
        self.use_ucb = parameters.use_ucb
        self.use_progressive_widening = parameters.pw
        self.n_feasibility_checks = parameters.n_feasibility_checks
        self.use_v_fcn = parameters.use_learned_q
        self.v_fcn = v_fcn
        self.learned_q = learned_q
        self.use_shaped_reward = parameters.use_shaped_reward
        self.planning_horizon = parameters.planning_horizon
        self.sampling_strategy = parameters.sampling_strategy
        self.explr_p = parameters.explr_p
        self.switch_frequency = parameters.switch_frequency

        # Hard-coded params
        self.check_reachability = True
        self.discount_rate = 1.0

        # Environment setup
        self.problem_env = problem_env
        self.env = self.problem_env.env
        self.robot = self.problem_env.robot

        # MCTS initialization
        self.s0_node = None
        self.tree = MCTSTree(self.ucb_parameter)
        self.best_leaf_node = None
        self.goal_entities = goal_entities

        # Logging purpose
        self.search_time_to_reward = []
        self.reward_lists = []
        self.progress_list = []

        self.found_solution = False
        self.swept_volume_constraint = None

    def load_pickled_tree(self, fname=None):
        if fname is None:
            fname = 'tmp_tree.pkl'
        self.tree = pickle.load(open(fname, 'r'))

    def visit_all_nodes(self, curr_node):
        children = curr_node.children.values()
        print curr_node in self.tree.nodes
        for c in children:
            self.visit_all_nodes(c)

    def save_tree(self, fname=None):
        if fname is None:
            fname = 'tmp_tree.pkl'
        self.tree.make_tree_picklable()
        pickle.dump(self.tree, open(fname, 'wb'))

    def load_tree(self, fname=None):
        if fname is None:
            fname = 'tmp_tree.pkl'
        self.tree = pickle.load(open(fname, 'r'))

    def get_node_at_idx(self, idx):
        for n in self.tree.nodes:
            if n.idx == idx:
                return n
        return None

    def create_sampling_agent(self, node):
        operator_skeleton = node.operator_skeleton
        if 'one_arm' in self.problem_env.name:
            dont_check_motion_existence = True
        else:
            dont_check_motion_existence = False
        if self.sampling_strategy == 'uniform':
            generator = UniformPaPGenerator(node, operator_skeleton, self.problem_env, None,
                                            n_candidate_params_to_smpl=self.n_motion_plan_trials,
                                            total_number_of_feasibility_checks=self.n_feasibility_checks,
                                            dont_check_motion_existence=dont_check_motion_existence)

        elif self.sampling_strategy == 'voo':
            return PaPVOOGenerator(node,
                                   operator_skeleton,
                                   self.problem_env,
                                   None,
                                   self.n_feasibility_checks,
                                   self.n_motion_plan_trials,
                                   dont_check_motion_existence,
                                   explr_p=self.explr_p,
                                   c1=1,
                                   sampling_mode='gaussian',
                                   counter_ratio=1)
        else:
            raise NotImplementedError
        return generator

    def compute_state(self, parent_node, parent_action):
        if self.problem_env.is_goal_reached():
            state = parent_node.state
        else:
            if parent_node is None:
                parent_state = None
            else:
                parent_state = parent_node.state
            # where is the parent state?
            if self.problem_env.name.find('one_arm') != -1:
                state = OneArmPaPState(self.problem_env,
                                       parent_state=parent_state,
                                       parent_action=parent_action,
                                       goal_entities=self.goal_entities)
            else:
                if socket.gethostname() == 'dell-XPS-15-9560':
                    if parent_node is None:
                        idx = -1
                    else:
                        idx = parent_node.idx
                    fname = './tmp_%d.pkl' % idx
                    if os.path.isfile(fname):
                        state = pickle.load(open(fname, 'r'))
                        state.make_plannable(self.problem_env)
                    else:
                        state = ShortestPathPaPState(self.problem_env,  # what's this?
                                                     parent_state=parent_state,
                                                     parent_action=parent_action,
                                                     goal_entities=self.goal_entities, planner='mcts')
                        state.make_pklable()
                        pickle.dump(state, open(fname, 'wb'))
                        state.make_plannable(self.problem_env)
                else:
                    state = ShortestPathPaPState(self.problem_env,  # what's this?
                                                 parent_state=parent_state,
                                                 parent_action=parent_action,
                                                 goal_entities=self.goal_entities, planner='mcts')
        return state

    def get_current_state(self, parent_node, parent_action, is_parent_action_infeasible):
        # this needs to be factored
        # why do I need a parent node? Can I just get away with parent state?
        is_operator_skeleton_node = (parent_node is None) or (not parent_node.is_operator_skeleton_node)
        if is_parent_action_infeasible:
            state = None
        elif is_operator_skeleton_node:
            state = self.compute_state(parent_node, parent_action)
        else:
            state = parent_node.state

        return state

    def create_node(self, parent_action, depth, parent_node, is_parent_action_infeasible, is_init_node=False):
        state_saver = utils.CustomStateSaver(self.problem_env.env)
        is_operator_skeleton_node = (parent_node is None) or (not parent_node.is_operator_skeleton_node)
        state = self.get_current_state(parent_node, parent_action, is_parent_action_infeasible)

        if is_operator_skeleton_node:
            applicable_op_skeletons = self.problem_env.get_applicable_ops(parent_action)
            node = DiscreteTreeNodeWithPriorQ(state, self.ucb_parameter, depth, state_saver, is_operator_skeleton_node,
                                              is_init_node, applicable_op_skeletons, self.learned_q)
        else:
            node = ContinuousTreeNode(state, parent_action, self.ucb_parameter, depth, state_saver,
                                      is_operator_skeleton_node, is_init_node)
            node.sampling_agent = self.create_sampling_agent(node)

        node.parent = parent_node
        node.parent_action = parent_action
        return node

    @staticmethod
    def get_best_child_node(node):
        if len(node.children) == 0:
            return None
        else:
            best_child_action_idx = np.argmax(node.Q.values())
            best_child_action = node.Q.keys()[best_child_action_idx]
            return node.children[best_child_action]

    def retrace_best_plan(self):
        plan = []
        _, _, best_leaf_node = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
        curr_node = best_leaf_node

        while not curr_node.parent is None:
            plan.append(curr_node.parent_action)
            curr_node = curr_node.parent

        plan = plan[::-1]
        return plan, best_leaf_node

    def get_best_goal_node(self):
        leaves = self.tree.get_leaf_nodes()
        goal_nodes = [leaf for leaf in leaves if leaf.is_goal_node]
        if len(goal_nodes) > 1:
            best_traj_reward, curr_node, _ = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
        else:
            curr_node = goal_nodes[0]
        return curr_node

    def switch_init_node(self, node):
        self.s0_node.is_init_node = False
        self.s0_node = node
        self.s0_node.is_init_node = True
        self.problem_env.reset_to_init_state(node)
        self.found_solution = False

    @staticmethod
    def choose_child_node_to_descend_to(node):
        # todo: implement the one with highest visitation
        if node.is_operator_skeleton_node and len(node.A) == 1:
            # descend to grand-child
            only_child_node = node.children.values()[0]
            best_action = only_child_node.Q.keys()[np.argmax(only_child_node.Q.values())]
            best_node = only_child_node.children[best_action]
        else:
            best_action = node.Q.keys()[np.argmax(node.Q.values())]
            best_node = node.children[best_action]
        return best_node

    def log_current_tree_to_dot_file(self, iteration, node_to_search_from):
        if socket.gethostname() == 'dell-XPS-15-9560':
            write_dot_file(self.tree, iteration, '', node_to_search_from)

    def log_performance(self, time_to_search, iteration):
        best_traj_rwd, progress, best_node = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
        self.search_time_to_reward.append([time_to_search, iteration, best_traj_rwd, progress, self.found_solution])
        self.progress_list.append(progress)
        self.best_leaf_node = best_node

    def is_optimal_solution_found(self):
        best_traj_rwd, best_node, reward_list = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
        if self.found_solution:
            return True
            # if self.problem_env.reward_function.is_optimal_plan_found(best_traj_rwd):
            #    print "Optimal score found"
            #    return True
            # else:
            #    return False
        else:
            return False

    def search(self, n_iter=np.inf, iteration_for_tree_logging=0, node_to_search_from=None, max_time=np.inf):
        depth = 0
        time_to_search = 0

        if node_to_search_from is None:
            self.s0_node = self.create_node(None,
                                            depth=0,
                                            parent_node=None,
                                            is_parent_action_infeasible=False,
                                            is_init_node=True)
            self.tree.set_root_node(self.s0_node)
            node_to_search_from = self.s0_node

        new_trajs = []
        plan = []
        if n_iter == np.inf:
            n_iter = 999999
        for iteration in range(1, n_iter):
            print '*****SIMULATION ITERATION %d' % iteration
            self.problem_env.reset_to_init_state(node_to_search_from)

            new_traj = []
            stime = time.time()
            self.simulate(node_to_search_from, node_to_search_from, depth, new_traj)
            time_to_search += time.time() - stime
            new_trajs.append(new_traj)

            # note that I need to evaluate all actions in a node to switch
            is_time_to_switch_node = iteration % self.switch_frequency == 0  # and the node should be feasible
            # I have to have a feasible action to switch if this is an instance node
            if is_time_to_switch_node:
                if node_to_search_from.is_operator_skeleton_node:
                    node_to_search_from = node_to_search_from.get_child_with_max_value()
                else:
                    max_child = node_to_search_from.get_child_with_max_value()
                    if max_child.parent_action.continuous_parameters['is_feasible']:
                        node_to_search_from = node_to_search_from.get_child_with_max_value()

            if iteration % 10 == 0:
                self.log_current_tree_to_dot_file(iteration_for_tree_logging + iteration, node_to_search_from)

            self.log_performance(time_to_search, iteration)
            print self.search_time_to_reward[iteration_for_tree_logging:]

            # break if the solution is found
            if self.is_optimal_solution_found():
                print "Optimal score found"
                plan, _ = self.retrace_best_plan()
                break

            if time_to_search > max_time:
                print "Time is up"
                break

        self.problem_env.reset_to_init_state(node_to_search_from)
        return self.search_time_to_reward, plan

    def get_best_trajectory(self, node_to_search_from, trajectories):
        traj_rewards = []
        curr_node = node_to_search_from
        for trj in trajectories:
            traj_sum_reward = 0
            for aidx, a in enumerate(trj):
                traj_sum_reward += np.power(self.discount_rate, aidx) * curr_node.reward_history[a][0]
                curr_node = curr_node.children[a]
            traj_rewards.append(traj_sum_reward)
        return trajectories[np.argmax(traj_rewards)], curr_node

    def choose_action(self, curr_node):
        if curr_node.is_operator_skeleton_node:
            print "Skeleton node"
            # here, perform psa with the learned q
            action = curr_node.perform_ucb_over_actions()
        else:
            print 'Instance node'
            if curr_node.sampling_agent is None:  # this happens if the tree has been pickled
                curr_node.sampling_agent = self.create_sampling_agent(curr_node)
            if not curr_node.is_reevaluation_step(self.widening_parameter,
                                                  self.problem_env.reward_function.infeasible_reward,
                                                  self.use_progressive_widening,
                                                  self.use_ucb):
                print "Sampling new action"
                new_continuous_parameters = self.sample_continuous_parameters(curr_node)
                curr_node.add_actions(new_continuous_parameters)
                action = curr_node.A[-1]
            else:
                print "Re-evaluation of actions"
                if self.use_ucb:
                    action = curr_node.perform_ucb_over_actions()
                else:
                    action = curr_node.choose_new_arm()
        return action

    @staticmethod
    def update_goal_node_statistics(curr_node, reward):
        # todo rewrite this function
        curr_node.Nvisited += 1
        curr_node.reward = reward

    def simulate(self, curr_node, node_to_search_from, depth, new_traj):
        if self.problem_env.reward_function.is_goal_reached():
            if not curr_node.is_goal_and_already_visited:
                self.found_solution = True
                curr_node.is_goal_node = True
                print "Solution found, returning the goal reward", self.problem_env.reward_function.goal_reward
                self.update_goal_node_statistics(curr_node, self.problem_env.reward_function.goal_reward)
            return self.problem_env.reward_function.goal_reward

        if depth == self.planning_horizon:
            # would it ever get here? why does it not satisfy the goal?
            print "Depth limit reached"
            return 0

        if DEBUG:
            print "At depth ", depth
            print "Is it time to pick?", self.problem_env.is_pick_time()

        action = self.choose_action(curr_node)

        is_action_feasible = self.apply_action(curr_node, action)

        if not curr_node.is_action_tried(action):
            next_node = self.create_node(action, depth + 1, curr_node, not is_action_feasible)
            self.tree.add_node(next_node, action, curr_node)
            reward = self.problem_env.reward_function(curr_node.state, next_node.state, action, depth)
            next_node.parent_action_reward = reward
            next_node.sum_ancestor_action_rewards = next_node.parent.sum_ancestor_action_rewards + reward
        else:
            next_node = curr_node.children[action]
            reward = next_node.parent_action_reward

        print "Reward", reward

        if not is_action_feasible:
            # this (s,a) is a dead-end
            print "Infeasible action"
            if self.use_v_fcn:
                sum_rewards = reward + curr_node.parent.v_fcn[curr_node.parent_action]
                print sum_rewards
            else:
                sum_rewards = reward
        else:
            sum_rewards = reward + self.discount_rate * self.simulate(next_node, node_to_search_from, depth + 1,
                                                                      new_traj)

        curr_node.update_node_statistics(action, sum_rewards, reward)
        if curr_node == node_to_search_from and curr_node.parent is not None:
            self.update_ancestor_node_statistics(curr_node.parent, curr_node.parent_action, sum_rewards)

        # todo return a plan
        return sum_rewards

    def update_ancestor_node_statistics(self, node, action, child_sum_rewards):
        if node is None:
            return
        parent_reward_to_node = node.reward_history[action][0]
        parent_sum_rewards = parent_reward_to_node + self.discount_rate * child_sum_rewards
        node.update_node_statistics(action, parent_sum_rewards, parent_reward_to_node)
        self.update_ancestor_node_statistics(node.parent, node.parent_action, parent_sum_rewards)

    def apply_action(self, node, action):
        if node.is_operator_skeleton_node:
            print "Applying skeleton", action.type, action.discrete_parameters['object'], \
                action.discrete_parameters['region']
            is_feasible = self.problem_env.apply_operator_skeleton(node.state, action)
        else:
            print "Applying instance", action.type, action.discrete_parameters['object'], action.discrete_parameters[
                'region']
            is_feasible = self.problem_env.apply_operator_instance(node.state, action, self.check_reachability)

        return is_feasible

    def sample_continuous_parameters(self, node):
        if self.problem_env.name.find('one_arm') != -1:
            raise NotImplementedError
        else:
            if isinstance(node.state, StateWithoutCspacePredicates):
                current_collides = None
            else:
                current_collides = node.state.collisions_at_all_obj_pose_pairs

            smpled_param = node.sampling_agent.sample_next_point(cached_collisions=current_collides,
                                                                 cached_holding_collisions=None)
        return smpled_param
