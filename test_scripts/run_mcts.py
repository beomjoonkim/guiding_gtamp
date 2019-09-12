import numpy as np
import random
import argparse
import socket
import os
import sys
import collections
import tensorflow as tf
import pickle

from learn.pap_gnn import PaPGNN
from gtamp_problem_environments.mover_env import Mover, PaPMoverEnv
from planners.subplanners.motion_planner import OperatorBaseMotionPlanner
from gtamp_problem_environments.reward_functions.reward_function import GenericRewardFunction
from gtamp_problem_environments.reward_functions.packing_problem.reward_function import ShapedRewardFunction
from planners.flat_mcts.mcts import MCTS
from planners.flat_mcts.mcts_with_leaf_strategy import MCTSWithLeafStrategy
from planners.heuristics import compute_hcount_with_action, get_objects_to_move


def make_and_get_save_dir(parameters, filename):
    hostname = socket.gethostname()
    if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra' or hostname == 'shakey' or hostname == 'lab' or \
            hostname == 'glaucus':
        root_dir = './'
    else:
        root_dir = '/data/public/rw/pass.port/guiding_gtamp/'

    save_dir = root_dir + '/test_results/mcts_results_with_q_bonus/' \
               + 'domain_' + str(parameters.domain) + '/' \
               + 'n_objs_pack_' + str(parameters.n_objs_pack) + '/' \
               + 'sampling_strategy_' + str(parameters.sampling_strategy) + '/' \
               + 'n_mp_trials_' + str(parameters.n_motion_plan_trials) + '/' \
               + 'widening_' + str(parameters.widening_parameter) + '/' \
               + 'uct_' + str(parameters.ucb_parameter) + '/' \
               + 'switch_frequency_' + str(parameters.switch_frequency) + '/' \
               + 'reward_shaping_' + str(parameters.use_shaped_reward) + '/' \
               + 'learned_q_' + str(parameters.use_learned_q) + '/'

    if 'uniform' not in parameters.sampling_strategy:
        save_dir += 'explr_p_' + str(parameters.explr_p) + '/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if os.path.isfile(save_dir + '/' + filename):
        print "Already done"
        if not parameters.f:
            sys.exit(-1)
    return save_dir


def parse_mover_problem_parameters():
    parser = argparse.ArgumentParser(description='planner parameters')

    # Problem-specific parameters
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-planner', type=str, default='mcts_with_leaf_strategy')
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-planner_seed', type=int, default=0)

    # Planner-agnostic parameters
    parser.add_argument('-timelimit', type=int, default=300)
    parser.add_argument('-dont_use_learned_q', action='store_false', default=True)
    parser.add_argument('-n_feasibility_checks', type=int, default=200)
    parser.add_argument('-n_motion_plan_trials', type=int, default=3)
    parser.add_argument('-planning_horizon', type=int, default=3 * 8)

    # Learning-related parameters
    parser.add_argument('-train_seed', type=int, default=0)
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-num_train', type=int, default=7000)
    parser.add_argument('-use_region_agnostic', action='store_true', default=False)

    # MCTS parameters
    parser.add_argument('-switch_frequency', type=int, default=50)
    parser.add_argument('-ucb_parameter', type=float, default=0.1)
    parser.add_argument('-widening_parameter', type=float, default=50)  # number of re-evals
    parser.add_argument('-explr_p', type=float, default=0.3)  # number of re-evals
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-mcts_iter', type=int, default=1000)
    parser.add_argument('-use_learned_q', action='store_true', default=False)
    parser.add_argument('-use_ucb', action='store_true', default=False)
    parser.add_argument('-pw', action='store_true', default=False)
    parser.add_argument('-f', action='store_true', default=False)  # what was this?
    parser.add_argument('-sampling_strategy', type=str, default='uniform')
    parser.add_argument('-use_shaped_reward', action='store_true', default=False)

    parameters = parser.parse_args()
    return parameters


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def load_learned_q(config, problem_env):
    mconfig_type = collections.namedtuple('mconfig_type',
                                          'operator n_msg_passing n_layers num_fc_layers n_hidden no_goal_nodes top_k optimizer lr use_mse batch_size seed num_train val_portion mse_weight diff_weight_msg_passing same_vertex_model weight_initializer loss use_region_agnostic')

    pap_mconfig = mconfig_type(
        operator='two_arm_pick_two_arm_place',
        n_msg_passing=1,
        n_layers=2,
        num_fc_layers=2,
        n_hidden=32,
        no_goal_nodes=False,

        top_k=1,
        optimizer='adam',
        lr=1e-4,
        use_mse=True,

        batch_size='32',
        seed=config.train_seed,
        num_train=config.num_train,
        val_portion=.1,
        mse_weight=0.0,
        diff_weight_msg_passing=False,
        same_vertex_model=False,
        weight_initializer='glorot_uniform',
        loss=config.loss,
        use_region_agnostic=False
    )
    if config.domain == 'two_arm_mover':
        num_entities = 10
        n_regions = 2
    elif config.domain == 'one_arm_mover':
        num_entities = 12
        n_regions = 2
    else:
        raise NotImplementedError
    num_node_features = 10
    num_edge_features = 44
    entity_names = problem_env.entity_names

    with tf.variable_scope('pap'):
        pap_model = PaPGNN(num_entities, num_node_features, num_edge_features, pap_mconfig, entity_names, n_regions)
    pap_model.load_weights()

    return pap_model


def main():
    parameters = parse_mover_problem_parameters()
    filename = 'pidx_%d_planner_seed_%d.pkl' % (parameters.pidx, parameters.planner_seed)
    save_dir = make_and_get_save_dir(parameters, filename)

    set_seed(parameters.pidx)
    problem_env = PaPMoverEnv(parameters.pidx)

    goal_object_names = [obj.GetName() for obj in problem_env.objects[:parameters.n_objs_pack]]
    goal_region_name = [problem_env.regions['home_region'].name]
    goal = goal_region_name + goal_object_names
    problem_env.set_goal(goal)

    goal_entities = goal_object_names + goal_region_name
    if parameters.use_shaped_reward:
        reward_function = ShapedRewardFunction(problem_env, goal_object_names, goal_region_name[0],
                                               parameters.planning_horizon)
    else:
        reward_function = GenericRewardFunction(problem_env, goal_object_names, goal_region_name[0],
                                                parameters.planning_horizon)

    motion_planner = OperatorBaseMotionPlanner(problem_env, 'prm')

    problem_env.set_reward_function(reward_function)
    problem_env.set_motion_planner(motion_planner)

    learned_q = None
    prior_q = None
    if parameters.use_learned_q:
        learned_q = load_learned_q(parameters, problem_env)

    v_fcn = lambda state: -len(get_objects_to_move(state, problem_env))

    if parameters.planner == 'mcts':
        planner = MCTS(parameters, problem_env, goal_entities, prior_q, learned_q)
    elif parameters.planner == 'mcts_with_leaf_strategy':
        planner = MCTSWithLeafStrategy(parameters, problem_env, goal_entities, v_fcn, learned_q)
    else:
        raise NotImplementedError

    set_seed(parameters.planner_seed)
    search_time_to_reward, plan = planner.search(max_time=parameters.timelimit)
    pickle.dump({"search_time_to_reward": search_time_to_reward, 'plan': plan,
                 'n_nodes': len(planner.tree.get_discrete_nodes())}, open(save_dir+filename, 'wb'))


if __name__ == '__main__':
    main()
