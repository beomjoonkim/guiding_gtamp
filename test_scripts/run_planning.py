import numpy as np
import random
import argparse
import socket
import os
import sys
import collections
import tensorflow as tf

from learn.pap_gnn import PaPGNN
from gtamp_problem_environments.mover_env import Mover, PaPMoverEnv
from planners.subplanners.motion_planner import OperatorBaseMotionPlanner
from gtamp_problem_environments.reward_functions.reward_function import GenericRewardFunction
from gtamp_problem_environments.reward_functions.packing_problem.reward_function import ShapedRewardFunction
from planners.flat_mcts.mcts import MCTS


def make_and_get_save_dir(parameters):
    hostname = socket.gethostname()
    if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra' or hostname == 'shakey' or hostname == 'lab' or \
            hostname == 'glaucus':
        root_dir = './'
    else:
        root_dir = '/data/public/rw/pass.port/tamp_q_results/'

    save_dir = root_dir + '/test_results/mcts_results_on_mover_domain/' \
               + 'n_objs_pack_' + str(parameters.n_objs_pack) + '/' \
               + 'n_mp_params_' + str(parameters.n_motion_plan_trials) + '/' \
               + 'widening_' + str(parameters.w) \
               + '/uct_' + str(parameters.uct)
    pidx = parameters.pidx

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if os.path.isfile(save_dir + '/' + str(pidx) + '.pkl'):
        print "Already done"
        if not parameters.f:
            sys.exit(-1)

    return save_dir


def parse_mover_problem_parameters():
    parser = argparse.ArgumentParser(description='planner parameters')

    # Problem-specific parameters
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-planner', type=str, default='mcts')
    parser.add_argument('-domain', type=str, default='two_arm_mover')

    # Planner-agnostic parameters
    parser.add_argument('-sampling_strategy', type=str, default='unif')
    parser.add_argument('-timelimit', type=int, default=300)
    parser.add_argument('-dont_use_learned_q', action='store_false', default=True)
    parser.add_argument('-n_feasibility_checks', type=int, default=200)
    parser.add_argument('-n_motion_plan_trials', type=int, default=3)

    # Learning-related parameters
    parser.add_argument('-train_seed', type=int, default=0)
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-num_train', type=int, default=5000)

    # MCTS parameters
    parser.add_argument('-n_switch', type=int, default=5)
    parser.add_argument('-uct', type=float, default=0.1)
    parser.add_argument('-w', type=float, default=3)
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-mcts_iter', type=int, default=1000)
    parser.add_argument('-use_learned_q', action='store_true', default=False)
    parser.add_argument('-use_ucb', action='store_true', default=False)
    parser.add_argument('-pw', action='store_true', default=False)
    parser.add_argument('-f', action='store_true', default=False)

    parameters = parser.parse_args()
    return parameters


def instantiate_mcts(parameters, problem_env, goal_entities, learned_q):
    planner = MCTS(parameters.w,
                   parameters.uct,
                   parameters.n_feasibility_checks,
                   problem_env,
                   depth_limit=11,
                   discount_rate=1,
                   check_reachability=True,
                   use_progressive_widening=parameters.pw,
                   use_ucb=parameters.use_ucb,
                   learned_q_function=learned_q,
                   n_motion_plan_trials=parameters.n_motion_plan_trials,
                   goal_entities=goal_entities)

    return planner


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def load_learned_q(config, problem_env):
    mconfig_type = collections.namedtuple('mconfig_type',
                                          'operator n_msg_passing n_layers num_fc_layers n_hidden no_goal_nodes top_k '
                                          'optimizer lr use_mse batch_size seed num_train val_portion num_test mse_'
                                          'weight diff_weight_msg_passing same_vertex_model weight_initializer loss')

    assert config.num_train <= 5000
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
        num_test=1882,
        mse_weight=1.0,
        diff_weight_msg_passing=False,
        same_vertex_model=False,
        weight_initializer='glorot_uniform',
        loss=config.loss,
    )

    if config.domain == 'two_arm_mover':
        num_entities = 11
        n_regions = 2
    elif config.domain == 'one_arm_mover':
        num_entities = 17
        n_regions = 3
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
    set_seed(parameters.pidx)
    problem_env = PaPMoverEnv(parameters.pidx)

    goal_object_names = [obj.GetName() for obj in problem_env.objects[:parameters.n_objs_pack]]
    goal_region_name = [problem_env.regions['home_region'].name]
    goal_entities = goal_object_names + goal_region_name
    #reward_function = GenericRewardFunction(problem_env, goal_object_names, goal_region_name[0])
    reward_function = ShapedRewardFunction(problem_env, goal_object_names, goal_region_name[0])


    motion_planner = OperatorBaseMotionPlanner(problem_env, 'prm')

    problem_env.set_reward_function(reward_function)
    problem_env.set_motion_planner(motion_planner)

    if parameters.use_learned_q:
        learned_q = load_learned_q(parameters, problem_env)
    else:
        learned_q = None

    if parameters.planner == 'mcts':
        planner = instantiate_mcts(parameters, problem_env, goal_entities, learned_q)
    else:
        raise NotImplementedError

    planner.search()


if __name__ == '__main__':
    main()
