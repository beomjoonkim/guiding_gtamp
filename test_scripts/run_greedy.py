import argparse
import pickle
import time
import numpy as np
import socket
import random
import os
import tensorflow as tf

from gtamp_problem_environments.mover_env import PaPMoverEnv
from gtamp_problem_environments.one_arm_mover_env import PaPOneArmMoverEnv
from planners.subplanners.motion_planner import BaseMotionPlanner
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from gtamp_utils import utils
from planners.sahs.greedy import search
from learn.pap_gnn import PaPGNN
import collections



def get_problem_env(config):
    n_objs_pack = config.n_objs_pack
    if config.domain == 'two_arm_mover':
        problem_env = PaPMoverEnv(config.pidx)
        goal = ['home_region'] + [obj.GetName() for obj in problem_env.objects[:n_objs_pack]]
        problem_env.set_goal(goal)
    elif config.domain == 'one_arm_mover':
        problem_env = PaPOneArmMoverEnv(config.pidx)
        goal = ['rectangular_packing_box1_region'] + [obj.GetName() for obj in problem_env.objects[:n_objs_pack]]
        problem_env.set_goal(goal)
    else:
        raise NotImplementedError
    return problem_env


def get_solution_file_name(config):
    hostname = socket.gethostname()
    if hostname in {'dell-XPS-15-9560', 'phaedra', 'shakey', 'lab', 'glaucus', 'luke-laptop-1'}:
        root_dir = './'
    else:
        root_dir = '/data/public/rw/pass.port/guiding_gtamp/'

    solution_file_dir = root_dir + '/test_results/sahs_results/domain_%s/n_objs_pack_%d' \
                        % (config.domain, config.n_objs_pack)

    if config.dont_use_gnn:
        solution_file_dir += '/no_gnn/'
    elif config.dont_use_h:
        solution_file_dir += '/gnn_no_h/loss_' + str(config.loss) + '/num_train_' + str(config.num_train) + '/'
    elif config.hcount:
        solution_file_dir += '/hcount/'
    elif config.state_hcount:
        solution_file_dir += '/state_hcount/'
    elif config.qlearned_hcount:
        solution_file_dir += '/qlearned_hcount_obj_already_in_goal/loss_' + str(config.loss) + '/num_train_' + str(config.num_train) \
                             + '/mse_weight_' + str(config.mse_weight) + '/mix_rate_' + str(config.mixrate) + '/'
    else:
        solution_file_dir += '/gnn/loss_' + str(config.loss) + '/num_train_' + str(config.num_train) \
                              + '/mse_weight_' + str(config.mse_weight) + '/'

    solution_file_name = 'pidx_' + str(config.pidx) + \
                         '_planner_seed_' + str(config.planner_seed) + \
                         '_train_seed_' + str(config.train_seed) + \
                         '_domain_' + str(config.domain) + '.pkl'
    if not os.path.isdir(solution_file_dir):
        os.makedirs(solution_file_dir)

    solution_file_name = solution_file_dir + solution_file_name
    return solution_file_name


def parse_arguments():
    parser = argparse.ArgumentParser(description='Greedy planner')
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-train_seed', type=int, default=0)
    parser.add_argument('-planner_seed', type=int, default=0)
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-num_train', type=int, default=7000)
    parser.add_argument('-timelimit', type=float, default=300)
    parser.add_argument('-mixrate', type=float, default=1.0)
    parser.add_argument('-mse_weight', type=float, default=1.0)
    parser.add_argument('-visualize_plan', action='store_true', default=False)
    parser.add_argument('-visualize_sim', action='store_true', default=False)
    parser.add_argument('-dontsimulate', action='store_true', default=False)
    parser.add_argument('-f', action='store_true', default=False)
    parser.add_argument('-dont_use_gnn', action='store_true', default=False)
    parser.add_argument('-dont_use_h', action='store_true', default=False)
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-problem_type', type=str, default='normal')  # supports normal, nonmonotonic
    parser.add_argument('-hcount', action='store_true', default=False)
    parser.add_argument('-qlearned_hcount', action='store_true', default=False)
    parser.add_argument('-state_hcount', action='store_true', default=False)
    config = parser.parse_args()
    return config


def set_problem_env_config(problem_env, config):
    np.random.seed(config.planner_seed)
    random.seed(config.planner_seed)
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'prm'))
    problem_env.seed = config.pidx
    problem_env.init_saver = DynamicEnvironmentStateSaver(problem_env.env)


def get_pap_gnn_model(mover, config):
    if not config.hcount:
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
            mse_weight=config.mse_weight,
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
        entity_names = mover.entity_names

        with tf.variable_scope('pap'):
            pap_model = PaPGNN(num_entities, num_node_features, num_edge_features, pap_mconfig, entity_names, n_regions)
        pap_model.load_weights()
    else:
        pap_model = None

    return pap_model


def main():
    config = parse_arguments()

    np.random.seed(config.pidx)
    random.seed(config.pidx)

    problem_env = get_problem_env(config)
    set_problem_env_config(problem_env, config)

    pap_model = get_pap_gnn_model(problem_env, config)
    solution_file_name = get_solution_file_name(config)

    is_problem_solved_before = os.path.isfile(solution_file_name)
    plan_length = 0
    num_nodes = 0
    if is_problem_solved_before and not config.f:
        with open(solution_file_name, 'rb') as f:
            trajectory = pickle.load(f)
            success = trajectory['success']
            tottime = trajectory['tottime']
    else:
        t = time.time()
        plan, num_nodes = search(problem_env, config, pap_model)
        tottime = time.time() - t
        success = plan is not None
        plan_length = len(plan) if success else 0

        data = {
            'n_objs_pack': config.n_objs_pack,
            'tottime': tottime,
            'success': success,
            'plan_length': plan_length,
            'num_nodes': num_nodes,
            'plan': plan
        }

        with open(solution_file_name, 'wb') as f:
            pickle.dump(data, f)
    print 'Time: %.2f Success: %d Plan length: %d Num nodes: %d' % (tottime, success, plan_length, num_nodes)


if __name__ == '__main__':
    main()
