import argparse
import pickle
import time
import numpy as np
import socket
import random
import os
import tensorflow as tf
import collections

from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from gtamp_problem_environments.mover_env import PaPMoverEnv
from gtamp_problem_environments.one_arm_mover_env import PaPOneArmMoverEnv
from planners.subplanners.motion_planner import BaseMotionPlanner

from planners.sahs.greedy_new import search
from learn.pap_gnn import PaPGNN

from generators.learning.utils.model_creation_utils import create_imle_model


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

    if config.gather_planning_exp:
        root_dir = root_dir + '/planning_experience/'
        solution_file_dir = root_dir + '/domain_%s/n_objs_pack_%d' \
                            % (config.domain, config.n_objs_pack)
    else:
        solution_file_dir = root_dir + '/test_results/sahs_results/using_weights_for_submission/domain_%s/n_objs_pack_%d' \
                            % (config.domain, config.n_objs_pack)

    if config.dont_use_gnn:
        solution_file_dir += '/no_gnn/'
    elif config.dont_use_h:
        solution_file_dir += '/gnn_no_h/loss_' + str(config.loss) + '/num_train_' + str(config.num_train) + '/'
    elif config.hcount:
        solution_file_dir += '/hcount/'
    elif config.hcount_number_in_goal:
        solution_file_dir += '/hcount_number_in_goal/'
    elif config.state_hcount:
        solution_file_dir += '/state_hcount/'
    elif config.integrated:
        solution_file_dir += '/learned_sampler/'
        # What about the tamp-q? Take as its input
        solution_file_dir += '/integrated/shortest_irsc/'
        q_config = '/q_config_num_train_' + str(config.num_train) + \
                   '_mse_weight_' + str(config.mse_weight) + \
                   '_use_region_agnostic_' + str(config.use_region_agnostic) + \
                   '_mix_rate_' + str(config.mixrate) + '/'
        sampler_config = '/smpler_config_num_train_' + str(config.num_train) + '/'
        solution_file_dir = solution_file_dir + q_config + sampler_config
    elif config.qlearned_hcount:
        solution_file_dir += '/qlearned_hcount_obj_already_in_goal/shortest_irsc' \
                             '/loss_' + str(config.loss) + \
                             '/num_train_' + str(config.num_train) + \
                             '/mse_weight_' + str(config.mse_weight) + \
                             '/use_region_agnostic_' + str(config.use_region_agnostic) + \
                             '/mix_rate_' + str(config.mixrate) + '/'
    elif config.qlearned_hcount_new_number_in_goal:
        solution_file_dir += '/qlearned_hcount_obj_already_in_goal_new_number_in_goal/shortest_irsc/loss_' + str(
            config.loss) + '/num_train_' + str(config.num_train) \
                             + '/mse_weight_' + str(config.mse_weight) + '/use_region_agnostic_' + str(
            config.use_region_agnostic) \
                             + '/mix_rate_' + str(config.mixrate) + '/'
    elif config.qlearned_hcount_old_number_in_goal:
        solution_file_dir += '/qlearned_hcount_obj_already_in_goal_old_number_in_goal/shortest_irsc/' \
                             'loss_' + str(config.loss) + \
                             '/num_train_' + str(config.num_train) + \
                             '/mse_weight_' + str(config.mse_weight) + \
                             '/use_region_agnostic_' + str(config.use_region_agnostic) + \
                             '/mix_rate_' + str(config.mixrate) + '/'
    elif config.qlearned_old_number_in_goal:
        solution_file_dir += '/qlearned_old_number_in_goal//shortest_irsc/loss_' + str(config.loss) \
                             + '/num_train_' + str(config.num_train) \
                             + '/mse_weight_' + str(config.mse_weight) + '/use_region_agnostic_' \
                             + str(config.use_region_agnostic) + '/'
    elif config.qlearned_new_number_in_goal:
        solution_file_dir += '/qlearned_new_number_in_goal//shortest_irsc/loss_' + str(config.loss) \
                             + '/num_train_' + str(config.num_train) \
                             + '/mse_weight_' + str(config.mse_weight) + '/use_region_agnostic_' \
                             + str(config.use_region_agnostic) + '/'
    elif config.pure_learned_q:
        solution_file_dir += '/gnn/shortest_irsc/loss_' + str(config.loss) + '/num_train_' + str(config.num_train) \
                             + '/mse_weight_' + str(config.mse_weight) + '/use_region_agnostic_' \
                             + str(config.use_region_agnostic) + '/'
    else:
        raise NotImplementedError
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
    parser.add_argument('-smpler_train_seed', type=int, default=0)
    parser.add_argument('-planner_seed', type=int, default=0)
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-num_node_limit', type=int, default=3000)
    parser.add_argument('-num_train', type=int, default=5000)
    parser.add_argument('-sampler_seed', type=int, default=0)
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
    parser.add_argument('-hcount_number_in_goal', action='store_true', default=False)
    parser.add_argument('-qlearned_hcount', action='store_true', default=False)
    parser.add_argument('-qlearned_hcount_new_number_in_goal', action='store_true', default=False)
    parser.add_argument('-qlearned_hcount_old_number_in_goal', action='store_true', default=False)
    parser.add_argument('-qlearned_old_number_in_goal', action='store_true', default=False)
    parser.add_argument('-qlearned_new_number_in_goal', action='store_true', default=False)
    parser.add_argument('-integrated', action='store_true', default=False)
    parser.add_argument('-pure_learned_q', action='store_true', default=False)
    parser.add_argument('-state_hcount', action='store_true', default=False)
    parser.add_argument('-use_region_agnostic', action='store_true', default=False)
    parser.add_argument('-gather_planning_exp', action='store_true', default=False)
    parser.add_argument('-pidxs', nargs=2, type=int, default=[0, 1])  # used for threaded runs
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
            use_region_agnostic=config.use_region_agnostic
        )
        if config.domain == 'two_arm_mover':
            num_entities = 11
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


def get_learned_smpler(sampler_seed):
    print "Creating the learned sampler.."
    admon = create_imle_model(sampler_seed)
    print "Created IMLE model with weight name", admon.weight_file_name
    return admon


def make_pklable(plan):
    for p in plan:
        obj = p.discrete_parameters['object']
        region = p.discrete_parameters['region']
        if not isinstance(region, str):
            p.discrete_parameters['region'] = region.name
        if not (isinstance(obj, unicode) or isinstance(obj, str)):
            p.discrete_parameters['object'] = obj.GetName()


def main():
    config = parse_arguments()
    if config.gather_planning_exp:
        config.timelimit = np.inf

    np.random.seed(config.pidx)
    random.seed(config.pidx)

    problem_env = get_problem_env(config)
    set_problem_env_config(problem_env, config)

    if not config.hcount:
        pap_model = get_pap_gnn_model(problem_env, config)
    else:
        pap_model = None
    if config.integrated:
        smpler = get_learned_smpler(config.sampler_seed)
    else:
        smpler = None

    solution_file_name = get_solution_file_name(config)
    is_problem_solved_before = os.path.isfile(solution_file_name)
    plan_length = 0
    num_nodes = 0
    if is_problem_solved_before and not config.f:
        with open(solution_file_name, 'rb') as f:
            trajectory = pickle.load(f)
            success = trajectory['success']
            tottime = trajectory['tottime']
            num_nodes = trajectory['num_nodes']
    else:
        t = time.time()
        plan, num_nodes, nodes = search(problem_env, config, pap_model, smpler)
        tottime = time.time() - t
        success = plan is not None
        plan_length = len(plan) if success else 0
        if success and config.domain == 'one_arm_mover':
            make_pklable(plan)

        for n in nodes:
            n.state.make_pklable()

        nodes = None

        data = {
            'n_objs_pack': config.n_objs_pack,
            'tottime': tottime,
            'success': success,
            'plan_length': plan_length,
            'num_nodes': num_nodes,
            'plan': plan,
            'nodes': nodes
        }
        with open(solution_file_name, 'wb') as f:
            pickle.dump(data, f)
    print 'Time: %.2f Success: %d Plan length: %d Num nodes: %d' % (tottime, success, plan_length, num_nodes)


if __name__ == '__main__':
    main()
