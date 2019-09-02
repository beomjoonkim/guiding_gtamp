import argparse
import pickle
import time
import numpy as np
import socket
import random
import os

from gtamp_problem_environments.mover_env import PaPMoverEnv
from gtamp_problem_environments.one_arm_mover_env import OneArmMover
from planners.subplanners.motion_planner import BaseMotionPlanner
from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from trajectory_representation.trajectory import Trajectory
from planners.sahs.greedy import search


def get_problem_env(config):
    n_objs_pack = config.n_objs_pack
    if config.domain == 'two_arm_mover':
        problem_env = PaPMoverEnv(config.pidx)
        goal = ['home_region'] + [obj.GetName() for obj in problem_env.objects[:n_objs_pack]]
        problem_env.set_goal(goal)
    elif config.domain == 'one_arm_mover':
        problem_env = OneArmMover(config.pidx)
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
        solution_file_dir += '/no_reachable_regions_while_holding_state_computation_hcount/'
    elif config.state_hcount:
        solution_file_dir += '/state_hcount/'
    elif config.hadd:
        solution_file_dir += '/gnn_hadd/loss_' + str(config.loss) + '/num_train_' + str(config.num_train) + '/'
    else:
        solution_file_dir += '/gnn/loss_' + str(config.loss) + '/num_train_' + str(config.num_train) + '/'

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
    parser.add_argument('-num_train', type=int, default=5000)
    parser.add_argument('-timelimit', type=float, default=600)
    parser.add_argument('-visualize_plan', action='store_true', default=False)
    parser.add_argument('-visualize_sim', action='store_true', default=False)
    parser.add_argument('-dontsimulate', action='store_true', default=False)
    parser.add_argument('-plan', action='store_true', default=False)
    parser.add_argument('-dont_use_gnn', action='store_true', default=False)
    parser.add_argument('-dont_use_h', action='store_true', default=False)
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-problem_type', type=str, default='normal')  # supports normal, nonmonotonic
    parser.add_argument('-hcount', action='store_true', default=False)
    parser.add_argument('-hadd', action='store_true', default=False)
    parser.add_argument('-state_hcount', action='store_true', default=False)
    config = parser.parse_args()
    return config


def set_problem_env_config(problem_env, config):
    np.random.seed(config.planner_seed)
    random.seed(config.planner_seed)
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'prm'))
    problem_env.seed = config.pidx
    problem_env.init_saver = DynamicEnvironmentStateSaver(problem_env.env)


def main():
    config = parse_arguments()

    np.random.seed(config.pidx)
    random.seed(config.pidx)

    problem_env = get_problem_env(config)
    set_problem_env_config(problem_env, config)

    solution_file_name = get_solution_file_name(config)

    is_problem_solved_before = os.path.isfile(solution_file_name)
    if is_problem_solved_before and not config.plan:
        with open(solution_file_name, 'rb') as f:
            trajectory = pickle.load(f)
            success = trajectory.metrics['success']
            tottime = trajectory.metrics['tottime']
    else:
        t = time.time()
        trajectory, num_nodes = search(problem_env, config)
        tottime = time.time() - t
        success = trajectory is not None
        plan_length = len(trajectory.actions) if success else 0
        if not success:
            trajectory = Trajectory(config.pidx, config.pidx)
        trajectory.states = [s.get_predicate_evaluations() for s in trajectory.states]
        trajectory.state_prime = None

        trajectory.metrics = {
            'n_objs_pack': config.n_objs_pack,
            'tottime': tottime,
            'success': success,
            'plan_length': plan_length,
            'num_nodes': num_nodes,
        }

        with open(solution_file_name, 'wb') as f:
            pickle.dump(trajectory, f)
    print 'Time: %.2f Success: %d Plan length: %d Num nodes: %d' % (tottime, success, trajectory.metrics['plan_length'],
                                                                    trajectory.metrics['num_nodes'])


if __name__ == '__main__':
    main()
