import os
import multiprocessing
import argparse

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from threaded_test_utils import get_sahs_configs


def worker_p(config):
    command = 'python ./test_scripts/run_greedy.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option
    command += ' -qlearned_hcount_old_number_in_goal'
    command += ' -gather_planning_exp'

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    parser = argparse.ArgumentParser(description='Greedy Planner parameters')
    parser.add_argument('-pidxs', nargs=2, type=int, default=[0, 1])
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parameters = parser.parse_args()
    pidx_begin = parameters.pidxs[0]
    pidx_end = parameters.pidxs[1]
    configs = []
    for pidx in range(pidx_begin, pidx_end):
        config = {
            'pidx': pidx,
            'domain': parameters.domain,
        }

        configs.append(config)

    n_workers = 10 #multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
