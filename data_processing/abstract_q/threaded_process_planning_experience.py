import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import multiprocessing
import argparse
import time

from process_planning_experience import parse_parameters


def worker_p(config):
    command = 'python ./data_processing/abstract_q/process_planning_experience.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    time.sleep(1)
    return worker_p(multi_args)


def main():
    params = parse_parameters()
    param_vals = vars(params)
    configs = []
    pidxs = params.pidxs
    for pidx in range(pidxs[0], pidxs[1]):
        config = {}
        for key, value in zip(param_vals.keys(), param_vals.values()):
            if key != 'pidxs':
                print key
                config[key] = value
        config['pidx'] = pidx
        configs.append(config)

    n_workers = multiprocessing.cpu_count()
    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
