import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import multiprocessing
import argparse
import time

from process_planning_experience import parse_parameters


def worker_p(config):
    command = 'python ./data_processing/sampler/process_planning_experience.py'

    for key, value in zip(config.keys(), config.values()):
        if key == 'f':
            continue
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

    for pidx in range(5000):
        config = {key: value for key, value in zip(param_vals.keys(), param_vals.values())}
        config['pidx'] = pidx
        configs.append(config)

    n_workers = multiprocessing.cpu_count()
    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
