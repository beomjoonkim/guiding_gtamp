import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import multiprocessing
import argparse
import time


def worker_p(config):
    command = 'python ./data_processing/process_planning_experience.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -'+str(key)+' ' + str(value)
        command += option

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    time.sleep(1)
    return worker_p(multi_args)


def main():
    configs = []
    for pidx in range(10000):
        config = {'pidx': pidx}
        configs.append(config)

    n_workers = 1#multiprocessing.cpu_count()
    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
