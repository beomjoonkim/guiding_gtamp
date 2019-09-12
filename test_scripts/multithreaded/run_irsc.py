import os
import sys
import multiprocessing

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from threaded_test_utils import get_configs
import argparse
import time


def worker_p(config):
    command = 'python ./test_scripts/run_irsc.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    _, configs = get_configs()
    n_workers = multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()