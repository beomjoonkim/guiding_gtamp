import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse
import time
import multiprocessing


def worker_p(config):
    command = 'python -m learn.train -loss mse'

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
    seed = sys.argv[1]
    data_range = [50, 100] + range(1000, 6000, 1000)
    for n_data in data_range:
        config = {'num_train': n_data, 'seed': seed}
        configs.append(config)

    n_workers = 1
    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)



if __name__ == '__main__':
    main()
