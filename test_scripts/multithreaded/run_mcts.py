import os
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from threaded_test_utils import get_mcts_configs


def worker_p(config):
    command = 'python ./test_scripts/run_mcts.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    configs = get_mcts_configs()

    n_workers = 1
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()