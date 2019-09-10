import os
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from threaded_test_utils import get_sahs_configs


def worker_p(config):
    command = 'python ./test_scripts/gather_planning_experience.py'

    for key, value in zip(config.keys(), config.values()):
        if 'sampling_strategy' in key:
            continue
        option = ' -' + str(key) + ' ' + str(value)
        command += option

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    configs = get_sahs_configs()
    n_workers = 10
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()