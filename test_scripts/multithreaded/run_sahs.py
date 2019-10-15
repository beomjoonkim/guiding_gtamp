import os

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from test_scripts.run_greedy import parse_arguments


def worker_p(config):
    command = 'python ./test_scripts/run_greedy.py'

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
    # configs = get_sahs_configs()
    params = parse_arguments()
    pidx_begin = params.pidxs[0]
    pidx_end = params.pidxs[1]
    params = vars(params)
    configs = []
    for pidx in range(pidx_begin, pidx_end):
        config = {}
        print pidx
        for key, value in zip(params.keys(), params.values()):
            if key == 'pidxs':
                continue

            if value is False:
                continue
            elif value is True:
                config[key] = ""
            else:
                config[key] = value
            config['pidx'] = pidx
        configs.append(config)
    n_workers = 1 #multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
