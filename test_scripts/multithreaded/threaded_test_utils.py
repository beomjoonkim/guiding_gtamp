import argparse


def parse_options():
    parser = argparse.ArgumentParser(description='Greedy Planner parameters')
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-planner_seed', type=int, default=0)
    parser.add_argument('-timelimit', type=int, default=1000)
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-pidxs', nargs=2, type=int, default=[0, 1])
    parser.add_argument('-num_train', type=int, default=5000)
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-train_seed', nargs=2, type=int, default=[0, 1])
    parser.add_argument('-use_shaped_reward', action='store_true', default=False)

    parameters = parser.parse_args()
    return parameters


def get_configs():
    parameters = parse_options()
    pidx_begin = parameters.pidxs[0]
    pidx_end = parameters.pidxs[1]
    train_seed_begin = parameters.train_seed[0]
    train_seed_end = parameters.train_seed[1]

    configs = []
    for train_seed in range(train_seed_begin, train_seed_end):
        for pidx in range(pidx_begin, pidx_end):
            config = {
                'pidx': pidx,
                'planner_seed': parameters.planner_seed,
                'train_seed': train_seed,
                'n_objs_pack': parameters.n_objs_pack,
                'timelimit': parameters.timelimit,
                'loss': parameters.loss,
                'num_train': parameters.num_train,
                'domain': parameters.domain,
            }
            if parameters.use_shaped_reward:
                config['use_shaped_reward'] = ""
            configs.append(config)

    return configs