import argparse


def parse_options():
    parser = argparse.ArgumentParser(description='Greedy Planner parameters')
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-planner_seed', type=int, default=0)
    parser.add_argument('-timelimit', type=int, default=300)
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-pidxs', nargs=2, type=int, default=[0, 1])
    parser.add_argument('-num_train', type=int, default=7000)
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    #parser.add_argument('-train_seed', nargs=2, type=int, default=[0, 1])
    parser.add_argument('-train_seed', type=int, default=0)
    parser.add_argument('-sampling_strategy', type=str, default='uniform')
    parser.add_argument('-hcount', action='store_true', default=False)
    parser.add_argument('-state_hcount', action='store_true', default=False)
    parser.add_argument('-explr_p', type=float, default=0.3)  # number of re-evals
    parser.add_argument('-planner', type=str, default='mcts_with_leaf_strategy')
    parser.add_argument('-mixrate', type=float, default=1)
    parser.add_argument('-use_region_agnostic', action='store_true', default=False)

    # MCTS
    parser.add_argument('-use_shaped_reward', action='store_true', default=False)
    parser.add_argument('-qlearned_hcount', action='store_true', default=False)
    parser.add_argument('-widening_parameter', type=float, default=50)  # number of re-evals
    parser.add_argument('-ucb_parameter', type=float, default=0.1)
    parser.add_argument('-switch_frequency', type=int, default=50)

    parser.add_argument('-mse_weight', type=float, default=1)
    parameters = parser.parse_args()
    return parameters


def get_configs():
    parameters = parse_options()
    pidx_begin = parameters.pidxs[0]
    pidx_end = parameters.pidxs[1]

    configs = []
    for pidx in range(pidx_begin, pidx_end):
        config = {
            'pidx': pidx,
            'planner_seed': parameters.planner_seed,
            'n_objs_pack': parameters.n_objs_pack,
            'timelimit': parameters.timelimit,
            'domain': parameters.domain,
        }

        configs.append(config)
    return parameters, configs


def get_sahs_configs():
    parameters, configs = get_configs()

    for config in configs:
        if parameters.hcount:
            config['hcount'] = ""
        elif parameters.state_hcount:
            config['state_hcount'] = ""
        elif parameters.qlearned_hcount:
            config['qlearned_hcount'] = ""

        if parameters.use_region_agnostic:
            config['use_region_agnostic'] = ""

        config['loss'] = parameters.loss
        config['train_seed'] = parameters.train_seed
        config['num_train'] = parameters.num_train
        config['mixrate'] = parameters.mixrate
        config['mse_weight'] = parameters.mse_weight
    return configs


def get_mcts_configs():
    parameters, configs = get_configs()

    for config in configs:
        config['explr_p'] = parameters.explr_p
        if parameters.use_shaped_reward:
            config['use_shaped_reward'] = ""

        config['ucb_parameter'] = parameters.ucb_parameter
        config['sampling_strategy'] = parameters.sampling_strategy
        config['loss'] = parameters.loss
        config['train_seed'] = parameters.train_seed
        config['num_train'] = parameters.num_train
        config['widening_parameter'] = parameters.widening_parameter
        config['planner'] = parameters.planner
        config['switch_frequency'] = parameters.switch_frequency
    return configs


