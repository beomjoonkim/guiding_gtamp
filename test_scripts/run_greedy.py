from planners.sahs.greedy import generate_training_data_single

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Greedy planner')
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-train_seed', type=int, default=0)
    parser.add_argument('-planner_seed', type=int, default=0)
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-num_train', type=int, default=5000)
    parser.add_argument('-timelimit', type=float, default=600)
    parser.add_argument('-visualize_plan', action='store_true', default=False)
    parser.add_argument('-visualize_sim', action='store_true', default=False)
    parser.add_argument('-dontsimulate', action='store_true', default=False)
    parser.add_argument('-plan', action='store_true', default=False)
    parser.add_argument('-dont_use_gnn', action='store_true', default=False)
    parser.add_argument('-dont_use_h', action='store_true', default=False)
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-problem_type', type=str, default='normal')  # supports normal, nonmonotonic
    parser.add_argument('-hcount', action='store_true', default=False)
    parser.add_argument('-hadd', action='store_true', default=False)
    parser.add_argument('-state_hcount', action='store_true', default=False)

    config = parser.parse_args()
    generate_training_data_single(config)

