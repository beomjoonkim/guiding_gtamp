import argparse
import os
import pickle


def load_data(traj_dir):
    traj_files = os.listdir(traj_dir)
    import pdb;pdb.set_trace()
    for traj_file in traj_files:
        try:
            traj = pickle.load(open(traj_dir+traj_file, 'r'))
        except:
            continue
        import pdb;pdb.set_trace()

    return 1


def train_admon(args):
    # Loads the processed data
    states, actions, sum_rewards = load_data('./planning_experience/processed/domain_two_arm_mover/'
                                             'n_objs_pack_1/irsc/sampler_trajectory_data/')
    import pdb;pdb.set_trace()


    # Loads the state, action, and reward tuples

    # Compute the sum of rewards

    # Run AdMon
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='Process configurations')
    parser.add_argument('-n_data', type=int, default=100)
    parser.add_argument('-a', default='ddpg')
    parser.add_argument('-g', action='store_true')
    parser.add_argument('-n_trial', type=int, default=-1)
    parser.add_argument('-i', type=int, default=0)
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-tau', type=float, default=1e-3)
    parser.add_argument('-d_lr', type=float, default=1e-3)
    parser.add_argument('-g_lr', type=float, default=1e-4)
    parser.add_argument('-n_score', type=int, default=5)
    parser.add_argument('-otherpi', default='uniform')
    parser.add_argument('-explr_p', type=float, default=0.3)
    parser.add_argument('-domain', type=str, default='convbelt')
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_admon(args)

if __name__ == '__main__':
    main()
