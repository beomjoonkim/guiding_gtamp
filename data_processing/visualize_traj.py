import pickle


def main():
    result_dir = './test_results/mcts_results_on_mover_domain/widening_5/uct_1.0/trajectory_data/'
    traj_file = 'traj_pidx_1.pkl'
    traj = pickle.load(open(result_dir+traj_file, 'r'))
    traj.visualize()


if __name__ == "__main__":
    main()