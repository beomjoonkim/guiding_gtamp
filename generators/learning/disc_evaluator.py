from CMAESAdMonWithPose import CMAESAdversarialMonteCarloWithPose
from policy_evaluator import *


def get_augmented_state_vec_and_poses(obj, state_vec, smpler_state):
    n_key_configs = 615
    utils.set_color(obj, [1, 0, 0])
    is_goal_obj = utils.convert_binary_vec_to_one_hot(np.array([obj in smpler_state.goal_entities]))
    is_goal_obj = np.tile(is_goal_obj, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
    is_goal_region = utils.convert_binary_vec_to_one_hot(np.array([smpler_state.region in smpler_state.goal_entities]))
    is_goal_region = np.tile(is_goal_region, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
    state_vec = np.concatenate([state_vec, is_goal_obj, is_goal_region], axis=2)

    poses = np.hstack(
        [utils.encode_pose_with_sin_and_cos_angle(utils.get_body_xytheta(obj).squeeze()), 0, 0, 0, 0]).reshape((1, 8))
    return state_vec, poses


def get_placements(state, poses, admon):
    placement = admon.get_max_x(state[None, :], poses[None, :])
    return [placement]


def visualize_samples(q_fcn):
    n_evals = 10
    pidxs = get_pidxs_to_evaluate_policy(n_evals)
    pidx = pidxs[5]
    config_type = collections.namedtuple('config', 'n_objs_pack pidx domain ')
    config = config_type(pidx=pidx, n_objs_pack=1, domain='two_arm_mover')
    problem_env = get_problem_env(config)
    pidx_poses = load_pose_file(pidx)

    problem_env.set_body_poses(pidx_poses)
    smpler_state = get_smpler_state(pidx)
    state_vec = np.delete(smpler_state.state_vec, [415, 586, 615, 618, 619], axis=1)

    obj = 'square_packing_box1'
    state_vec, poses = get_augmented_state_vec_and_poses(obj, state_vec, smpler_state)

    places = get_placements(state_vec, poses, q_fcn)
    utils.viewer()
    utils.visualize_path(places)
    import pdb;
    pdb.set_trace()


def main():
    n_key_configs = 615  # indicating whether it is a goal obj and goal region
    savedir = 'generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/cmaes_place_admon/' % (
        state_data_mode, action_data_mode)

    mconfig_type = collections.namedtuple('mconfig_type',
                                          'tau seed')

    config = mconfig_type(
        tau=1.0,
        seed=int(sys.argv[1])
    )
    epoch_number = int(sys.argv[2])
    dim_state = (n_key_configs, 6, 1)
    dim_action = 4
    policy = CMAESAdversarialMonteCarloWithPose(dim_action=dim_action, dim_collision=dim_state,
                                                save_folder=savedir, tau=1.0, config=config)

    print "Trying epoch number ", epoch_number
    fname = 'admonpose_seed_1_epoch_214_batch_idx_2464_drop_in_mse_-10.42228.h5'
    policy.a_gen.load_weights(policy.save_folder + fname)
    visualize_samples(policy)


if __name__ == '__main__':
    main()
