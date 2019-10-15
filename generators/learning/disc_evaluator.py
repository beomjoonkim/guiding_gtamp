from CMAESAdMonWithPose import CMAESAdversarialMonteCarloWithPose
from RelKonfAdMonWithPose import RelKonfMSEPose
from policy_evaluator import get_pidxs_to_evaluate_policy, load_pose_file, get_smpler_state
from data_processing.utils import state_data_mode, action_data_mode, convert_pose_rel_to_region_to_abs_pose, \
    unnormalize_pose_wrt_region

from gtamp_utils import utils
from gtamp_utils.utils import *
from test_scripts.run_greedy import get_problem_env

import collections
import pickle


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


def make_body(height, i, x, y, color=[0,0.5,0]):
    env = openravepy.RaveGetEnvironments()[0]
    new_body = box_body(env, 0.1, 0.1, height,
                        name='value_obj%s' % i,
                        color=color)
    env.AddKinBody(new_body)
    trans = np.eye(4)
    trans[2, -1] = 1.0
    trans[0, -1] = x
    trans[1, -1] = y
    new_body.SetTransform(trans)
    utils.set_body_transparency(new_body, 0.5)


def get_placements(state, poses, admon, smpler_state):
    stime = time.time()
    # placement, value = admon.get_max_x(state, poses)
    max_x = None
    max_val = -np.inf
    exp_val = {}

    cols = state[:, :, 0:2, :]
    goal_flags = state[:, :, 2:, :]

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
    rel_konfs = []
    for k in key_configs:
        konf = utils.clean_pose_data(k)
        obj_pose = utils.clean_pose_data(smpler_state.obj_pose)
        rel_konf = utils.subtract_pose2_from_pose1(konf, obj_pose)
        rel_konfs.append(rel_konf)
    rel_konfs = np.array(rel_konfs).reshape((1, 615, 3, 1))

    x_range = np.linspace(0., 3.5, 10)
    y_range = np.linspace(-8., -6., 10)
    placement = np.array([0., 0., 0.])
    for x in x_range:
        for y in y_range:
            placement = placement.squeeze()
            placement[0] = x
            placement[1] = y
            placement = utils.clean_pose_data(placement)
            obj_pose = utils.clean_pose_data(smpler_state.obj_pose)
            rel_placement = utils.subtract_pose2_from_pose1(placement, obj_pose)
            obj_pose = utils.encode_pose_with_sin_and_cos_angle(obj_pose)
            val = admon.q_mse_model.predict([rel_placement[None, :], goal_flags, obj_pose[None, :], rel_konfs, cols])
            #print np.mean(admon.relevance_model.predict([rel_placement[None, :], goal_flags, rel_konfs, cols]).squeeze())
            if val > max_val:
                max_x = copy.deepcopy(placement)
                max_val = val
            exp_val[(x, y)] = np.exp(val) * 100
            print rel_placement, x, y, val, exp_val[(x, y)]
    total = np.sum(exp_val.values())
    total = 1
    i = 0
    utils.viewer()

    for x in x_range:
        for y in y_range:
            height = exp_val[(x, y)]  # / total + 1
            # print x, y, height
            placement = placement.squeeze()
            placement[0] = x
            placement[1] = y
            # absx, absy = unnormalize_pose_wrt_region(placement, 'loading_region')[0:2]
            # make_body(height, i, absx, absy)

            if height == np.exp(max_val) * 100:
                make_body(height, i, x, y, color=[1, 0, 0])
            else:
                make_body(height, i, x, y)
            i += 1
    placement = max_x
    print placement, max_val, np.exp(max_val)
    print 'maximizing x time', time.time() - stime


def visualize_samples(q_fcn):
    n_evals = 10
    pidxs = get_pidxs_to_evaluate_policy(n_evals)
    pidx = pidxs[2]
    config_type = collections.namedtuple('config', 'n_objs_pack pidx domain ')
    config = config_type(pidx=pidx, n_objs_pack=1, domain='two_arm_mover')
    problem_env = get_problem_env(config)
    pidx_poses = load_pose_file(pidx)

    problem_env.set_body_poses(pidx_poses)
    smpler_state = get_smpler_state(pidx)
    state_vec = np.delete(smpler_state.state_vec, [415, 586, 615, 618, 619], axis=1)

    obj = 'square_packing_box3'
    state_vec, poses = get_augmented_state_vec_and_poses(obj, state_vec, smpler_state)
    get_placements(state_vec, poses, q_fcn, smpler_state)


def main():
    n_key_configs = 615  # indicating whether it is a goal obj and goal region
    savedir = 'generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/rel_konf_place_mse/' % (
        state_data_mode, action_data_mode)

    mconfig_type = collections.namedtuple('mconfig_type',
                                          'tau seed')

    config = mconfig_type(
        tau=1.0,
        seed=int(sys.argv[1])
    )

    use_rel_konf = True
    dim_action = 3
    fname = 'pretrained_%d.h5' % config.seed
    if use_rel_konf:
        dim_state = (n_key_configs, 2, 1)
        policy = RelKonfMSEPose(dim_action, dim_state, savedir, 1.0, config)
        policy.q_mse_model.load_weights(policy.save_folder + fname)
    else:
        dim_state = (n_key_configs, 6, 1)
        policy = CMAESAdversarialMonteCarloWithPose(dim_action=dim_action, dim_collision=dim_state,
                                                    save_folder=savedir, tau=1.0, config=config)
        policy.disc.load_weights(policy.save_folder + fname)
    visualize_samples(policy)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
