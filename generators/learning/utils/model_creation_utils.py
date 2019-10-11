from generators.learning.utils.data_processing_utils import state_data_mode, action_data_mode

from generators.learning.RelKonfIMLE import RelKonfIMLEPose

import collections


def create_imle_model(seed):
    n_key_configs = 615  # indicating whether it is a goal obj and goal region
    savedir = 'generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/rel_konf_place_admon/' % (
        state_data_mode, action_data_mode)

    mconfig_type = collections.namedtuple('mconfig_type',
                                          'tau seed')

    config = mconfig_type(
        tau=1.0,
        seed=seed
    )

    dim_action = 4
    fname = 'imle_pose_seed_%d.h5' % config.seed
    dim_state = (n_key_configs, 2, 1)
    policy = RelKonfIMLEPose(dim_action, dim_state, savedir, 1.0, config)
    policy.policy_model.load_weights(policy.save_folder + fname)
    return policy
