from generators.learning.utils.data_processing_utils import state_data_mode, action_data_mode

from generators.learning.RelKonfIMLE import RelKonfIMLEPose

import collections


def load_weights(policy, seed, weight_fname=None):
    if weight_fname is None:
        weight_fname = 'imle_pose_seed_%d.h5' % seed
    else:
        assert seed is None, 'Seed is not supposed to be used'

    policy.policy_model.load_weights(policy.save_folder + weight_fname)



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
    dim_state = (n_key_configs, 2, 1)
    policy = RelKonfIMLEPose(dim_action, dim_state, savedir, 1.0, config)
    load_weights(policy, seed)

    return policy
