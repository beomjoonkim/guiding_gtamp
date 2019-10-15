from gtamp_problem_environments.mover_env import Mover
from gtamp_utils import utils
from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.learned_generator import LearnedGenerator
from generators.learning.AdMon import AdversarialMonteCarlo
from generators.learning.AdMonWithPose import AdversarialMonteCarloWithPose

import numpy as np
import random
import pickle
import sys


def get_learned_smpler(algo):
    n_key_configs = 620
    dim_state = (n_key_configs, 2, 1)
    dim_action = 8
    weight_folder = './generators/learning/learned_weights/'
    if algo == 'admon':
        model = AdversarialMonteCarlo(dim_action=dim_action, dim_state=dim_state,
                                      save_folder=weight_folder,
                                      tau=1.0,
                                      explr_const=0.0)
    elif algo == 'admonpose':
        model = AdversarialMonteCarloWithPose(dim_action=dim_action, dim_collision=dim_state,
                                              save_folder=weight_folder,
                                              tau=1.0,
                                              explr_const=0.0)

    else:
        raise NotImplementedError
    model.load_weights(agen_file='a_gen_epoch_10.h5')
    return model


def get_pick_base_poses(action, smples):
    pick_base_poses = []
    for smpl in smples:
        smpl = smpl[0:4]
        sin_cos_encoding = smpl[-2:]
        decoded_angle = utils.decode_sin_and_cos_to_angle(sin_cos_encoding)
        smpl = np.hstack([smpl[0:2], decoded_angle])
        abs_base_pose = utils.get_absolute_pick_base_pose_from_ir_parameters(smpl, action.discrete_parameters['object'])
        pick_base_poses.append(abs_base_pose)
    return pick_base_poses


def get_place_base_poses(action, smples, mover):
    place_base_poses = smples[:, 4:]
    to_return = []
    for bsmpl in place_base_poses:
        sin_cos_encoding = bsmpl[-2:]
        decoded_angle = utils.decode_sin_and_cos_to_angle(sin_cos_encoding)
        bsmpl = np.hstack([bsmpl[0:2], decoded_angle])
        to_return.append(bsmpl)
    to_return = np.array(to_return)
    to_return[:, 0:2] += mover.regions[action.discrete_parameters['region']].box[0]
    return to_return


def compute_state(obj, region, problem_env, key_configs):
    # todo the state of a concrete node consists of the object, region, and the collision vector.
    goal_entities = ['square_packing_box1', 'home_region']
    return ConcreteNodeState(problem_env, obj, region, goal_entities, key_configs)


def create_environment(problem_idx):
    problem_env = Mover(problem_idx)
    openrave_env = problem_env.env
    return problem_env, openrave_env


def visualize(plan, problem_idx, algo):
    np.random.seed(problem_idx)
    random.seed(problem_idx)

    problem_env, openrave_env = create_environment(problem_idx)
    key_configs = pickle.load(open('prm.pkl', 'r'))[0]

    state = None
    learned_smpler = get_learned_smpler(algo)
    utils.viewer()
    for action_idx, action in enumerate(plan):
        if 'pick' in action.type:
            associated_place = plan[action_idx + 1]
            state = compute_state(action.discrete_parameters['object'],
                                  associated_place.discrete_parameters['region'],
                                  problem_env,
                                  key_configs)
            smpler = LearnedGenerator(action, problem_env, learned_smpler, state)
            smples = np.vstack([smpler.generate() for _ in range(10)])
            action.discrete_parameters['region'] = associated_place.discrete_parameters['region']
            pick_base_poses = get_pick_base_poses(action, smples)
            place_base_poses = get_place_base_poses(action, smples, problem_env)
            utils.visualize_path(place_base_poses)
            import pdb;pdb.set_trace()

        action.execute()


def main():
    pidx = int(sys.argv[1])
    algo = str(sys.argv[2])
    filename = './planning_experience/raw/two_arm_mover/n_objs_pack_1//' + 'seed_0_pidx_' + str(pidx) + '.pkl'
    plan_data = pickle.load(open(filename, 'r'))
    plan = plan_data['plan']
    visualize(plan, pidx, algo)


if __name__ == '__main__':
    main()
