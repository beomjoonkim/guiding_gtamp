from gtamp_utils import utils
import numpy as np
import pickle


class ConcreteNodeState:
    def __init__(self, problem_env, obj, region, goal_entities, key_configs=None, collision_vector=None):
        self.obj = obj
        self.region = region
        self.goal_entities = goal_entities

        self.key_configs = self.get_key_configs(key_configs)
        self.collision_vector = self.get_collison_vector(collision_vector)

        if type(obj) == str or type(obj) == unicode:
            obj = problem_env.env.GetKinBody(obj)

        self.abs_robot_pose = utils.clean_pose_data(utils.get_body_xytheta(problem_env.robot))
        self.abs_obj_pose = utils.clean_pose_data(utils.get_body_xytheta(obj))
        self.abs_goal_obj_pose = utils.clean_pose_data(utils.get_body_xytheta('square_packing_box1'))
        self.goal_flags = self.get_goal_flags()
        self.rel_konfs = None

    def get_goal_flags(self):
        n_key_configs = self.collision_vector.shape[1]
        is_goal_obj = utils.convert_binary_vec_to_one_hot(np.array([self.obj in self.goal_entities]))
        is_goal_obj = np.tile(is_goal_obj, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
        is_goal_region = utils.convert_binary_vec_to_one_hot(
            np.array([self.region in self.goal_entities]))
        is_goal_region = np.tile(is_goal_region, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
        return np.concatenate([is_goal_obj, is_goal_region], axis=2)

    def get_key_configs(self, given_konfs):
        if given_konfs is None:
            key_configs = pickle.load(open('prm.pkl', 'r'))[0]
            key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
        else:
            key_configs = given_konfs
        return key_configs

    def get_collison_vector(self, given_collision_vector):
        if given_collision_vector is None:
            collision_vector = utils.compute_occ_vec(self.key_configs)
        else:
            collision_vector = given_collision_vector
        collision_vector = utils.convert_binary_vec_to_one_hot(collision_vector)
        collision_vector = collision_vector.reshape((1, len(collision_vector), 2, 1))
        return collision_vector
