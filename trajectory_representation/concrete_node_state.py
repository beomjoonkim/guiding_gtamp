from gtamp_utils import utils
import numpy as np
import pickle


class ConcreteNodeState:
    def __init__(self, problem_env, obj, region, goal_entities, key_configs=None, collision_vector=None):
        self.obj = obj
        self.region = region
        self.goal_entities = goal_entities
        self.problem_env = problem_env

        if collision_vector is None:
            self.key_configs = key_configs
            self.collision_vector = utils.compute_occ_vec(self.key_configs)
        else:
            self.collision_vector = collision_vector
            self.key_configs = pickle.load(open('prm.pkl', 'r'))[0]
            self.key_configs = np.delete(self.key_configs, [415, 586, 615, 618, 619], axis=0)

        self.collision_vector = utils.convert_binary_vec_to_one_hot(self.collision_vector)
        self.collision_vector = self.collision_vector.reshape((1, len(self.collision_vector), 2, 1))

        if type(obj) == str or type(obj) == unicode:
            obj = problem_env.env.GetKinBody(obj)

        self.abs_robot_pose = utils.get_body_xytheta(self.problem_env.robot)
        self.abs_obj_pose = utils.get_body_xytheta(obj)

        n_key_configs = len(self.collision_vector)
        is_goal_obj = utils.convert_binary_vec_to_one_hot(np.array([obj in self.goal_entities]))
        is_goal_obj = np.tile(is_goal_obj, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
        is_goal_region = utils.convert_binary_vec_to_one_hot(
            np.array([self.region in self.goal_entities]))
        is_goal_region = np.tile(is_goal_region, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
        self.goal_flags = np.concatenate([is_goal_obj, is_goal_region], axis=2)
