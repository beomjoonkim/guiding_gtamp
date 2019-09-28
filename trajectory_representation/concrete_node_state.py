from gtamp_utils import utils
import numpy as np


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
            self.key_configs = None

        self.one_hot = utils.convert_binary_vec_to_one_hot(self.collision_vector)
        is_goal_obj = utils.convert_binary_vec_to_one_hot(np.array([obj in goal_entities]))
        is_goal_region = utils.convert_binary_vec_to_one_hot(np.array([region in goal_entities]))

        self.state_vec = np.vstack([self.one_hot, is_goal_obj, is_goal_region])
        self.state_vec = self.state_vec.reshape((1, len(self.state_vec), 2, 1))

        if type(obj) == str or type(obj) == unicode:
            obj = problem_env.env.GetKinBody(obj)
        #robot_xy_wrt_o = np.dot(np.linalg.inv(obj.GetTransform()), problem_env.robot.GetTransform())[:-2, 3]
        #self.robot_wrt_obj = robot_xy_wrt_o.reshape((1, 2))
        #self.obj_pose = utils.get_body_xytheta(obj)[0, 0:2].reshape((1, 2))

        self.robot_pose = utils.get_body_xytheta(self.problem_env.robot)
        self.obj_pose = utils.get_body_xytheta(obj)



