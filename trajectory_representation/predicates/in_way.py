import numpy as np


class InWay:
    def __init__(self, problem_env, collides=None):
        self.problem_env = problem_env
        self.robot = self.problem_env.robot
        self.reachable_entities = None
        self.pick_used = {}
        self.reachable_entities = []
        self.paths_to_reachable_entities = None

        self.mc_to_entity = {}
        self.mc_path_to_entity = {}
        self.sampled_pick_configs_for_objects = {}

        self.evaluations = {}
        self.collides = collides
        self.place_used = {}
        self.mc_calls = 0

    def set_pick_used(self, pick_used):
        if pick_used is None:
            self.pick_used = {}
        else:
            self.pick_used = pick_used

    def set_place_used(self, place_used):
        if place_used is None:
            self.place_used = {}
        else:
            self.place_used = place_used

    def set_reachable_entities(self, r):
        self.reachable_entities = r

    def set_paths_to_reachable_entities(self, r):
        self.paths_to_reachable_entities = r

    @staticmethod
    def get_goal_config_used(motion_plan, potential_goal_configs):
        which_goal = np.argmin(np.linalg.norm(motion_plan[-1] - potential_goal_configs, axis=-1))
        return potential_goal_configs[which_goal]

