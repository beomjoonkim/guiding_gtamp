class AbstractRewardFunction:
    def __init__(self, problem_env, goal_objects, goal_region):
        self.problem_env = problem_env
        self.robot = problem_env.robot
        self.goal_objects = goal_objects
        self.goal_region = goal_region
        self.infeasible_reward = -2
        self.achieved = []

    def __call__(self, curr_state, next_state, action):
        raise NotImplementedError

    def is_goal_reached(self):
        raise NotImplementedError


class GenericRewardFunction(AbstractRewardFunction):
    def __init__(self, problem_env, goal_objects, goal_region):
        AbstractRewardFunction.__init__(self, problem_env, goal_objects, goal_region)

    def __call__(self, curr_state, next_state, action):
        if action.is_skeleton:
            return 0
        else:
            if self.is_goal_reached():
                return 1
            else:
                return -1

    def is_goal_reached(self):
        # todo udpate self.achieved?
        for obj in self.goal_objects:
            if not (obj, self.goal_region) in self.achieved:
                return False
        return True




