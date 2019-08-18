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
        if not isinstance(self.goal_region, str):
            assert False, "We only support single goal region"
        goal_region_name = self.goal_region
        goal_region = self.problem_env.regions[goal_region_name]
        goal_objects = [self.problem_env.env.GetKinBody(obj_name) for obj_name in self.goal_objects]
        for obj in goal_objects:
            if not goal_region.contains(obj.ComputeAABB()):
                return False
        return True




