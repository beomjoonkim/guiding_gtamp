class AbstractRewardFunction:
    def __init__(self, problem_env, goal_objects, goal_region, planning_horizon):
        self.problem_env = problem_env
        self.robot = problem_env.robot
        self.goal_objects = goal_objects
        self.goal_region = goal_region
        self.infeasible_reward = -9999
        self.goal_reward = planning_horizon
        self.achieved = []
        self.planning_horizon = planning_horizon

    def __call__(self, curr_state, next_state, action, time_step):
        raise NotImplementedError

    def is_goal_reached(self):
        raise NotImplementedError


class GenericRewardFunction(AbstractRewardFunction):
    def __init__(self, problem_env, goal_objects, goal_region, planning_horizon):
        AbstractRewardFunction.__init__(self, problem_env, goal_objects, goal_region, planning_horizon)
        self.worst_reward = planning_horizon*-1

    def __call__(self, curr_state, next_state, action, time_step):
        if action.is_skeleton:
            return 0
        else:
            is_infeasible_action = next_state is None
            remaining_steps = self.planning_horizon - time_step
            if self.is_goal_reached():
                return self.goal_reward
            elif is_infeasible_action:
                return -1*remaining_steps
            else:
                return 0

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




