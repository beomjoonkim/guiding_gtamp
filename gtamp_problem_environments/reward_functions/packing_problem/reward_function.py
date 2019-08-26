from gtamp_problem_environments.reward_functions.reward_function import GenericRewardFunction
from planners.heuristics import compute_hcount


class ShapedRewardFunction(GenericRewardFunction):
    def __init__(self, problem_env, goal_objects, goal_region, planning_horizon):
        GenericRewardFunction.__init__(self, problem_env, goal_objects, goal_region, planning_horizon)
        self.potential_function = lambda state: -compute_hcount(state, self.problem_env)
        # potential_function is minus of the number of objects to move (smaller the n_objs_to_move, the better)

    def __call__(self, curr_state, next_state, action, time_step):
        if action.is_skeleton:
            return 0
        else:
            if self.is_goal_reached():
                return 10
            elif next_state is None:
                true_reward = GenericRewardFunction.__call__(self, curr_state, next_state, action, time_step)
                return true_reward
            else:
                true_reward = GenericRewardFunction.__call__(self, curr_state, next_state, action, time_step)

                potential_curr = self.potential_function(curr_state)
                potential_next = self.potential_function(next_state)

                print potential_curr, potential_next
                shaping_val = (potential_next - potential_curr)
                return true_reward + shaping_val


