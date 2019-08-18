from gtamp_problem_environments.reward_functions.reward_function import GenericRewardFunction
from planners.heuristics import compute_hcount


class ShapedRewardFunction(GenericRewardFunction):
    def __init__(self, problem_env, goal_objects, goal_region):
        GenericRewardFunction.__init__(self, problem_env, goal_objects, goal_region)
        self.potential_function = lambda state: compute_hcount(state, self.problem_env)

    def __call__(self, curr_state, next_state, action):
        if action.is_skeleton:
            return 0
        else:
            if self.is_goal_reached():
                return 10
            elif next_state is None:
                return self.infeasible_reward
            else:
                #state = PaPState(problem_env, goal_entities, parent_state, parent_action, paps_used)
                true_reward = GenericRewardFunction.__call__(self, curr_state, next_state, action)
                if curr_state is None:
                    return true_reward
                else:
                    potential_curr = self.potential_function(curr_state)
                    potential_next = self.potential_function(next_state)

                    print potential_curr, potential_next
                    shaping_val = (potential_next - potential_curr)
                    return true_reward + shaping_val


