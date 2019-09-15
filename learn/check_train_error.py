def main():
    problem_env, openrave_env = self.create_environment()
    motion_planner = BaseMotionPlanner(problem_env, 'prm')
    problem_env.set_motion_planner(motion_planner)
    parent_state = None
    parent_action = None
    state = self.compute_state(parent_state, parent_action, goal_entities, problem_env, curr_paps_used, 0)

if __name__ == '__main__':
    main()