from trajectory_representation.predicates.predicate import NullaryPredicate


class IsHoldingGoalEntity(NullaryPredicate):
    def __init__(self, problem_env, goal_entities):
        NullaryPredicate.__init__(self, problem_env)
        self.goal_entities = goal_entities

    def __call__(self):
        if len(self.problem_env.robot.GetGrabbed()) == 0:
            return False
        else:
            held = self.problem_env.robot.GetGrabbed()[0].GetName()
            return held in self.goal_entities



