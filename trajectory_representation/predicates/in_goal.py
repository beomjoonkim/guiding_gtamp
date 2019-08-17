from trajectory_representation.predicates.predicate import UnaryPredicate


class InGoal(UnaryPredicate):
    def __init__(self, problem_env):
        UnaryPredicate.__init__(self, problem_env)

    def __call__(self, entity):
        is_entity_region = entity in self.problem_env.regions
        if is_entity_region:
            return False
        return 0
