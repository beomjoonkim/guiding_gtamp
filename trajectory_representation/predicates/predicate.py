class Predicate:
    def __init__(self, problem_env):
        self.problem_env = problem_env


class NullaryPredicate(Predicate):
    def __init__(self, problem_env):
        Predicate.__init__(self, problem_env)

    def __call__(self):
        raise NotImplementedError


class UnaryPredicate(Predicate):
    def __init__(self, problem_env):
        Predicate.__init__(self, problem_env)

    def __call__(self, a):
        raise NotImplementedError


class BinaryPredicate(Predicate):
    def __init__(self, problem_env):
        Predicate.__init__(self, problem_env)

    def __call__(self, a, b):
        # why is goal necessary?
        raise NotImplementedError


class TernaryPredicate(Predicate):
    def __init__(self, problem_env):
        Predicate.__init__(self, problem_env)

    def __call__(self, a, b, c):
        # why is goal necessary?
        raise NotImplementedError







