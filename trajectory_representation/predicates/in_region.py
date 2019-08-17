from trajectory_representation.predicates.predicate import BinaryPredicate


class InRegion(BinaryPredicate):
    def __init__(self, problem_env):
        BinaryPredicate.__init__(self, problem_env)

    def __call__(self, a, b):

        is_a_region = a in self.problem_env.regions
        if is_a_region:
            return False

        is_b_region = b in self.problem_env.regions
        if is_b_region:
            b = self.problem_env.regions[b]
            if 'two_arm' in self.problem_env.name:
                is_a_in_b = b.contains(self.problem_env.env.GetKinBody(a).ComputeAABB())
            else:
                if 'rectangular_packing_box1' == b:
                    is_a_in_b = b.contains(self.problem_env.env.GetKinBody(a).ComputeAABB())
                else:
                    is_a_in_b = True
                """
                if 'top_shelf_region' in b.name:
                    is_a_in_b = b.contains(self.problem_env.env.GetKinBody(a).ComputeAABB())
                else:
                    if self.problem_env.regions['center_top_shelf_region'].contains(self.problem_env.env.GetKinBody(a).ComputeAABB()):
                        # if b is not top shelf region but top shelf contains a
                        is_a_in_b = False
                    else:
                        is_a_in_b = b.contains(self.problem_env.env.GetKinBody(a).ComputeAABB())
                """

            return is_a_in_b
        else:
            return False

