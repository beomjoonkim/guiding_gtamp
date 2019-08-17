from trajectory_representation.predicates.predicate import UnaryPredicate
from trajectory_representation.swept_volume import PickAndPlaceSweptVolume


class BlocksKeyConfigs(UnaryPredicate):
    def __init__(self, problem_env, target_pap):
        UnaryPredicate.__init__(self, problem_env)
        self.target_pap = target_pap
        self.swept_volume = PickAndPlaceSweptVolume(self.problem_env)
        self.swept_volume.add_swept_volume(target_pap[0], target_pap[1])

    def __call__(self, entity, goal):
        is_obj = entity not in self.problem_env.regions
        if not is_obj:
            return False

        entity = self.problem_env.env.GetKinBody(entity)
        is_col = self.swept_volume.is_obj_in_collision_with_given_place_volume(self.target_pap[1].low_level_motion,
                                                                               entity,
                                                                               self.target_pap[0])
        return is_col
