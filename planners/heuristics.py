import Queue


def compute_hcount(state, problem_env):
    objects_to_move = set()
    potential_obj_to_move_queue = Queue.Queue()
    goal_r = [entity for entity in state.goal_entities if 'region' in entity][0]

    # Putting goal objects that are not in the goal region to objects_to_move set
    for entity in state.goal_entities:
        is_obj_entity = 'region' not in entity
        if is_obj_entity:
            goal_obj_body = problem_env.env.GetKinBody(entity)
            is_goal_region_contains_entity = problem_env.regions[goal_r].contains(goal_obj_body.ComputeAABB())
            if not is_goal_region_contains_entity:
                potential_obj_to_move_queue.put(entity)

    object_names = [o for o in problem_env.entity_names if 'region' not in o]
    n_occludes_pre = 0
    n_occludes_manip = 0
    n_occludes = 0
    while not potential_obj_to_move_queue.empty():
        obj_to_move = potential_obj_to_move_queue.get()
        if obj_to_move not in objects_to_move:
            objects_to_move.add(obj_to_move)
            for o2 in object_names:
                # OccludesPre
                is_o2_in_way_of_obj_to_move = state.binary_edges[(o2, obj_to_move)][1]

                if is_o2_in_way_of_obj_to_move:
                    n_occludes_pre += 1

                goal_region = 'home_region'
                original_obj_region = 'loading_region'
                is_o2_in_way_of_obj_to_move_to_region = False
                if obj_to_move in state.goal_entities:
                    is_o2_in_way_of_obj_to_move_to_region = state.ternary_edges[(obj_to_move, o2, goal_region)][0]
                else:
                    pass

                if is_o2_in_way_of_obj_to_move_to_region:
                    n_occludes_manip += 1

                if is_o2_in_way_of_obj_to_move or is_o2_in_way_of_obj_to_move_to_region:
                    n_occludes += 1
                    print o2
                    potential_obj_to_move_queue.put(o2)

    print "n occludes pre %d n occludes manip %d n_occludes %d" % (n_occludes_pre, n_occludes_manip, n_occludes)
    print objects_to_move
    #if len(objects_to_move) == 0 or (n_occludes_pre == 0 and n_occludes_manip == 0 and n_occludes == 0):
    #    import pdb;
    #    pdb.set_trace()
    # import pdb;pdb.set_trace()

    """
    # Count the objects that need to be moved recursively
    object_names = [o for o in problem_env.entity_names if 'region' not in o]
    while not potential_obj_to_move_queue.empty():
        obj_to_move = potential_obj_to_move_queue.get()
        if obj_to_move not in objects_to_move:
            objects_to_move.add(obj_to_move)
            for o2 in object_names:
                # OccludesPre
                is_o2_in_way_of_obj_to_move = state.binary_edges[(o2, obj_to_move)][1]

                # OccludesManip
                region_names = ['loading_region', 'home_region']
                for r in region_names:
                    if state.ternary_edges[(obj_to_move, o2, r)][0]:
                        print obj_to_move, o2, r, state.ternary_edges[(obj_to_move, o2, r)]

                goal_region = 'home_region'
                original_obj_region = 'loading_region'
                if obj_to_move in state.goal_entities:
                    is_o2_in_way_of_obj_to_move_to_region = state.ternary_edges[(obj_to_move, o2, goal_region)]
                else:
                    is_o2_in_way_of_obj_to_move_to_region = state.ternary_edges[(obj_to_move, o2, original_obj_region)]

                if is_o2_in_way_of_obj_to_move or is_o2_in_way_of_obj_to_move_to_region:
                    potential_obj_to_move_queue.put(o2)
    """

    return -len(objects_to_move)
