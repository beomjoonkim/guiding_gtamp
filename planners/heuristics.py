import Queue


def get_objects_to_move(state, problem_env):
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

    if 'two_arm' in problem_env.name:
        regions = ['home_region', 'loading_region']
    else:
        regions = ['center_shelf_region', 'rectangular_packing_box1_region']

    n_occludes_pre = 0
    n_occludes_manip = 0
    while not potential_obj_to_move_queue.empty():
        obj_to_move = potential_obj_to_move_queue.get()
        if obj_to_move not in objects_to_move:
            objects_to_move.add(obj_to_move)
            for o2 in object_names:
                # OccludesPre
                is_o2_in_way_of_obj_to_move = state.binary_edges[(o2, obj_to_move)][1]

                # OccludesManip - should this be to any region?
                is_o2_in_way_of_obj_to_move_to_any_region = any(
                    [state.ternary_edges[(obj_to_move, o2, r)][0] for r in regions])

                if is_o2_in_way_of_obj_to_move:
                    if obj_to_move in state.goal_entities:
                        #print "%s occludes goal entity" % o2
                        pass

                    n_occludes_pre += 1

                if is_o2_in_way_of_obj_to_move_to_any_region:
                    if obj_to_move in state.goal_entities:
                        #print "%s occludes manipfree of goal entity" % o2
                        pass
                    n_occludes_manip += 1

                if is_o2_in_way_of_obj_to_move or is_o2_in_way_of_obj_to_move_to_any_region:
                    potential_obj_to_move_queue.put(o2)
    #print "n occludes pre %d n occludes manip %d" % (n_occludes_pre, n_occludes_manip)
    return objects_to_move


def compute_hcount(state, problem_env):
    objects_to_move = get_objects_to_move(state, problem_env)
    return len(objects_to_move)


def compute_hcount_with_action(state, action, problem_env):
    objects_to_move = get_objects_to_move(state, problem_env)
    n_objs_to_move = len(objects_to_move)

    if 'two_arm' in problem_env.name:
        a_obj = action.discrete_parameters['two_arm_place_object']
        a_region = action.discrete_parameters['two_arm_place_region']
    else:
        a_obj = action.discrete_parameters['object'].GetName()
        a_region = action.discrete_parameters['region'].name

    is_a_obj_manip_free_to_a_region = state.binary_edges[(a_obj, a_region)][-1]
    is_a_in_objects_to_move = a_obj in objects_to_move
    is_a_obj_reachable = state.nodes[a_obj][9]

    if is_a_obj_reachable and is_a_in_objects_to_move and is_a_obj_manip_free_to_a_region:
        n_objs_to_move -= 1
    return n_objs_to_move
