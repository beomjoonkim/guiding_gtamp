from motion_planners.rrt import TreeNode, configs
from motion_planners.utils import argmin
import numpy as np
from random import randint
import pickle
import Queue
import openravepy
import time

from mover_library.utils import visualize_path, se2_distance, are_base_confs_close_enough

prm_vertices = prm_edges = None


def get_number_of_confs_in_between(q1, q2, body):
    n = int(
        np.max(np.abs(np.divide(body.SubtractActiveDOFValues(q1, q2), np.array([0.3, 0.3, 40 * np.pi / 180.0]))))) + 1
    return n


def leftarm_torso_linear_interpolation(body, q1, q2, resolution):  # Sequence doesn't include q1
    """
    config_lower_limit = np.array([0.0115, -0.5646018, -0.35360022, -0.65000076, -2.12130808, -3.14159265, -2.0000077,
                                   -3.14159265]),
    config_upper_limit = np.array([0.305, 2.13539289, 1.29629967, 3.74999698, -0.15000005, 3.14159265,
                                   -0.10000004,  3.14159265])
    resolution = (config_upper_limit - config_lower_limit) * resolution # a portion of the configuration limits
    n = int(np.max(np.abs(np.divide(body.SubtractActiveDOFValues(q2, q1), resolution)))) + 1
    """
    n = int(np.max(np.abs(np.divide(body.SubtractActiveDOFValues(q2, q1), body.GetActiveDOFResolutions())))) + 1
    # If the resolution for a particular joint angle is say, 0.02, then we are assuming that within the 0.02 of the
    # angle value, there would not be a collision, or even if there is, we are going to ignore it.
    q = q1
    for i in range(n):
        q = (1. / (n - i)) * body.SubtractActiveDOFValues(q2, q) + q  # NOTE - do I need to repeatedly do the subtract?
        yield q


def leftarm_torso_extend_fn(body, resolution=0.05):
    return lambda q1, q2: leftarm_torso_linear_interpolation(body, q1, q2, resolution)


def collision_fn(env, body, check_self=False):
    def fn(q):
        with body:
            body.SetActiveDOFValues(q)
            return env.CheckCollision(body) or (check_self and body.CheckSelfCollision())

    return fn


def extend_fn(body):
    return lambda q1, q2: linear_interpolation(body, q1, q2)


def base_extend_fn(body):
    return lambda q1, q2: base_linear_interpolation(body, q1, q2)


def arm_base_extend_fn(body):
    return lambda q1, q2: arm_base_linear_interpolation(body, q1, q2)


def sample_fn(body, collisions=False):
    return lambda: cspace_sample(body) if not collisions else cspace_sample_collisions(body)


def arm_base_sample_fn(body, x_extents, y_extents, x=0, y=0):
    return lambda: base_arm_cspace_sample(body, x_extents, y_extents, x, y)


def base_sample_fn(body, x_extents, y_extents, x=0, y=0):  # body is passed in for consistency
    return lambda: base_cspace_sample(x_extents, y_extents, x, y)


def distance_fn(body):
    return lambda q1, q2: cspace_distance_2(body, q1, q2)


def base_distance_fn(body, x_extents, y_extents):
    return lambda q1, q2: base_distance(q1, q2, x_extents, y_extents)


def arm_base_distance_fn(body, x_extents, y_extents):
    return lambda q1, q2: arm_base_cspace_distance_2(body, q1, q2, x_extents, y_extents)


def base_distance(q1, q2, x_extents, y_extents):
    return se2_distance(q1, q2, c1=1, c2=0.1)

    # distance = abs(q1 - q2)
    # if distance[-1] > np.pi:
    #    distance[-1] = 2 * np.pi - distance[-1]
    # return np.dot(distance, 1 / np.array([2 * x_extents, 2 * y_extents, np.pi]))  # normalize them by their max values


def cspace_sample_collisions(body):
    while True:
        config = cspace_sample(body)
        body.SetActiveDOFValues(config)
        if not body.env().CheckCollision(body):  # NOTE - not thread-safe get rid of
            return config


def arm_base_cspace_distance_2(body, q1, q2, x_extents, y_extents):
    arm_diff = body.SubtractActiveDOFValues(q2, q1)[:-3]
    dim_arm = len(q1) - 3
    arm_weights = np.ones((dim_arm,)) * 2 * np.pi
    arm_dist = np.dot(1. / arm_weights, arm_diff * arm_diff)
    base_dist = base_distance(q1[-3:], q2[-3:], x_extents, y_extents)
    return base_dist + arm_dist


def cspace_distance_2(body, q1, q2):
    diff = body.SubtractActiveDOFValues(q2, q1)
    return np.dot(body.GetActiveDOFWeights(), diff * diff)


def base_cspace_sample(x_extents, y_extents, x, y):
    lower_lim = np.array([x - x_extents, y - y_extents, -np.pi])
    upper_lim = np.array([x + x_extents, y + y_extents, np.pi])
    return np.random.uniform(lower_lim, upper_lim)


def base_arm_cspace_sample(body, x_extents, y_extents, x, y):
    lower_lim = np.array([x - x_extents, y - y_extents, -np.pi])
    upper_lim = np.array([x + x_extents, y + y_extents, np.pi])
    base_config = np.random.uniform(lower_lim, upper_lim)
    arm_config = cspace_sample(body)[:-3]
    return np.hstack([arm_config, base_config])


def cspace_sample(body):
    return np.random.uniform(*body.GetActiveDOFLimits())  # TODO - adjust theta limits to be between [-PI, PI)


def arm_base_linear_interpolation(body, q1, q2):
    diff = body.SubtractActiveDOFValues(q2, q1)

    arm_resolution = body.GetActiveDOFResolutions()[:-3] * 5
    base_resolution = np.array([0.3, 0.3, 20 * np.pi / 180.0])
    full_config_resolution = np.hstack([arm_resolution, base_resolution])
    n = int(np.max(np.abs(np.divide(diff, full_config_resolution)))) + 1
    q = q1
    for i in range(n):
        q = (1. / (n - i)) * body.SubtractActiveDOFValues(q2, q) + q  # NOTE - do I need to repeatedly subtract?
        yield q


def linear_interpolation(body, q1, q2):  # Sequence doesn't include q1
    n = int(np.max(np.abs(np.divide(body.SubtractActiveDOFValues(q2, q1), body.GetActiveDOFResolutions() * 10)))) + 1
    q = q1
    for i in range(n):
        q = (1. / (n - i)) * body.SubtractActiveDOFValues(q2, q) + q  # NOTE - do I need to repeatedly do the subtract?
        yield q


"""
def base_linear_interpolation(body, q1, q2):
    n = int(
        np.max(np.abs(np.divide(body.SubtractActiveDOFValues(q2, q1), np.array([0.2, 0.2, 20 * np.pi / 180.0]))))) + 1
    q = q1
    interpolated_qs = []
    for i in range(n):
        q = (1. / (n - i)) * body.SubtractActiveDOFValues(q2, q) + q  # NOTE - do I need to repeatedly do the subtract?
        interpolated_qs.append(q)
    return interpolated_qs
"""


def base_linear_interpolation(body, q1, q2):
    n = get_number_of_confs_in_between(q1, q2, body)
    q = q1
    interpolated_qs = []
    for i in range(n):
        curr_q = q
        q = (1. / (n - i)) * body.SubtractActiveDOFValues(q2,
                                                          curr_q) + curr_q  # NOTE - do I need to repeatedly do the subtract?
        if q[-1] > np.pi:
            q[-1] = q[-1] - 2 * np.pi
        if q[-1] < -np.pi:
            q[-1] = q[-1] + 2 * np.pi
        interpolated_qs.append(q)
    return interpolated_qs


def rrt_region(q1, region, distance, sample, extend, collision, iterations):
    # check if q1 or q2 is in collision
    if collision(q1):
        print 'ignoring collision in initial'
        # return None

    # define two roots of the tree
    root1 = TreeNode(q1)

    # tree1_nodes grows from q1, tree2_nodes grows from q2
    tree1_nodes = [root1]

    # sample and extend iterations number of times
    for ntry in range(iterations):
        # sample a configuration
        s = sample()

        # returns the node with the closest distance to s from a set of nodes tree1_nodes
        extended_tree1_node = argmin(lambda n: distance(n.config, s), tree1_nodes)

        # extend from the closest config to s
        for q in extend(extended_tree1_node.config, s):  # I think this is what is taking up all the time
            # if there is a collision, extend upto that collision
            if collision(q):
                break
            extended_tree1_node = TreeNode(q, parent=extended_tree1_node)
            tree1_nodes.append(extended_tree1_node)
            import pdb;
            pdb.set_trace()
            if region.contains_point(q):
                path1 = extended_tree1_node.retrace()
                return configs(path1)

    return None


def rrt_connect(q1, q2, distance, sample, extend, collision, iterations):
    # check if q1 or q2 is in collision
    if collision(q1) or collision(q2):
        print 'collision in either initial or goal'
        return None

    # define two roots of the tree
    root1, root2 = TreeNode(q1), TreeNode(q2)

    # tree1_nodes grows from q1, tree2_nodes grows from q2
    tree1_nodes, tree2_nodes = [root1], [root2]

    # sample and extend iterations number of times
    for ntry in range(iterations):
        if len(tree1_nodes) > len(tree2_nodes):  # ????
            tree1_nodes, tree2_nodes = tree2_nodes, tree1_nodes

        # sample a configuration
        s = sample()

        # returns the node with the closest distance to s from a set of nodes tree1_nodes
        tree1_node_closest_to_new_config = argmin(lambda n: distance(n.config, s), tree1_nodes)

        # extend from the closest config to s
        extended_tree1_node = tree1_node_closest_to_new_config
        for q in extend(extended_tree1_node.config, s):  # I think this is what is taking up all the time
            # if there is a collision, extend upto that collision
            if collision(q):
                break
            extended_tree1_node = TreeNode(q, parent=extended_tree1_node)
            tree1_nodes.append(extended_tree1_node)

        # try to extend to the tree grown from the other side
        extended_tree2_node = argmin(lambda n: distance(n.config, extended_tree1_node.config), tree2_nodes)

        for q in extend(extended_tree2_node.config, extended_tree1_node.config):
            if collision(q):
                break
            extended_tree2_node = TreeNode(q, parent=extended_tree2_node)
            tree2_nodes.append(extended_tree2_node)
        else:  # where is the if for this else?
            # apparently, if none of q gets into
            # if  collision(q) stmt, then it will enter here
            # two trees meet at the new configuration s
            path1, path2 = extended_tree1_node.retrace(), extended_tree2_node.retrace()
            if path1[0] != root1:
                path1, path2 = path2, path1
            return configs(path1[:-1] + path2[::-1])
    return None


def rrt_connect_renamed(q1, q2, distance, sample, extend, collision, iterations):
    # check if q1 or q2 is in collision
    if collision(q1) or collision(q2):
        print 'collision in either initial or goal'
        return None

    # define two roots of the tree
    root1, root2 = TreeNode(q1), TreeNode(q2)

    # tree1_nodes grows from q1, tree2_nodes grows from q2
    tree1_nodes, tree2_nodes = [root1], [root2]

    # sample and extend iterations number of times
    for ntry in range(iterations):
        if len(tree1_nodes) > len(tree2_nodes):  # balances the sizes of the trees
            tree1_nodes, tree2_nodes = tree2_nodes, tree1_nodes

        # sample a configuration
        new_config = sample()

        # returns the node with the closest distance to s from a set of nodes tree1_nodes
        tree1_node_closest_to_new_config = argmin(lambda n: distance(n.config, new_config), tree1_nodes)

        # extend from the closest config to s
        extended_tree1_node = tree1_node_closest_to_new_config
        for q in extend(tree1_node_closest_to_new_config.config, new_config):
            # if there is a collision, extend upto that collision
            if collision(q):
                break
            extended_tree1_node = TreeNode(q, parent=tree1_node_closest_to_new_config)
            tree1_nodes.append(extended_tree1_node)

        # try to extend to the tree grown from the other side
        tree2_node_closest_to_extended_tree1_node = argmin(lambda n: distance(n.config, extended_tree1_node.config),
                                                           tree2_nodes)  # min dist to extended_tree1_node (extended to new config s)
        extended_tree2_node = tree2_node_closest_to_extended_tree1_node
        for q in extend(tree2_node_closest_to_extended_tree1_node.config, extended_tree1_node.config):
            if collision(q):
                break
            extended_tree2_node = TreeNode(q, parent=tree2_node_closest_to_extended_tree1_node)
            tree2_nodes.append(extended_tree2_node)
        else:  # where is the if for this else? if none of q gets into if-collision(q) stmt, then it will enter here
            # two trees meet
            path1, path2 = extended_tree1_node.retrace(), extended_tree2_node.retrace()
            if path1[0] != root1:
                path1, path2 = path2, path1
            return configs(path1[:-1] + path2[::-1])

    return None


from manipulation.regions import AARegion


def get_goal_config_used(motion_plan, potential_goal_configs):
    which_goal = np.argmin(np.linalg.norm(motion_plan[-1][0:2] - np.array(potential_goal_configs)[:, 0:2], axis=-1))
    return potential_goal_configs[which_goal]


# returns list of paths, 1 for each goal function
def find_prm_path(start, goal_fns, heuristic, is_collision):
    results = [None] * len(goal_fns)  # why do you have multiple goal functions?
    visited = {s for s in start}
    queue = Queue.PriorityQueue()
    for s in start:
        queue.put((heuristic(s), 0, np.random.rand(), s, [s]))

    while not queue.empty():
        _, dist, _, vertex, path = queue.get()

        for next in prm_edges[vertex] - visited:
            visited.add(next)
            if is_collision(next): # I think this can be lazily checked?
                continue
            for i, goal_fn in enumerate(goal_fns):
                if results[i] is None and goal_fn(next):
                    results[i] = path + [next]
            else:
                newdist = dist + np.linalg.norm(prm_vertices[vertex] - prm_vertices[next])
                queue.put((newdist + heuristic(next), newdist, np.random.rand(), next, path + [next]))
    return results


def init_prm():
    global prm_vertices
    global prm_edges
    if prm_vertices is None or prm_edges is None:
        prm_vertices, prm_edges = pickle.load(open('./prm.pkl', 'rb'))


def prm_connect(q1, q2, collision_checker):
    global prm_vertices
    global prm_edges

    is_goal_region = False
    is_multiple_goals = False
    if type(q2) is AARegion:
        is_goal_region = True
    else:
        is_multiple_goals = isinstance(q2, list)  # and len(q2[0]) == len(q1)

    is_single_goal = not is_goal_region and not is_multiple_goals
    collision_checker_is_set = isinstance(collision_checker, set)

    if prm_vertices is None or prm_edges is None:
        prm_vertices, prm_edges = pickle.load(open('./prm.pkl', 'rb'))

    no_collision_checking = collision_checker_is_set and len(collision_checker) == 0
    if not no_collision_checking:
        env = openravepy.RaveGetEnvironment(1)
        robot = env.GetRobot('pr2')
        non_prm_config_collision_checker = collision_fn(env, robot)

        if non_prm_config_collision_checker(q1):
            #print "initial config in collision"
            return None

        if is_single_goal and non_prm_config_collision_checker(q2):
            #print "goal config in collision"
            return None

        if is_multiple_goals:
            q2_original = q2
            q2 = [q_goal for q_goal in q2_original if not non_prm_config_collision_checker(q_goal)]

            if len(q2) == 0:
                return None

    if is_single_goal and are_base_confs_close_enough(q1, q2, xy_threshold=0.8, th_threshold=50.):
        return [q1, q2]

    if is_multiple_goals:
        for q_goal in q2:
            if are_base_confs_close_enough(q1, q_goal, xy_threshold=0.8, th_threshold=50.):
                return [q1, q_goal]

    if is_multiple_goals:
        def is_connected_to_goal(prm_vertex_idx):
            q = prm_vertices[prm_vertex_idx]
            goals = q2
            if len(q.squeeze()) != 3:
                raise NotImplementedError
            return any(are_base_confs_close_enough(q, g, xy_threshold=0.8, th_threshold=50.) for g in goals)
    elif is_goal_region:
        def is_connected_to_goal(prm_vertex_idx):
            q = prm_vertices[prm_vertex_idx]
            goal_region = q2
            return goal_region.contains_point(q)
    else:
        def is_connected_to_goal(prm_vertex_idx):
            q = prm_vertices[prm_vertex_idx]
            return are_base_confs_close_enough(q, q2, xy_threshold=0.8, th_threshold=50.)

    def heuristic(q):
        return 0

    def is_collision(q):
        return q in collision_checker if collision_checker_is_set else collision_checker(prm_vertices[q])

    # start = {i for i, q in enumerate(prm_vertices) if np.linalg.norm((q - q1)[:2]) < .8}

    start = set()
    for idx, q in enumerate(prm_vertices):
        q_close_enough_to_q1 = are_base_confs_close_enough(q, q1, xy_threshold=0.8, th_threshold=50.)
        if q_close_enough_to_q1:
            if not is_collision(idx):
                start.add(idx)
    path = find_prm_path(start, [is_connected_to_goal], heuristic, is_collision)[0]
    if path is not None:
        path = [q1] + [prm_vertices[i] for i in path]
        if is_single_goal:
            path += [q2]
        elif is_multiple_goals:
            path += [get_goal_config_used(path, q2)]
        return path
    else:
        return None


def direct_path(q1, q2, extend, collision):
    path = [q1]
    for q in extend(q1, q2):
        if collision(q):
            return None
        path.append(q)
    return path


def smooth_path(path, extend, collision, iterations=50):
    smoothed_path = path
    for _ in range(iterations):
        if len(smoothed_path) <= 2:
            return smoothed_path
        i, j = randint(0, len(smoothed_path) -
                       1), randint(0, len(smoothed_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend(smoothed_path[i], smoothed_path[j]))
        if len(shortcut) < j - i and all(not collision(q) for q in shortcut):
            smoothed_path = smoothed_path[
                            :i + 1] + shortcut + smoothed_path[j + 1:]
    return smoothed_path
