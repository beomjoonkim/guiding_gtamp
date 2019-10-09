
class Node(object):
    def __init__(self, parent, action, state, reward=0):
        self.parent = parent  # parent.state is initial state
        self.action = action
        self.state = state  # resulting state
        self.reward = reward  # resulting reward
        self.heuristic_vals = {}
        #for a in self.action:
        #    self.heuristic_vals[a] = None # used only for the root node

        if parent is None:
            self.depth = 1
        else:
            self.depth = parent.depth + 1

    def set_heuristic(self, action, val):
        self.heuristic_vals[action] = val

    def backtrack(self):
        node = self
        while node is not None:
            yield node
            node = node.parent