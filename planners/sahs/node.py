
class Node(object):
    def __init__(self, parent, action, state, reward=0):
        self.parent = parent  # parent.state is initial state
        self.action = action
        self.state = state  # resulting state
        self.reward = reward  # resulting reward

        if parent is None:
            self.depth = 1
        else:
            self.depth = parent.depth + 1

    def backtrack(self):
        node = self
        while node is not None:
            yield node
            node = node.parent