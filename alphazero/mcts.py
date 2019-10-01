import numpy as np


class MCTreeNode:

    EDGE_DTYPE = [('N', np.uint32),
                  ('W', np.float32),
                  ('P', np.float32)]

    def __init__(self, state, parent=None):
        """MCTreeNode is a thin wrapper around a game state that implements MCTS."""

        N = len(state.valid_actions)

        self._state = state
        self._parent = parent
        self._value = 0
        self._expanded = False
        self._children = [None] * N
        self._edges = np.zeros(N, dtype=MCTreeNode.EDGE_DTYPE)

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        # TODO: Replace explicit children here with a dict mapping states to
        # nodes. Since a game state can be reached from multiple paths, we should
        # let the game node decide how much game history (if any) belongs in the state.
        return self._children

    @property
    def state(self):
        return self._state

    @property
    def action_prior(self):
        # P(s, a)
        return self._edges['P']

    @property
    def num_visits(self):
        # N(s, a)
        return self._edges['N']

    @property
    def total_value(self):
        # W(s, a)
        return self._edges['W']

    @property
    def mean_value(self):
        # Q(s, a)
        W = self._edges['W']
        N = self._edges['N']
        Q = W
        Q[N != 0] /= N[N != 0]
        Q[N == 0] = 0
        return Q

    def pi(self, temperature=1):
        assert temperature > 0
        nt = self._edges['N'] ** (1 / temperature)
        return nt / np.sum(nt)

    def kill_siblings(self):
        if self._parent is None:
            return
        for i, sibling in enumerate(self._parent.children):
            if sibling != self:
                self._parent._children[i] = None

    def traverse(self, action_index):
        """Construct new game state that results from taking given action.
        Return the corresponding MCTreeNode in the search tree.
        """
        if self._children[action_index] is None:
            new_state = self.state.copy()
            new_state.take_action(action_index)
            new_node = MCTreeNode(new_state, self)
            self._children[action_index] = new_node
        return self._children[action_index]

    def expand(self, net, c_puct=1.0, epsilon=0.25, alpha=0.8, maxdepth=200):
        """Expand the tree rooted at this node."""

        node = self
        v = None

        # Traverse tree to find leaf, taking caution to avoid infinite loops
        for _ in range(maxdepth):

            # i.   If game is over in current (simulated) state, backup result
            # ii.  If state hasn't been visited, evaluate (v, p) = f(s), backup result
            # iii. Else, traverse branch of a = argmax Q(s,a) + U(s,a)

            if node.state.winner is not None:
                v = 0 if node.state.winner == -1 else 1
                break
            elif not node._expanded:
                # TODO: Enqueue and block for results
                p, v = node.state.eval(net)
                node._edges['P'] = p
                node._expanded = True
                break

            # TODO: Utility function for evaluating UCT?
            Q = node.mean_value
            P = node.action_prior
            if node == self:
                eta = np.random.dirichlet([alpha] * len(P))
                P = (1 - epsilon) * P + epsilon * eta
            s = np.sqrt(np.sum(node.num_visits)) / (1 + node.num_visits)
            U = c_puct * P * s
            action_index = np.argmax(Q + U)
            node = node.traverse(action_index)

        # Search terminated early
        if v is None:
            print("WARNING: search tree expansion terminated early.")
            return

        # Backup results
        # Assume each player acts optimally, so that our action prior and state value
        # is always from the point-of-view of whoever's turn it is. However, (for a
        # two player game), we need to reverse the sign of the value in each backup
        # step working towards the root of the tree. Intuitively, a strong position
        # for the current player implies weakness in the opponents prior position.
        self._value = v
        while node.parent is not None:
            v *= -1
            prev_action_index = node.parent.children.index(node)
            node.parent._edges['W'][prev_action_index] += v
            node.parent._edges['N'][prev_action_index] += 1
            node = node.parent
