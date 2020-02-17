import numpy as np


class MCTreeNode:

    EDGE_DTYPE = [('N', np.uint32),
                  ('W', np.float32),
                  ('P', np.float32)]

    def __init__(self, state, parent=None):
        """MCTreeNode is a thin wrapper around a game state that implements MCTS."""

        N = len(state.valid_actions)

        self.state = state
        self.parent = parent
        self.children = [None] * N
        self._expanded = False
        self._edges = np.zeros(N, dtype=MCTreeNode.EDGE_DTYPE)

    @property
    def action_prior(self):
        return self._edges['P']

    @property
    def num_visits(self):
        return self._edges['N']

    @property
    def mean_value(self):
        # Q(s, a)
        W = self._edges['W']
        N = self._edges['N']
        Q = W.copy()
        Q[N != 0] /= N[N != 0]
        Q[N == 0] = 0
        return Q

    def pi(self, temperature=1):
        assert temperature > 0
        nt = self._edges['N'] ** (1 / temperature)
        s = np.sum(nt)
        if s > 0:
            return nt / s
        else:
            return np.ones_like(self._edges['N']) / len(self._edges)

    def kill_siblings(self):
        if self.parent is None:
            return
        for i, sibling in enumerate(self.parent.children):
            if sibling != self:
                self.parent.children[i] = None

    def traverse(self, action_index):
        """Construct new game state that results from taking given action.
        Return the corresponding MCTreeNode in the search tree.
        """
        if self.children[action_index] is None:
            new_state = self.state.copy()
            new_state.take_action(action_index)
            new_node = MCTreeNode(new_state, self)
            self.children[action_index] = new_node
        return self.children[action_index]

    def expand(self, net, c_base=19652, c_init=1.25, epsilon=0.25, alpha=0.3, maxdepth=200):
        """Expand the tree rooted at this node."""

        node = self
        v = None

        # Traverse tree to find leaf, taking caution to avoid infinite loops
        for depth in range(maxdepth):

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
            Ns = np.sum(node.num_visits)
            Cs = np.log((1 + Ns + c_base) / c_base) + c_init
            s = np.sqrt(Ns) / (1 + node.num_visits)
            U = Cs * P * s
            action_index = np.argmax(Q + U)
            node = node.traverse(action_index)

        # Search terminated early
        if v is None:
            print("WARNING: search tree expansion terminated early.")
            return

        # Backup results
        # Assume each player acts optimally, so that our action prior and state value
        # is always from the point-of-view of whoever's turn it is (e.g. the parent of
        # the node being evaluated). However, (for a two player game) we need to reverse
        # the sign of the value in each backup step working towards the root of the tree.
        # Intuitively, a strong position for the current player implies weakness in the
        # opponent's prior position.
        while node.parent is not None:
            prev_action_index = node.parent.children.index(node)
            node.parent._edges['W'][prev_action_index] += v
            node.parent._edges['N'][prev_action_index] += 1
            v *= -1
            if node == self:
                # Important: Shouldn't backpropogate all the way to root of game!
                # Break here so we only backprop to the root of this particular move's MCTS.
                break
            else:
                node = node.parent

