from connectfour import AlphaC4, ConnectFourState

import numpy as np


class MCTreeNode:

    EDGE_DTYPE = [('N', np.uint32),
                  ('W', np.float32),
                  ('P', np.float32)]

    def __init__(self, state, parent=None):
        """MCTreeNode is a thin wrapper around a game state to provide helper interfaces
        and book-keeping for MCTS.
        """

        self._state = state
        self._parent = parent

        N = len(state.valid_actions)
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
        Q = W / N
        Q[N == 0] = 0
        return Q

    @property
    def action_prior(self):
        # P(s, a)
        return self._edges['P']

    def traverse(self, action):
        """Construct new game state that results from taking given action and return the
        corresponding MCTreeNode in the search tree.
        """
        if self._children[action] is None:
            new_state = self._state.copy()
            new_state.take_action(action)
            new_node = MCTreeNode(new_state, self)
            self._children[action] = new_node
        return self._children[action]

    def expand(self, net):
        """Evaluate policy and value of this node's state with the given model.
        
        Update the internally tracked values in the ndoe and backup results in tree.
        """
        assert not self._expanded, "Multiple MCTreeNode expandsions is undefined."
        p, v = self.state.eval(net) # TODO: Enqueue and block for results
        self._edges['P'] = p
        # TODO: Need to handle multi-players here--i.e. should be -v for other
        # player if +v for me in a two player game.
        self._value = v
        node = self
        while node is not None:
            if node.parent is not None:
                prev_action = node.parent.children.index(node)
                node.parent._edges['W'][prev_action] += v
                node.parent._edges['N'][prev_action] += 1
            node = node.parent


def mcts(root, net, num_expansions, c_puct=0.001):
    """Expand the tree from the given 'root' node 'num_expansion' times.

    In other words, add 'num_expansion' MCTreeNode's to the tree originating at 'root',
    where the added nodes are selected via MCTS upper-confidence bound. TODO: clarify.

    """

    #       i.   If game is over in current (simulated) state, backup result
    #       ii.  If state hasn't been visited, evaluate (v, pi) = f(s), backup result
    #       iii. Else, traverse branch of a = argmax U(s,a)

    for _ in range(num_expansions):

        node = root

        while True:

            if np.sum(node.num_visits) == 0:
                node.expand(net)
                break

            else:
                # TODO: implement for real
                import pdb; pdb.set_trace()
                U = c_puct * node.P_a * sqrt(sum(node.num_visits)) / (1 + node.num_visits)
                a = argmax(node.Q + U)

                # Just traverse, nothing fancy
                node = node.traverse(a)


def update_network(net, state_buffer):
    print('TODO: update net')

def update_databuffer(leaf, state_buffer):
    pass


# TODO: MCTS should be a utility
# Two main functions are to "play" or "simulate" and training
# Could run in parallel or in a single thread as below

def test_mode(model):
    model.train(False)
    for param in model.parameters():
        param.requires_grad = False

def alphazero_train():
    # Until converged:
    # 0. Start new game
    # 1. Until game is over:
    #    a. Run MCTS from game state (for 800 expansions)
    #       i.   If game is over in current (simulated) state, backup result
    #       ii.  If state hasn't been visited, evaluate (v, pi) = f(s), backup result
    #       iii. Else, traverse branch of a = argmax U(s,a)
    #    b. Sample move from the state's UCB posterior
    #    c. Update search tree root to reflect new game state
    # 2. Annotate and push game states from game into length B buffer
    # 3. Train network for N batches of K game states

    net = AlphaC4()
    test_mode(net)
    state_buffer = [] # TODO: more fancy

    NUM_EXPANSIONS_PER_DECISION = 2

    while True: # TODO: check for convergence
        tree = MCTreeNode(ConnectFourState())

        while tree.state.winner is None:
            mcts(tree, net, NUM_EXPANSIONS_PER_DECISION)
            action = np.argmax(tree.num_visits)
            tree = tree.traverse(action)
        update_databuffer(tree, state_buffer)
        update_network(net, state_buffer)


if __name__ == '__main__':
    alphazero_train()
