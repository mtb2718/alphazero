from connectfour import AlphaC4, ConnectFourState

import numpy as np
import torch


class MCTreeNode:

    EDGE_DTYPE = [('N', np.uint32),
                  ('W', np.float32),
                  ('P', np.float32)]

    def __init__(self, state, parent=None):
        """MCTreeNode is a thin wrapper around a game state to provide helper interfaces
        and book-keeping for MCTS.
        """

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
    def expanded(self):
        return self._expanded

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
        Q = W
        Q[N != 0] /= N[N != 0]
        Q[N == 0] = 0
        return Q

    @property
    def action_prior(self):
        # P(s, a)
        return self._edges['P']

    def traverse(self, action_index):
        """Construct new game state that results from taking given action and return the
        corresponding MCTreeNode in the search tree.
        """

        # TODO: This method should return the tree's new root and discard all dead paths
        # Should also invert all 'values' in tree to reflect new current player
        if self._children[action_index] is None:
            new_state = self.state.copy()
            new_state.take_action(action_index)
            new_node = MCTreeNode(new_state, self)
            self._children[action_index] = new_node
        return self._children[action_index]

    def expand(self, turn, p, v):
        """Evaluate policy and value of this node's state with the given model.
        
        Update the internally tracked values in the node and backup results in tree.
        """
        assert not self._expanded, "Multiple MCTreeNode expandsions is undefined."
        self._expanded = True
        print(f"Expanding state: {self.state._history}")
        #print(f"  Turn: {turn}")
        #print(f"  P: {p}")
        #print(f"  v: {v}")

        # Assume each player acts optimally, so that our action prior and state value
        # is always from the point-of-view of whoever's turn it is. However, (for a
        # two player game), we need to reverse the sign of the value in each backup
        # step working back towards the root of the tree. Intuitively, a strong position
        # for the current player implies weakness in the opponents prior position.

        self._edges['P'] = p
        self._value = v

        node = self
        while node.parent is not None:
            v *= -1 # Reverse sign of value for each successive parent
            prev_action = node.parent.children.index(node)
            node.parent._edges['W'][prev_action] += v
            node.parent._edges['N'][prev_action] += 1
            node = node.parent


def mcts_expand(root, net, c_puct=0.001):
    """Expand the tree from the given 'root' node.

    In other words, add one MCTreeNode to the tree originating at 'root',
    where the added nodes are selected via MCTS upper-confidence bound. TODO: clarify.
    """
    #       i.   If game is over in current (simulated) state, backup result
    #       ii.  If state hasn't been visited, evaluate (v, pi) = f(s), backup result
    #       iii. Else, traverse branch of a = argmax U(s,a)

    node = root

    while True:

        if not node.expanded:
            if node.state.winner is not None:
                p = node.action_prior
                v = 0 if node.state.winner == -1 else 1
            else:
                # TODO: Enqueue and block for results
                p, v = node.state.eval(net)
            node.expand(root.state.turn, p, v)
            break

        elif node.state.winner is not None:
            # Traversed tree to terminal state, nothing to do.
            break

        else:
            # TODO: Move this calculation to a TreeNode method
            U = c_puct * node.action_prior * np.sqrt(np.sum(node.num_visits)) / (1 + node.num_visits)
            i = np.argmax(node.mean_value + U)
            node = node.traverse(i)


def update_network(net, state_buffer):
    print('TODO: update net')


def update_databuffer(leaf, state_buffer):
    import pdb; pdb.set_trace()

    # Fill state buffer with (node, value, visit count)


# TODO: MCTS should be a utility
# Two main functions are to "play" or "simulate" and training
# Could run in parallel or in a single thread as below
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
    state_buffer = [] # TODO: more fancy

    NUM_EXPANSIONS_PER_DECISION = 4

    #while True: # TODO: check for convergence
    for _ in range(4):
        print('----------')

        tree = MCTreeNode(ConnectFourState())

        with torch.no_grad():
            # Play one game
            while tree.state.winner is None:
                # Make one move
                for _ in range(NUM_EXPANSIONS_PER_DECISION):
                    mcts_expand(tree, net)
                tree = tree.traverse(np.argmax(tree.num_visits))
            print(f'Winner {tree.state.winner}')

        update_databuffer(tree, state_buffer)
        update_network(net, state_buffer)


if __name__ == '__main__':
    alphazero_train()

