from connectfour import AlphaC4, ConnectFourState

import numpy as np
import torch
from torch.nn.functional import log_softmax
from torch.optim import SGD


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

    @property
    def UCT(self):
        # U(s, a)
        num = self.action_prior * np.sqrt(np.sum(self.num_visits))
        den = 1 + self.num_visits
        return num / den


    def pi(self, temperature):
        assert temperature > 0
        nt = self._edges['N'] ** (1 / temperature)
        return nt / np.sum(nt)


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


    def expand(self, net, c_puct=0.001, epsilon=0.25, alpha=0.5, maxdepth=200):
        """Expand the tree rooted at this node."""

        node = self
        v = None

        # Traverse tree to find leaf, taking caution to avoid infinite loops
        for _ in range(maxdepth):

            # i.   If game is over in current (simulated) state, backup result
            # ii.  If state hasn't been visited, evaluate (v, pi) = f(s), backup result
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

            Q = node.action_prior
            if node == self:
                eta = np.random.dirichlet([alpha] * len(Q))
                Q = (1 - epsilon) * Q + epsilon * eta
            action_index = np.argmax(Q + c_puct * node.UCT)
            node = node.traverse(action_index)

        # Search terminated early
        if v is None:
            print('WARNING: search tree expansion terminated early.')
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




def update_databuffer(leaf, state_buffer):
    # Assign value of leaf state and parents based on outcome of game.
    # 0 for draw, otherwise assume all leaf states corresponding to current player winning.
    # TODO: Add support for multiplayer games (Blokus)
    assert leaf.state.winner is not None
    value = 1 if leaf.state.winner >= 0 else 0

    MAX_BUFFER_SIZE = 64 * 1024

    # Fill state buffer with (node, value, action distribution)
    node = leaf

    while node is not None:
        total_num_visits = np.sum(node.num_visits)
        if total_num_visits > 0:
            action_distribution = node.num_visits / total_num_visits
        else:
            action_distribution = None
        state_buffer.append((node.state, value, action_distribution))
        value *= -1
        node = node.parent

    state_buffer = state_buffer[-MAX_BUFFER_SIZE:]
    print(f'Updated buffer, now contains {len(state_buffer)} states')


def update_network(net, optimizer, state_buffer, game_iter):

    # 1. sample a batch from state_buffer
    # 2. do a forward pass
    # 3. post-processing for valid moves?
    # 4. evaluate loss, do .backward()
    # 5. save results, checkpoints

    B = 64

    # Don't update network unless we can make a full batch
    if len(state_buffer) < B:
        return

    sample_indices = np.random.choice(len(state_buffer), B, False)

    # TODO: shared code with node.eval
    # Should maybe also make use of datasets / dataworkers

    x = torch.zeros(B, 2, 6, 7)
    z = torch.zeros(B, 1, 1, 1)
    p = torch.zeros(B, 1, 6, 7)
    p_valid = torch.ones(B, dtype=torch.bool)
    valid_action_mask = torch.zeros(B, 1, 6, 7)

    for i, s in enumerate(sample_indices):
        state, zi, pi = state_buffer[s]
        board = state.board

        # Channel 0 always equals current player's position
        if state.turn == 1:
            board[[0, 1]] = board[[1, 0]]

        x[i] = torch.from_numpy(board)
        z[i] = zi

        for c in range(7):
            next_row = state._history.count(c)
            if next_row < 6:
                valid_action_mask[i, 0, next_row, c] = 1

        # Convert pi from list of action probabilities to board representation
        # Note we'll mask all invalid moves outside of this loop, below.
        if pi is None:
            # pi may be None for terminal game states, where children visit counts will be 0.
            p_valid[i] = 0
        else:
            for j, ai in enumerate(state.valid_actions):
                p[i, 0, :, ai] = pi[j]

    # Run inference
    p_hat, v_hat = net(x.float())

    # Mask out invalid actions in both the prediction and the target.
    p *= valid_action_mask
    p_hat *= valid_action_mask

    # Calculate loss
    l_v = ((z - v_hat) ** 2).view(B)
    logp_hat = log_softmax(p_hat.view(B, -1), dim=1)
    l_p = torch.sum(p.view(B, -1) * logp_hat, dim=1)
    l_p[~p_valid] = 0.0

    total_loss = torch.sum(l_v + l_p)
    print(f'Total loss: {total_loss}, vloss: {torch.sum(l_v)}, ploss: {torch.sum(l_p)}')

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()



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
    optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    state_buffer = [] # TODO: more fancy

    NUM_EXPANSIONS_PER_DECISION = 16

    #while True: # TODO: check for convergence
    i = 0
    for _ in range(100):

        tree = MCTreeNode(ConnectFourState())

        with torch.no_grad():
            # Play one game
            while tree.state.winner is None:
                # Make one move
                for _ in range(NUM_EXPANSIONS_PER_DECISION):
                    tree.expand(net)
                action_index = np.random.choice(len(tree.state.valid_actions), p=tree.pi(1))
                tree = tree.traverse(action_index)

            print(f'Game {i} winner: Player {tree.state.winner}')

        update_databuffer(tree, state_buffer)
        update_network(net, optimizer, state_buffer, i)
        i += 1


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    alphazero_train()

    print('Done.')
