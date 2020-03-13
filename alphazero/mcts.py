import numpy as np
import torch


class MCTreeNode:

    EDGE_DTYPE = [('N', np.uint32),
                  ('W', np.float32),
                  ('P', np.float32)]

    def __init__(self, parent=None):
        self.parent = parent
        self.children = None
        self._edges = None

    @property
    def expanded(self):
        return self.children is not None

    def expand(self, p):
        num_actions = len(p)
        self.children = [None] * num_actions
        self._edges = np.zeros(num_actions, dtype=MCTreeNode.EDGE_DTYPE)
        assert np.all(p >= 0), f'Node must be expanded with proper, non-negative distribution. Got: {p}'
        assert np.abs(np.sum(p) - 1) < 1e-6, f'Node must be expanded with normalized distribution. Got: {p}'
        self._edges['P'] = p

    def traverse(self, action_index):
        assert action_index < len(self.children)
        if self.children[action_index] is None:
            self.children[action_index] = MCTreeNode(self)
        return self.children[action_index]

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

    def ucb_score(self, c_base=19652, c_init=1.25):
        Q = self.mean_value
        P = self.action_prior
        Ns = np.sum(self.num_visits)
        Cs = np.log((1 + Ns + c_base) / c_base) + c_init
        s = np.sqrt(Ns) / (1 + self.num_visits)
        U = Cs * P * s
        return Q + U

    def pi(self, temperature=1):
        assert temperature > 0
        nt = self._edges['N'] ** (1 / temperature)
        s = np.sum(nt)
        if s > 0:
            return nt / s
        else:
            return np.ones_like(nt) / len(nt)

    def sample_action(self, temperature=1):
        return np.random.choice(len(self.children), p=self.pi(temperature))

    def greedy_action(self):
        return np.argmax(self.pi())

    def kill_siblings(self):
        if self.parent is None:
            return
        for i, sibling in enumerate(self.parent.children):
            if sibling != self:
                self.parent.children[i] = None
                del sibling

    def add_value_observation(self, action_index, v):
        self._edges['W'][action_index] += v
        self._edges['N'][action_index] += 1

    def add_exploration_noise(self, alpha, epsilon):
        P = self.action_prior
        eta = np.random.dirichlet([alpha] * len(P))
        self._edges['P'] = (1 - epsilon) * P + epsilon * eta


def evaluate(game, model):
    model.eval()
    with torch.no_grad():
        try:
            device = next(model.parameters()).device
        except:
            device = torch.device('cpu')
        x = game.render()
        x = torch.from_numpy(x[None, ...]).to(device)
        valid = torch.zeros(1, game.NUM_ACTIONS, dtype=torch.bool, device=device)
        valid[0, game.valid_actions] = 1
        p, v = model(x, valid)
        p = torch.nn.functional.softmax(p[valid], dim=0)
        return p.cpu().numpy(), v[0].cpu().numpy()


def run_mcts(game, root, model, num_simulations, alpha=0.3, epsilon=0.25):
    # TODO: Tree search args from config

    # Expand root and add exploration noise
    if not root.expanded:
        p, _ = evaluate(game, model)
        root.expand(p)
    root.add_exploration_noise(alpha, epsilon)

    for _ in range(num_simulations):
        node = root
        simulation = game.clone()

        while node.expanded and not simulation.terminal:
            action_index = np.argmax(node.ucb_score())
            node = node.traverse(action_index)
            action = simulation.valid_actions[action_index]
            simulation.take_action(action)

        # Took 'action' to land in the current terminal or unevaluated state.

        if simulation.terminal:
            # Get the terminal value of the simulation from the viewpoint of
            # the player that took the action leading into the terminal state.
            v = simulation.terminal_value(simulation.prev_player)
        else:
            # Evaluate leaf node and expand search frontier
            p, v = evaluate(simulation, model)
            node.expand(p)

            # v is an estimate of V(s), the value of the evaluated state from the viewpoint of the
            # next player whose turn it is to make a move. Since we're using this result to
            # estimate Q(s,a) for the parent state and action leading into the current state,
            # we invert the value.
            v *= -1

        # Backup value through this simulation's search path
        while node.parent is not None:
            # Important: Shouldn't backpropogate all the way to root of game!
            # Break here so we only backprop to the root of this particular move's MCTS.
            if node == root:
                break
            action_index = node.parent.children.index(node)
            node.parent.add_value_observation(action_index, v)
            # We need to reverse the sign of the value in each backup step working towards
            # the root of the tree. Intuitively, a strong position for the current player
            # implies weakness in the opponent's prior position.
            v *= -1
            node = node.parent
