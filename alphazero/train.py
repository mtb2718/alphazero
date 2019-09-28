from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn.functional import log_softmax
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

from connectfour import AlphaZeroC4, ConnectFourState
from mcts import MCTreeNode


def update_databuffer(leaf, state_buffer, summary_writer):
    # Assign value of leaf state and parents based on outcome of game.
    # 0 for draw, otherwise assume all leaf states corresponding to current player winning.
    # TODO: Add support for multiplayer games (Blokus)
    assert leaf.state.winner is not None
    value = 1 if leaf.state.winner >= 0 else 0

    MAX_BUFFER_SIZE = 64 * 1024

    # Fill state buffer with (node, value, action distribution)
    # Implicit assumption is that leaf node is state after winning move (or draw) of current player
    node = leaf

    while node is not None:
        total_num_visits = np.sum(node.num_visits)
        action_distribution = node.pi() if total_num_visits > 0 else None
        state_buffer.append((node.state, value, action_distribution))
        value *= -1
        node = node.parent

    state_buffer = state_buffer[-MAX_BUFFER_SIZE:]
    print(f'Updated buffer, now contains {len(state_buffer)} states')


def update_network(net, optimizer, state_buffer, train_iter, summary_writer):

    # 1. sample a batch from state_buffer
    # 2. do a forward pass
    # 3. post-processing for valid moves?
    # 4. evaluate loss, do .backward()
    # 5. save results, checkpoints

    B = 128

    # Don't update network unless we can make a full batch
    if len(state_buffer) < B:
        return

    sample_indices = np.random.choice(len(state_buffer), B, False)

    # TODO: shared code with node.eval
    # Should maybe also make use of datasets / dataworkers

    x = torch.zeros(B, 2, 6, 7)
    p = torch.zeros(B, 1, 6, 7)
    z = torch.zeros(B, 1)
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
    # Value estimate
    l_v = 0.01 * ((z - v_hat) ** 2).view(B)
    value_loss = torch.mean(l_v)

    # Prior policy
    p = p.view(B, -1)[p_valid]
    p_hat = p_hat.view(B, -1)[p_valid]
    logp_hat = log_softmax(p_hat, dim=1)
    l_p = -torch.sum(p * logp_hat, dim=1)
    prior_loss = torch.mean(l_p)

    total_loss = value_loss + prior_loss

    summary_writer.add_scalar('total_loss', total_loss, train_iter)
    summary_writer.add_scalar('value loss', value_loss, train_iter)
    summary_writer.add_scalar('prior loss', prior_loss, train_iter)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


def alphazero_train(summary_writer):
    # Until converged:
    # 0. Start new game
    # 1. Until game is over:
    #    a. Run MCTS from game state (for 800 expansions)
    #       i.   If game is over in current (simulated) state, backup result
    #       ii.  If state hasn't been visited, evaluate (v, pi) = f(s), backup result
    #       iii. Else, traverse branch of a = argmax U(s,a)
    #    b. Sample move from the state's UCT posterior
    #    c. Update search tree root to reflect new game state
    # 2. Annotate and push game states from game into length B buffer
    # 3. Train network for N batches of K game states

    net = AlphaZeroC4()
    optimizer = SGD(net.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-4)
    state_buffer = [] # TODO: more fancy

    NUM_EXPANSIONS_PER_DECISION = 64

    train_iter = 0
    for _ in range(10000):
        train_iter += 1

        tree = MCTreeNode(ConnectFourState())

        with torch.no_grad():
            # Play one game
            while tree.state.winner is None:
                # Make one move
                for _ in range(NUM_EXPANSIONS_PER_DECISION):
                    tree.expand(net)

                # TODO: Keep track of duration of game, set temperature -> 0 after N moves
                # ^^ From "Self-play" section of AlphaGoZero, double check for this in AlphaZero
                action_index = np.random.choice(len(tree.state.valid_actions), p=tree.pi(1))
                tree = tree.traverse(action_index)
                tree.kill_siblings()

            print(f'Game {train_iter} winner: Player {tree.state.winner}')

        update_databuffer(tree, state_buffer, summary_writer)
        update_network(net, optimizer, state_buffer, train_iter, summary_writer)

        # Checkpoint every 50 models
        if train_iter % 50 == 0:
            torch.save({
                'train_iter': train_iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # TODO: state_buffer state
            }, f'{summary_writer.get_logdir()}/ckpt.{train_iter}.pt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--outdir',
        help="Directory for tensorboard logging and checkpoints")
    parser.add_argument('--debug',
        action='store_true',
        help="Enable debug mode")
    # TODO: Training config
    args = parser.parse_args()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    summary_writer = SummaryWriter(log_dir=args.outdir)
    alphazero_train(summary_writer)
    summary_writer.close()
    print("Done.")