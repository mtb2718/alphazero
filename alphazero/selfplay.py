from alphazero.mcts import MCTreeNode, run_mcts


class SelfPlayWorker:
    def __init__(self, config, model_server, dataset):
        self._config = config
        self._model_server = model_server
        self._dataset = dataset

    def play_game(self):
        game = self._config.Game()
        tree = MCTreeNode()

        model_version, model = self._model_server.latest()

        while not game.terminal:
            run_mcts(game, tree, model, self._config.selfplay['num_expansions_per_sim'])
            if len(game) < self._config.selfplay['exploration_depth']:
                action_index = tree.sample_action()
            else:
                action_index = tree.greedy_action()
            action = game.valid_actions[action_index]
            game.take_action(action, search_statistics=tree.num_visits)

            # pseudocode in supplement suggests False, paper says True
            if self._config.selfplay['reuse_search_tree']:
                tree = tree.traverse(action_index)
                tree.kill_siblings()
            else:
                tree = MCTreeNode()

        print(f'Player {game.winner} (v{model_version}) wins game after {len(game.history)} turns')
        self._dataset.add_game(game, model_version)
