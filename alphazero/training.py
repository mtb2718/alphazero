import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from alphazero.config import AlphaZeroConfig
from alphazero.loss import (
    masked_cross_entropy,
    masked_kl_div)

from alphazero.replay_buffer import ReplayBufferSampler


class TrainingWorker:
    def __init__(self, config, model_server, dataset, summary_writer, device, num_workers):

        self._config = config
        self._model_server = model_server
        self._dataset = dataset
        self._summary_writer = summary_writer
        self._device = device
        self._num_workers = num_workers

        self._model = config.Model().to(device)
        self._loss = config.Loss()

        self._optimizer = SGD(self._model.parameters(),
                              lr=config.training['lr'],
                              momentum=config.training['momentum'],
                              weight_decay=config.training['weight_decay'])
        self._lr_schedule = MultiStepLR(self._optimizer,
                                        config.training['lr_schedule'],
                                        gamma=config.training['lr_schedule_gamma'])
        # TODO: Replay buffer settings from config
        self._sampler = ReplayBufferSampler(dataset,
                                            batch_size=config.training['batch_size'],
                                            bufferlen=(64 * 1024))
        self._dataloader = None
        self._batch_iter = None

    def process_batch(self):

        # Don't update network unless we can make a full batch
        N = len(self._dataset)
        B = self._config.training['batch_size']
        M = max(2 * self._num_workers, 1) * B
        if N < M:
            print(f"Skipping batch, too few examples in replay buffer ({N} < {M}).")
            return 0
        elif self._dataloader is None:
            self._dataloader = DataLoader(self._dataset,
                                        batch_sampler=self._sampler,
                                        pin_memory=True,
                                        num_workers=self._num_workers)
            self._batch_iter = enumerate(self._dataloader)

        # Run inference, evaluate loss, backprop
        self._model.train()
        train_iter, batch = next(self._batch_iter)
        x = batch['x'].to(self._device)
        p = batch['p'].to(self._device)
        z = batch['z'].to(self._device)
        p_valid = batch['p_valid'].to(self._device)
        p_hat, v_hat = self._model(x, p_valid)
        policy_loss, value_loss = self._loss(p, z, p_hat, v_hat, p_valid)
        self._optimizer.zero_grad()
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self._optimizer.step()
        self._lr_schedule.step()

        # Publish latest model
        self._model_server.update(self._model, train_iter + 1)

        # Log training stats
        self._summary_writer.add_scalar('loss/policy', policy_loss, train_iter)
        self._summary_writer.add_scalar('loss/value', value_loss, train_iter)
        self._summary_writer.add_scalar('loss/total', total_loss, train_iter)
        self._summary_writer.add_scalar('optim/learning_rate', self._lr_schedule.get_last_lr()[0], train_iter)

        if train_iter % 10 == 0:
            print(f'Step: {train_iter}, policy loss: {policy_loss}, value_loss: {value_loss}, total loss: {total_loss}')

        # Log eval stats
        solved = batch['solved']
        if torch.any(solved):
            with torch.no_grad():
                v_star = batch['v*'][solved].to(self._device)
                move_scores = batch['move_scores'][solved].to(self._device)
                move_scores[~p_valid[solved]] = -float('inf')
                p_star = F.softmax(move_scores, dim=1)

                # KL divergence of pseudo-optimal policy and target, predicted policy
                logp = torch.log(p)
                KL_policy_target = masked_kl_div(logp[solved], p_star, p_valid[solved]).mean()
                KL_policy_prediction = masked_kl_div(p_hat[solved], p_star, p_valid[solved]).mean()
                self._summary_writer.add_scalar('eval/policy_target_KL', KL_policy_target, train_iter)
                self._summary_writer.add_scalar('eval/policy_prediction_KL', KL_policy_prediction, train_iter)

                # True MSE of target & predicted state values
                mse_target_value = ((v_star - z[solved]) ** 2).mean()
                mse_predicted_value = ((v_star - v_hat[solved]) ** 2).mean()
                self._summary_writer.add_scalar('eval/value_target_mse', mse_target_value, train_iter)
                self._summary_writer.add_scalar('eval/value_prediction_mse', mse_predicted_value, train_iter)

                # Entropy of target policy, from MCTS
                Ht = masked_cross_entropy(logp, p, p_valid).mean()
                self._summary_writer.add_scalar('eval/policy_target_entropy', Ht, train_iter)

                # Entropy of policy prediction
                Hp = masked_cross_entropy(p_hat, F.softmax(p_hat, dim=1), p_valid).mean()
                self._summary_writer.add_scalar('eval/policy_prediction_entropy', Hp, train_iter)

                # TODO: Add separate eval for specific opening / end game states

        # System state logging
        Gtotal = self._dataset.db.num_games()
        Gdistinct = self._dataset.db.num_distinct_games()
        L = self._dataset.db.num_states()
        self._summary_writer.add_scalar('data/total_games', Gtotal, train_iter)
        self._summary_writer.add_scalar('data/distinct_games', Gdistinct, train_iter)
        self._summary_writer.add_scalar('data/total_states', L, train_iter)

        return train_iter + 1
