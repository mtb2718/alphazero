import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from alphazero.config import AlphaZeroConfig
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
            return False
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
        prior_loss, value_loss = self._loss(p, z, p_hat, v_hat, p_valid)
        self._optimizer.zero_grad()
        total_loss = prior_loss + value_loss
        total_loss.backward()
        self._optimizer.step()
        self._lr_schedule.step()

        # Publish latest model
        self._model_server.update(self._model, train_iter + 1)

        # Log training stats
        self._summary_writer.add_scalar('loss/prior', prior_loss, train_iter)
        self._summary_writer.add_scalar('loss/value', value_loss, train_iter)
        self._summary_writer.add_scalar('loss/total', total_loss, train_iter)
        self._summary_writer.add_scalar('learning_rate', self._lr_schedule.get_last_lr()[0], train_iter)

        # Log eval stats
        solved = batch['solved']
        if torch.any(solved):
            with torch.no_grad():
                move_scores = batch['move_scores'][solved].to(self._device)
                v_star = batch['v*'][solved].to(self._device)
                p_star = F.softmax(move_scores, dim=1)

                targ_prior_loss, targ_value_loss = self._loss(p_star, v_star, p[solved], z[solved], p_valid[solved])
                pred_prior_loss, pred_value_loss = self._loss(p_star, v_star, p_hat[solved], v_hat[solved], p_valid[solved])
                self._summary_writer.add_scalar('eval/target_value_loss', targ_value_loss, train_iter)
                self._summary_writer.add_scalar('eval/target_prior_loss', targ_prior_loss, train_iter)
                self._summary_writer.add_scalar('eval/prediction_value_loss', pred_value_loss, train_iter)
                self._summary_writer.add_scalar('eval/prediction_prior_loss', pred_prior_loss, train_iter)

                # ^^ Above are eval of current batch
                # TODO: Add separate eval for specific opening / end game states

        # System state logging
        Gtotal = self._dataset.db.num_games()
        Gdistinct = self._dataset.db.num_distinct_games()
        L = self._dataset.db.num_states()
        self._summary_writer.add_scalar('data/total_games', Gtotal, train_iter)
        self._summary_writer.add_scalar('data/distinct_games', Gdistinct, train_iter)
        self._summary_writer.add_scalar('data/total_states', L, train_iter)

        return True
