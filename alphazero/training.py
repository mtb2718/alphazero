import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from alphazero.config import AlphaZeroConfig
from alphazero.replay_buffer import ReplayBufferSampler


class TrainingWorker:
    def __init__(self, config, model_server, dataset, summary_writer, device):

        self._config = config
        self._summary_writer = summary_writer
        self._model = config.Model().to(device)
        self._device = device
        self._model_server = model_server
        self._loss = config.Loss()

        self._optimizer = SGD(self._model.parameters(),
                              lr=config.training['lr'],
                              momentum=config.training['momentum'],
                              weight_decay=config.training['weight_decay'])
        self._lr_schedule = MultiStepLR(self._optimizer,
                                        config.training['lr_schedule'],
                                        gamma=config.training['lr_schedule_gamma'])
        self._dataset = dataset
        # TODO: Replay buffer settings from config
        self._sampler = ReplayBufferSampler(dataset,
                                            batch_size=config.training['batch_size'],
                                            bufferlen=(64 * 1024))
        self._dataloader = DataLoader(dataset,
                                      batch_sampler=self._sampler,
                                      pin_memory=True,
                                      num_workers=0)
        self._batch_iter = enumerate(self._dataloader)

    def process_batch(self):

        # Don't update network unless we can make a full batch
        N = len(self._dataset)
        B = self._config.training['batch_size']
        if N < B:
            print(f"Skipping batch (batch_size={B}), too few examples in replay buffer ({N}).")
            return False

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
        if torch.any(batch['solved']):
            with torch.no_grad():
                move_scores = batch['move_scores']
                p_star = F.softmax(move_scores, dim=1)
                v_star = batch['v*']
                targ_prior_loss, targ_value_loss = self._loss(p_star, v_star, p, z, p_valid)
                pred_prior_loss, pred_value_loss = self._loss(p_star, v_star, p_hat, v_hat, p_valid)
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
