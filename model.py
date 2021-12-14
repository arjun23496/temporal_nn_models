import os
import torch
from torch import nn
from convolutional_lstm import ConvLSTM
from data import build_dataloader
from torch.nn import functional as F

from tqdm import tqdm


def peak_signal_to_noise_ratio(true, pred):
    return 10.0 * torch.log(torch.tensor(1.0) / F.mse_loss(true, pred)) / torch.log(torch.tensor(10.0))


class Model():
    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device

        train_dataloader, valid_dataloader = build_dataloader(opt)
        self.dataloader = {'train': train_dataloader, 'valid': valid_dataloader}

        self.net = ConvLSTM(input_dim=self.opt.channels,
                            hidden_dim=[16, 3],
                            kernel_size=(5, 5),
                            num_layers=2,
                            batch_first=True,
                            bias=True,
                            return_all_layers=True)
        self.net.to(self.device)
        self.mse_loss = nn.MSELoss()
        self.w_state = 1e-4
        if self.opt.pretrained_model:
            self.load_weight()
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.learning_rate)

    def train_epoch(self, epoch):
        print("--------------------start training epoch %2d--------------------" % epoch)
        hidden_state = None
        for sample_id, sample in tqdm(enumerate(self.dataloader['train'])):
            self.net.zero_grad()
            time_id, images, actions = sample
            print("time: ", time_id)
            if (time_id == 0).any():
                hidden_state = None  # reset hidden state when timeline resets

            images.requires_grad = True
            gen_images, hidden_state = self.net(images, hidden_state)
            gen_images = gen_images[-1]  # Take only output from final layer

            recon_loss = self.mse_loss(images[:, self.opt.context_frames:, :, :, :],
                                       gen_images[:, self.opt.context_frames-1:-1, :, :, :])

            loss = recon_loss/torch.tensor(self.opt.sequence_length - self.opt.context_frames)
            loss.backward()
            self.optimizer.step()

            loss.detach()  # detach the graph so that graph does not explode

            if sample_id % self.opt.print_interval == 0:
                print("training epoch: %3d, iterations: %3d/%3d loss: %6f" %
                      (epoch, sample_id, 30, loss))

    def train(self):
        for epoch_i in range(0, self.opt.epochs):
            self.train_epoch(epoch_i)
            self.evaluate(epoch_i)
            self.save_weight(epoch_i)

    def evaluate(self, epoch):
        with torch.no_grad():
            hidden_state = None
            for sample_id, sample in tqdm(enumerate(self.dataloader['val'])):
                self.net.zero_grad()
                images, actions = sample
                images.requires_grad = True
                print(images.size())
                gen_images, hidden_state = self.net(images, hidden_state)
                gen_images = gen_images[-1]  # Take only output from final layer

                recon_loss = self.mse_loss(images[:, self.opt.context_frames:, :, :, :],
                                           gen_images[:, self.opt.context_frames-1:-1, :, :, :])

                loss = recon_loss/torch.tensor(self.opt.sequence_length - self.opt.context_frames)

            print("evaluation epoch: %3d, recon_loss: %6f" % (epoch, loss))

    def save_weight(self, epoch):
        torch.save(self.net.state_dict(), os.path.join(self.opt.output_dir, "net_epoch_%d.pth" % epoch))

    def load_weight(self):
        self.net.load_state_dict(torch.load(self.opt.pretrained_model))
