import os
import torch
import numpy as np
import cv2
from torch import nn
from models.convolutional_lstm import ConvLSTM
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
                            return_all_layers=True,
                            device=self.device)
        self.net.to(self.device)
        self.mse_loss = nn.MSELoss()
        self.w_state = 1e-4
        if self.opt.pretrained_model:
            self.load_weight()
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.learning_rate)

    def pooled_batches(self, loader):
        loader_it = iter(loader)
        while True:
            samples = []
            for _ in range(loader.num_workers):
                try:
                    samples.append(next(loader_it))
                except StopIteration:
                    pass
            if len(samples) == 0:
                break
            else:
                feature_lists = []
                # collate all the tensors
                for sample_component_id in range(len(samples[0])):
                    feature_lists.append(torch.cat([sample[sample_component_id] for sample in samples], dim=0))

                yield tuple(feature_lists)

    def train_epoch(self, epoch):
        print("--------------------start training epoch %2d--------------------" % epoch)
        hidden_state = None
        for sample_id, sample in tqdm(enumerate(self.pooled_batches(self.dataloader['train']))):
            self.net.zero_grad()
            time_id, images, actions = sample
            images = images.to(self.device)
            actions = actions.to(self.device)

            if hidden_state is None:
                context_frames = self.opt.context_frames
            else:
                context_frames = 0
            _, sequence_length, _, _, _ = images.size()  # set sequence length correctly

            if (time_id == 0).any() and hidden_state is not None:
                break  # epoch is defined as the end of a set of timelines
                # hidden_state = None  # reset hidden state when timeline resets

            images.requires_grad = True
            gen_images, hidden_state = self.net(images, hidden_state)
            gen_images = gen_images[-1]  # Take only output from final layer

            recon_loss = self.mse_loss(images[:, context_frames+1:, :, :, :],
                                       gen_images[:, context_frames:-1, :, :, :])

            loss = recon_loss/torch.tensor(sequence_length - context_frames)
            loss.backward()
            self.optimizer.step()

            loss.detach()  # detach the graph so that graph does not explode
            torch.cuda.empty_cache()  # empty cuda cache for better memory utilization

            if sample_id % self.opt.print_interval == 0:
                print("training epoch: %3d, iterations: %3d/%3d loss: %6f" %
                      (epoch, sample_id, 30, loss))

    def train(self):
        for epoch_i in range(0, self.opt.epochs):
            self.train_epoch(epoch_i)
            self.evaluate(epoch_i)
            self.save_weight(epoch_i)

    def test(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(os.path.join(self.opt.output_dir, 'video.avi'),
                                fourcc, 1, (self.opt.width, self.opt.height))
        with torch.no_grad():
            hidden_state = None
            losses = []
            for sample_id, sample in tqdm(enumerate(self.pooled_batches(self.dataloader['valid']))):
                time_id, images, actions = sample
                images = images.to(self.device)
                actions = actions.to(self.device)

                if hidden_state is None:
                    context_frames = self.opt.context_frames
                else:
                    context_frames = 0
                _, sequence_length, _, _, _ = images.size()  # set sequence length correctly

                if (time_id == 0).any() and hidden_state is not None:
                    break  # epoch is defined as the end of a set of timelines

                images.requires_grad = True
                gen_images, hidden_state = self.net(images, hidden_state)
                gen_images = gen_images[-1]  # Take only output from final layer

                recon_loss = self.mse_loss(images[:, context_frames + 1:, :, :, :],
                                           gen_images[:, context_frames:-1, :, :, :])

                loss = recon_loss / torch.tensor(sequence_length - context_frames)
                losses.append(loss.cpu().item())

                # add images to video
                for frame_id in range(sequence_length):
                    frame_image = gen_images[0, frame_id, :, :, :].squeeze()
                    frame_image = frame_image.permute(1, 2, 0)
                    frame_image = frame_image.cpu().data.numpy()
                    frame_image = (frame_image*255).astype(np.uint8)
                    video.write(frame_image)
            print("video released")
            video.release()
            print("test recon_loss: %6f" % (np.mean(losses)))


    def evaluate(self, epoch):
        with torch.no_grad():
            hidden_state = None
            losses = []
            for sample_id, sample in tqdm(enumerate(self.pooled_batches(self.dataloader['valid']))):
                time_id, images, actions = sample
                images = images.to(self.device)
                actions = actions.to(self.device)

                if hidden_state is None:
                    context_frames = self.opt.context_frames
                else:
                    context_frames = 0
                _, sequence_length, _, _, _ = images.size()  # set sequence length correctly

                if (time_id == 0).any() and hidden_state is not None:
                    break  # epoch is defined as the end of a set of timelines

                images.requires_grad = True
                gen_images, hidden_state = self.net(images, hidden_state)
                gen_images = gen_images[-1]  # Take only output from final layer

                recon_loss = self.mse_loss(images[:, context_frames + 1:, :, :, :],
                                           gen_images[:, context_frames:-1, :, :, :])

                loss = recon_loss / torch.tensor(sequence_length - context_frames)
                losses.append(loss.cpu().item())
            print("evaluation epoch: %3d, recon_loss: %6f" % (epoch, np.mean(losses)))

    def save_weight(self, epoch):
        torch.save(self.net.state_dict(), os.path.join(self.opt.output_dir, "net_epoch_%d.pth" % epoch))

    def load_weight(self):
        self.net.load_state_dict(torch.load(self.opt.pretrained_model))
