import os
import torch
import numpy as np
import cv2
import scipy.misc
from PIL import Image

from torch import nn
from models.convolutional_lstm import ConvLSTM
from data import build_dataloader
from torch.nn import functional as F

from tqdm import tqdm


def mse_loss(true, pred):
    return np.mean((true-pred)**2)


def peak_signal_to_noise_ratio(true, pred):
    return 10.0 * np.log(1.0 / mse_loss(true, pred)) / np.log(10)


class Model():
    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device

        train_dataloader, valid_dataloader, test_dataloader = build_dataloader(opt)
        self.dataloader = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}

        self.net = ConvLSTM(image_width=self.opt.width,
                            image_height=self.opt.height,
                            input_dim=self.opt.channels,
                            num_actions=self.opt.num_actions,
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
            gen_images, action_output, hidden_state = self.net(images, hidden_state)
            gen_images = gen_images[-1]  # Take only output from final layer

            recon_loss = self.mse_loss(images[:, context_frames+1:, :, :, :],
                                       gen_images[:, context_frames:-1, :, :, :])

            action_output = F.softmax(action_output, dim=1)
            action_output = action_output.reshape(actions.size())
            avg_recon_loss = recon_loss/torch.tensor(sequence_length - context_frames)
            avg_action_loss = -torch.mean(torch.sum(actions[:, context_frames + 1:, :] *
                                                        torch.log(action_output[:, context_frames:-1, :]), dim=2))

            print("reconstruction: {} action classification: {}".format(avg_recon_loss,
                                                                        avg_action_loss))

            loss = self.opt.recon_loss_weight*avg_recon_loss + \
                   self.opt.action_classification_loss_weight*avg_action_loss

            loss.backward()
            self.optimizer.step()

            loss.detach()  # detach the graph so that graph does not explode
            torch.cuda.empty_cache()  # empty cuda cache for better memory utilization

            if sample_id % self.opt.print_interval == 0:
                print("training epoch: %3d, iterations: %3d/%3d loss: %6f" %
                      (epoch, sample_id, 30, loss))

    def write_diagnostic(self, image, text, org=(10, 10)):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # fontScale
        fontScale = 0.2

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 1

        image = cv2.putText(image, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        return image

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
            recon_losses = []
            class_losses = []
            accuracies = []

            for sample_id, sample in tqdm(enumerate(self.pooled_batches(self.dataloader['test']))):
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
                gen_images, action_output, hidden_state = self.net(images, hidden_state)
                action_output = F.softmax(action_output, dim=1)
                action_output = action_output.reshape(actions.size())
                gen_images = gen_images[-1]  # Take only output from final layer

                recon_loss = self.mse_loss(images[:, context_frames + 1:, :, :, :],
                                           gen_images[:, context_frames:-1, :, :, :])

                avg_recon_loss = recon_loss / torch.tensor(sequence_length - context_frames)
                avg_action_loss = -torch.mean(torch.sum(actions[:, context_frames + 1:, :] *
                                                        torch.log(action_output[:, context_frames: -1, :]), dim=2))

                recon_losses.append(avg_recon_loss.cpu().item())
                class_losses.append(avg_action_loss.cpu().item())
                accuracies.append(torch.mean((torch.argmax(actions,
                                                          dim=2) == torch.argmax(action_output,
                                                                                 dim=2)).type(torch.FloatTensor)).cpu().item())

                # add images to video
                for frame_id in range(sequence_length):
                    frame_image = gen_images[0, frame_id, :, :, :].squeeze()
                    frame_actions = torch.argsort(action_output[0, frame_id, :].squeeze(), descending=True)
                    frame_actions = [ self.dataloader["test"].dataset.id_actions_map[id.cpu().item()] for id in frame_actions ]
                    frame_image = frame_image.permute(1, 2, 0)
                    frame_image = frame_image.cpu().data.numpy()

                    ground_truth = images[0, frame_id, :, :, :].squeeze()
                    ground_truth = ground_truth.permute(1, 2, 0)
                    ground_truth = ground_truth.cpu().data.numpy()

                    # diagnostic information
                    frame_mse = mse_loss(ground_truth, frame_image)
                    frame_psnr = peak_signal_to_noise_ratio(ground_truth, frame_image)

                    # convert frames to int8
                    frame_image = (frame_image * 255).astype(np.uint8)
                    ground_truth = (ground_truth * 255).astype(np.uint8)

                    frame_image = self.write_diagnostic(np.ascontiguousarray(frame_image, dtype=np.uint8),
                                                        "recon: {:.6f}, "
                                                        "class: {:.6f}".format(frame_mse,
                                                                               frame_psnr),
                                                        org=(10, 10))

                    frame_image = self.write_diagnostic(np.ascontiguousarray(frame_image, dtype=np.uint8),
                                                        " | ".join(frame_actions),
                                                        org=(15, 15))

                    # write video to file
                    video.write(frame_image)

            print("video released")
            video.release()
            print("test recon_loss: %6f" % (np.mean(recon_losses)))
            print("classification loss: %6f" % (np.mean(class_losses)))
            print("average_accuracy: %6f" % (np.mean(accuracies)))


    def evaluate(self, epoch):
        with torch.no_grad():
            hidden_state = None
            recon_losses = []
            action_losses = []
            losses = []
            accuracies = []
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
                gen_images, action_output, hidden_state = self.net(images, hidden_state)
                action_output = F.softmax(action_output, dim=1)
                action_output = action_output.reshape(actions.size())

                gen_images = gen_images[-1]  # Take only output from final layer

                recon_loss = self.mse_loss(images[:, context_frames + 1:, :, :, :],
                                           gen_images[:, context_frames:-1, :, :, :])

                avg_recon_loss = recon_loss / torch.tensor(sequence_length - context_frames)
                avg_action_loss = -torch.mean(torch.sum(actions[:, context_frames + 1:, :] *
                                                        torch.log(action_output[:, context_frames: -1, :]), dim=2))

                loss = self.opt.recon_loss_weight * avg_recon_loss + \
                       self.opt.action_classification_loss_weight * avg_action_loss

                recon_losses.append(avg_recon_loss.cpu().item())
                action_losses.append(avg_action_loss.cpu().item())
                accuracies.append(torch.mean((torch.argmax(actions,
                                                          dim=2) == torch.argmax(action_output,
                                                                                 dim=2)).type(torch.FloatTensor)).cpu().item())

                losses.append(loss.cpu().item())

            print("evaluation epoch: %3d, recon_loss: %6f classification_loss: %6f"
                  " accuracy: %6f" % (epoch,
                                      np.mean(recon_losses),
                                      np.mean(action_losses),
                                      np.mean(accuracies)))

    def save_weight(self, epoch):
        torch.save(self.net.state_dict(), os.path.join(self.opt.output_dir, "net_epoch_%d.pth" % epoch))

    def load_weight(self):
        self.net.load_state_dict(torch.load(self.opt.pretrained_model))
