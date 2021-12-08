import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

from datasets.modules import constants
from datasets.modules.spurious_actions import SpuriousActionsManager
from datasets.modules.timed_actions import TimedActionsManager
from datasets.modules.utils import get_duration
from datasets.forecast_actions import Sims4ActionDataset

import matplotlib.pyplot as plt

IMG_EXTENSIONS = ('.npy',)


def make_dataset(path):
    timed_actions = {
        "00:05:00": ("Kitchen", "Cook"),
        "00:08:00": ("Dining", "Eat"),
        "00:10:00": (None, "Drink"),
        "00:15:00": ("Kitchen", "Cook"),
        "00:18:00": ("Dining", "Eat"),
        "00:20:00": (None, "Drink"),
    }

    event_driven_actions = {}

    spurious_actions = {
        "Readbook": 0.2,
        "Usecomputer": 0.2,
        "Usephone": 0.2,
        "Usetablet": 0.2,
        "WatchTV": 0.2
    }

    action_df = pd.read_csv(os.path.join(path, "SimsSplitsCompleteVideos.csv"), sep=";")

    timed_action_manager = TimedActionsManager(action_df)
    spurious_action_manager = SpuriousActionsManager(action_df, available_rooms=["Living", "Dining", "Kitchen"])

    for time, action in timed_actions.items():
        timed_action_manager.add_action(action, time)

    for action, prob in spurious_actions.items():
        spurious_action_manager.add_action(action, prob)

    ds = Sims4ActionDataset(path,
                            timed_action_manager,
                            spurious_action_manager)

    return ds


def npy_loader(path):
    samples = torch.from_numpy(np.load(path))
    return samples


class PushDataset(Dataset):
    def __init__(self, root, image_transform=None, action_transform=None, state_transform=None, loader=npy_loader, device='cpu'):
        if not os.path.exists(root):
            raise FileExistsError('{0} does not exists!'.format(root))
        # self.subfolders = [f[0] for f in os.walk(root)][1:]
        self.image_transform = image_transform
        self.action_transform = action_transform
        self.state_transform = state_transform
        self.dataset = make_dataset(root)
        # if len(self.samples) == 0: raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n" "Supported
        # image extensions are: " + ",".join( IMG_EXTENSIONS)))
        self.loader = loader
        self.device = device

    def __getitem__(self, index):
        count = 0

        # for ds in self.dataset:
        #     plt.imshow(ds[-1])
        #     plt.show()
        #     count += 1
        #     if count > 10:
        #         break
        #
        # sys.exit()

        for ds in self.dataset:
            time, room, camera_angle, action, image = ds

            yield torch.from_numpy(image), torch.zeros(10), torch.zeros(10)

        # image, action, state = self.samples[index]
        # image, action, state = self.loader(image), self.loader(action), self.loader(state)
        #
        # if self.image_transform is not None:
        #     image = torch.cat([self.image_transform(single_image).unsqueeze(0) for single_image in image.unbind(0)], dim=0)
        # if self.action_transform is not None:
        #     action = torch.cat([self.action_transform(single_action).unsqueeze(0) for single_action in action.unbind(0)], dim=0)
        # if self.state_transform is not None:
        #     state = torch.cat([self.state_transform(single_state).unsqueeze(0) for single_state in state.unbind(0)], dim=0)

        # return image.to(self.device), action.to(self.device), state.to(self.device)

    def __len__(self):
        return 10


def build_dataloader(opt):
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.height, opt.width)),
        transforms.ToTensor()
    ])

    train_ds = PushDataset(
        root=opt.data_dir,
        image_transform=image_transform,
        loader=npy_loader,
        device=opt.device
    )

    testseen_ds = PushDataset(
        root=opt.data_dir,
        image_transform=image_transform,
        loader=npy_loader,
        device=opt.device
    )

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    testseen_dl = DataLoader(dataset=testseen_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    return train_dl, testseen_dl

