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


def make_dataset(path, image_transform, opt):
    timed_actions = {
        "00:05:00": ("Kitchen", "Cook"),
        "00:08:00": ("Dining", "Eat"),
        "00:10:00": (None, "Drink"),
        "00:15:00": ("Kitchen", "Cook"),
        "00:18:00": ("Dining", "Eat"),
        "00:20:00": (None, "Drink"),
    }

    event_driven_actions = {}

    actions_list = []
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
        if action[1] not in actions_list:
            actions_list.append(action[1])
        timed_action_manager.add_action(action, time)

    for action, prob in spurious_actions.items():
        if action not in actions_list:
            actions_list.append(action)
        spurious_action_manager.add_action(action, prob)

    ds = Sims4ActionDataset(path,
                            timed_action_manager,
                            spurious_action_manager,
                            actions_list,
                            image_transform=image_transform)

    return ds


def npy_loader(path):
    samples = torch.from_numpy(np.load(path))
    return samples




def build_dataloader(opt):
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.height, opt.width)),
        transforms.ToTensor()
    ])

    train_ds = make_dataset(opt.data_dir, image_transform, opt)
    test_ds = make_dataset(opt.data_dir, image_transform, opt)

    # for batch in train_ds:
    #     # sample =
    #     print(batch[0][0])
    #     plt.imshow(np.swapaxes(batch[0][1].data.numpy(), 0, 2))
    #     plt.show()
    #     break

    # sys.exit()

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, drop_last=False, num_workers=opt.batch_size)
    testseen_dl = DataLoader(dataset=test_ds, batch_size=opt.batch_size, drop_last=False, num_workers=opt.batch_size)
    return train_dl, testseen_dl

