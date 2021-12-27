import os
import sys
import torch
import yaml
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


def make_dataset(path, image_transform, opt):

    with open(opt.environment_config, 'r') as f:
        try:
            environment_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()

    timed_actions = environment_config["timed_actions"]

    event_driven_actions = {}

    actions_list = []

    spurious_actions = environment_config["spurious_actions"]

    action_df = pd.read_csv(os.path.join(path, "SimsSplitsCompleteVideos.csv"), sep=";")

    timed_action_manager = TimedActionsManager(action_df,
                                               min_time=environment_config["min_time"],
                                               max_time=environment_config["max_time"])
    spurious_action_manager = SpuriousActionsManager(action_df,
                                                     available_rooms=environment_config["rooms"],
                                                     min_time=environment_config["min_time"],
                                                     max_time=environment_config["max_time"])

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
                            image_transform=image_transform,
                            min_time=environment_config["min_time"],
                            max_time=environment_config["max_time"])

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

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size // opt.batch_size, num_workers=opt.batch_size,
                          drop_last=False)
    validation_dl = DataLoader(dataset=test_ds, batch_size=opt.batch_size // opt.batch_size, num_workers=opt.batch_size,
                             drop_last=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=opt.batch_size // opt.batch_size, num_workers=opt.batch_size,
                               drop_last=False)

    return train_dl, validation_dl, test_dl
