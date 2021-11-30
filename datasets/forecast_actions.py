import re

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

import os
import sys

# user defined
from modules import constants
from modules.spurious_actions import SpuriousActionsManager
from modules.timed_actions import TimedActionsManager
from modules.utils import get_duration

import datasets.modules.constants as constants


class Sims4ActionDataset(torch.utils.data.IterableDataset):
    def __init__(self, timed_action_manager, spurious_action_manager, house=1, room="Living", subject=1):
        super(Sims4ActionDataset).__init__()
        self.action_managers = [
            timed_action_manager,
            spurious_action_manager
        ]
        self.timeline = []
        self.house = house
        self.room = room
        self.subject = subject

    def build_timeline(self):
        time = constants.MIN_TIME
        room = self.room

        # reset action managers prior to building timeline
        for action_manager in self.action_managers:
            action_manager.reset()

        while time < constants.MAX_TIME:
            print(time)
            for action_manager in self.action_managers:
                add_status, time, room, self.timeline = action_manager.update_timeline(self.house, self.subject,
                                                                                       room, time, self.timeline)
                print(room)
                if add_status:
                    break

    def __iter__(self):
        # while True:
        self.build_timeline()

        for action_item in self.timeline:
            time, room, camera_angle, action = action_item
            yield time, room, camera_angle, action


timed_actions = {
    "08:00:00": ("Kitchen", "Cook"),
    "08:30:00": ("Dining", "Eat"),
    "08:45:00": (None, "Drink"),
    "09:30:00": ("Kitchen", "Cook"),
    "09:45:00": ("Dining", "Eat"),
    "10:00:00": (None, "Drink"),
    "10:30:00": (None, "Drink"),
    "11:00:00": ("Kitchen", "Cook"),
    "11:30:00": ("Dining", "Eat")
}

event_driven_actions = {}

spurious_actions = {
    "Readbook": 0.2,
    "Usecomputer": 0.2,
    "Usephone": 0.2,
    "Usetablet": 0.2,
    "WatchTV": 0.2
}

action_df = pd.read_csv(os.path.join("/media/arjun/Shared/research_projects/temporal_weight_gating/Sims4ActionVideos",
                                     "SimsSplitsCompleteVideos.csv"), sep=";")

timed_action_manager = TimedActionsManager(action_df)
spurious_action_manager = SpuriousActionsManager(action_df, available_rooms=["Living", "Dining", "Kitchen"])

for time, action in timed_actions.items():
    timed_action_manager.add_action(action, time)

for action, prob in spurious_actions.items():
    spurious_action_manager.add_action(action, prob)

ds = Sims4ActionDataset(timed_action_manager,
                        spurious_action_manager)

id = 0
for dp in ds:
    id += 1
    time, room, camera_angle, action = dp
    print("{} {} fC{} {}".format(time, room, camera_angle, action))
    # if id > 1000:
    #     break
