import re

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

import os
import sys
import cv2

# user defined
from datasets.modules import constants
from datasets.modules.spurious_actions import SpuriousActionsManager
from datasets.modules.timed_actions import TimedActionsManager
from datasets.modules.utils import get_duration

import datasets.modules.constants as constants


class Sims4ActionDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_root, timed_action_manager, spurious_action_manager, actions_list, image_transform=None,
                 house=1, room="Living", subject=1, device='cpu', frames_per_clip=350):
        super(Sims4ActionDataset).__init__()
        self.actions_list = actions_list
        self.actions_id_map = { action: id for id, action in enumerate(self.actions_list) }
        self.action_managers = [
            timed_action_manager,
            spurious_action_manager
        ]
        self.data_root = data_root
        # self.timeline = []
        self.house = house
        self.room = room
        self.subject = subject
        self.batch_size = 1
        self.image_transform = image_transform
        self.frames_per_clip = frames_per_clip
        self.device = device

    def build_timeline(self):
        # torch.manual_seed(index)
        # np.random.seed(index)
        timeline = []
        time = constants.MIN_TIME
        room = self.room

        # reset action managers prior to building timeline
        for action_manager in self.action_managers:
            action_manager.reset()

        while time < constants.MAX_TIME:
            for action_manager in self.action_managers:
                add_status, time, room, timeline = action_manager.update_timeline(self.house, self.subject,
                                                                                  room, time, timeline)
                if add_status:
                    break

        return timeline

    def __iter__(self):
        # while True:
        timelines = []
        time = []
        min_actions = -1
        for i in range(self.batch_size):
            timelines.append(self.build_timeline())

            if min_actions == -1 or len(timelines[-1]) < min_actions:
                min_actions = len(timelines[-1])

        batch = []
        for action_idx in range(min_actions):
            batch_infos = []
            vidcaps = []
            for sample_id in range(self.batch_size):
                batch_infos.append(timelines[sample_id][action_idx])
                _, _, _, action, filename = batch_infos[-1]
                vidcaps.append(cv2.VideoCapture(os.path.join(self.data_root,
                                                             action,
                                                             filename)))

            success = []
            images = []
            for sample_id in range(self.batch_size):
                success_sample, image_sample = vidcaps[sample_id].read()
                success.append(success_sample)
                images.append(image_sample)

            success = np.array(success)

            image_intime = []
            action_intime = []
            while np.all(success):
                batch = []
                for sample_id in range(self.batch_size):
                    success_sample, image_sample = vidcaps[sample_id].read()
                    success[sample_id] = success_sample

                    if success_sample:
                        time0, room, camera_angle, action, filename = batch_infos[sample_id]
                        action_array = np.zeros(len(self.actions_list))
                        action_array[self.actions_id_map[action]] = 1
                        action_array = torch.from_numpy(action_array).to(self.device)
                        image_sample = self.image_transform(image_sample).to(self.device)

                        image_intime.append(image_sample)
                        action_intime.append(action_array)
                    # batch.append((action_array, image_sample))

                # intime.append(batch[0])
                # yield batch[0]
            image_intime = image_intime[:self.frames_per_clip]
            action_intime = action_intime[:self.frames_per_clip]
            image_intime = torch.cat([img.unsqueeze(0) for img in image_intime], dim=0)
            action_intime = torch.cat([action.unsqueeze(0) for action in action_intime], dim=0)
            # print("image ", image_intime.size())
            # print("action ", action_intime.size())
            # image_intime = torch.FloatTensor(image_intime)
            # action_intime = torch.FloatTensor(action_intime)
            yield image_intime, action_intime

# timed_actions = {
#     "08:00:00": ("Kitchen", "Cook"),
#     "08:30:00": ("Dining", "Eat"),
#     "08:45:00": (None, "Drink"),
#     "09:30:00": ("Kitchen", "Cook"),
#     "09:45:00": ("Dining", "Eat"),
#     "10:00:00": (None, "Drink"),
#     "10:30:00": (None, "Drink"),
#     "11:00:00": ("Kitchen", "Cook"),
#     "11:30:00": ("Dining", "Eat")
# }
#
# event_driven_actions = {}
#
# spurious_actions = {
#     "Readbook": 0.2,
#     "Usecomputer": 0.2,
#     "Usephone": 0.2,
#     "Usetablet": 0.2,
#     "WatchTV": 0.2
# }
#
# action_df = pd.read_csv(os.path.join("/media/arjun/Shared/research_projects/temporal_weight_gating/Sims4ActionVideos",
#                                      "SimsSplitsCompleteVideos.csv"), sep=";")
#
# timed_action_manager = TimedActionsManager(action_df)
# spurious_action_manager = SpuriousActionsManager(action_df, available_rooms=["Living", "Dining", "Kitchen"])
#
# for time, action in timed_actions.items():
#     timed_action_manager.add_action(action, time)
#
# for action, prob in spurious_actions.items():
#     spurious_action_manager.add_action(action, prob)
#
# ds = Sims4ActionDataset(timed_action_manager,
#                         spurious_action_manager)
#
# id = 0
# for dp in ds:
#     id += 1
#     time, room, camera_angle, action = dp
#     print("{} {} fC{} {}".format(time, room, camera_angle, action))
#     # if id > 1000:
#     #     break
