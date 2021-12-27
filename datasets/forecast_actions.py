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
                 house=1, room="Living", subject=1, device='cpu', frames_per_clip=320, skip_frequency=4,
                 max_time="00:30:00", min_time="00:00:00"):
        super(Sims4ActionDataset).__init__()
        self.actions_list = actions_list
        self.actions_id_map = { action: id for id, action in enumerate(self.actions_list) }
        self.action_managers = [
            timed_action_manager,
            spurious_action_manager
        ]
        self.data_root = data_root
        self.house = house
        self.room = room
        self.subject = subject
        self.batch_size = 1
        self.image_transform = image_transform
        self.frames_per_clip = frames_per_clip
        self.skip_frequency = skip_frequency
        self.device = device
        self.MAX_TIME = datetime.strptime(max_time, constants.TIME_FORMAT)
        self.MIN_TIME = datetime.strptime(min_time, constants.TIME_FORMAT)

    def build_timeline(self):
        timeline = []
        time = self.MIN_TIME
        room = self.room

        # reset action managers prior to building timeline
        for action_manager in self.action_managers:
            action_manager.reset()

        while time < self.MAX_TIME:
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
                        action_array = torch.from_numpy(action_array)
                        image_sample = self.image_transform(image_sample)

                        image_intime.append(image_sample)
                        action_intime.append(action_array)

            image_intime = image_intime[:self.frames_per_clip]
            action_intime = action_intime[:self.frames_per_clip]

            while len(image_intime) < self.frames_per_clip:
                image_intime.append(image_intime[-1].clone())
                action_intime.append(action_intime[-1].clone())

            image_intime = image_intime[::self.skip_frequency]
            action_intime = action_intime[::self.skip_frequency]

            image_intime = torch.cat([img.unsqueeze(0) for img in image_intime], dim=0)
            action_intime = torch.cat([action.unsqueeze(0) for action in action_intime], dim=0)
            yield torch.FloatTensor([action_idx]), image_intime.to(self.device), action_intime.to(self.device)
