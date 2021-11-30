import re

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

import os
import sys


class Sims4ActionDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, timed_actions, event_driven_actions, spurious_actions, current_subject=1,
                 spurious_action_probability=0.9, current_house=1, fixed_camera=True, idle_action="WatchTV",
                 room_change_probability=0.1,
                 maxtime="12:00:00"):
        super(Sims4ActionDataset).__init__()
        self.timed_actions = timed_actions
        self.event_driven_actions = event_driven_actions
        self.spurious_actions = spurious_actions
        self.spurious_action_probability = spurious_action_probability
        self.fixed_camera = fixed_camera
        self.data_path = data_path

        self.TIME_ZERO = datetime.strptime("00:00:00", "%H:%M:%S")
        self.current_time = datetime.strptime("00:00:00", "%H:%M:%S")
        self.maxtime = datetime.strptime(maxtime, "%H:%M:%S")
        self.current_house = current_house
        self.current_subject = current_subject
        self.current_room = "Living"
        self.idle_action = idle_action
        self.action_queue = []
        self.action_df = pd.read_csv(os.path.join(self.data_path, "SimsSplitsCompleteVideos.csv"), sep=";")
        self.camera_angles = {
            "Dining": {
                "1": [1, 2, 3, 4],
                "2": [13, 14, 15, 16]
            },
            "Kitchen": {
                "1": [5, 6, 7, 8],
                "2": [17, 18, 19, 20]
            },
            "Living": {
                "1": [9, 10, 11, 12],
                "2": [21, 22, 23, 24]
            }
        }
        self.rooms = list(self.camera_angles.keys())
        self.room_change_probability = room_change_probability

        self.action_map = {
            "Cook": "Co",
            "Eat": "Ea",
            "Drink": "Dr",
            "Readbook": "RB",
            "Usecomputer": "UC",
            "Usephone": "UP",
            "Usetablet": "UT",
            "Walk": "Wa",
            "WatchTV": "TV"
        }

    def get_random_camera_angle(self, subject, room, action, house):
        action_info = self.action_df.query("VideoName.str.contains(\"{}_S{}{}{}_f\")".format(self.action_map[action],
                                                                                             subject,
                                                                                             room[0],
                                                                                             house),
                                           engine="python").reset_index(drop=True)
        indices = action_info.index
        filename = action_info["VideoName"][np.random.choice(indices)]
        m = re.match(r"(.*)_fC(.*).avi", filename)

        return int(m.group(2))
        # sys.exit()
        # return

    def get_duration(self, subject, room, camera_angle, action, house):
        # print(self.action_df.columns)
        # print("VideoName.str.contains(\"{}_S{}{}{}_f\")".format(self.action_map[action],
        #                                                         subject,
        #                                                         room[0],
        #                                                         house))
        action_info = self.action_df.query("VideoName.str.contains(\"{}_S{}{}{}_fC{}\")".format(self.action_map[action],
                                                                                                subject,
                                                                                                room[0],
                                                                                                house,
                                                                                                camera_angle),
                                           engine="python").reset_index(drop=True)
        return datetime.strptime(action_info["Duration"][0], "%H:%M:%S")

    def build_timeline(self):
        time = datetime.strptime("00:00:00", "%H:%M:%S")
        current_room = self.current_room
        timed_action_keys = np.sort(list(self.timed_actions.keys()))
        timed_event_idx = 0
        next_event_time = datetime.strptime(timed_action_keys[timed_event_idx], "%H:%M:%S")
        while time < self.maxtime:
            print(time)
            if np.random.rand() < self.room_change_probability:
                current_room = np.random.choice(self.rooms)
            # logic for current action
            if np.random.rand() < self.spurious_action_probability:
                current_action = np.random.choice(self.spurious_actions)
            else:
                current_action = self.idle_action

            # logic for timed action
            if time >= next_event_time:
                try:
                    camera_angle = self.get_random_camera_angle(self.current_subject,
                                                                current_room,
                                                                self.idle_action,
                                                                self.current_house)
                    duration = self.get_duration(self.current_subject,
                                                 current_room,
                                                 camera_angle,
                                                 self.idle_action,
                                                 self.current_house)
                    time += timedelta(hours=duration.hour, minutes=duration.minute, seconds=duration.second)
                    self.action_queue.append((current_room, camera_angle, self.idle_action))
                except (IndexError, ValueError):
                    pass
                current_room, current_action = timed_actions[next_event_time.strftime("%H:%M:%S")]

                if current_room == None:
                    current_room = self.current_room

                timed_event_idx += 1
                try:
                    next_event_time = datetime.strptime(timed_action_keys[timed_event_idx], "%H:%M:%S")
                except IndexError:
                    next_event_time = self.maxtime
                camera_angle = self.get_random_camera_angle(self.current_subject,
                                                            current_room,
                                                            current_action,
                                                            self.current_house)
                time = time - self.TIME_ZERO + self.get_duration(self.current_subject,
                                                                 current_room,
                                                                 camera_angle,
                                                                 current_action,
                                                                 self.current_house)
                self.action_queue.append((current_room, camera_angle, current_action))
                continue

            try:
                camera_angle = self.get_random_camera_angle(self.current_subject,
                                                            current_room,
                                                            current_action,
                                                            self.current_house)
                duration = self.get_duration(self.current_subject,
                                             current_room,
                                             camera_angle,
                                             current_action,
                                             self.current_house)
                time = time + timedelta(hours=duration.hour,
                                        minutes=duration.minute,
                                        seconds=duration.second)
                self.action_queue.append((current_room, camera_angle, current_action))
            except (IndexError, ValueError, AttributeError):
                pass

    def __iter__(self):
        # while True:
        self.build_timeline()
        self.time = self.TIME_ZERO
        for action_item in self.action_queue:
            room, camera_angle, action = action_item
            duration = self.get_duration(self.current_subject,
                                         room,
                                         camera_angle,
                                         action,
                                         self.current_house)
            self.time = self.time + timedelta(hours=duration.hour,
                                              minutes=duration.minute,
                                              seconds=duration.second)
            yield self.time, room, camera_angle, action


# timed_actions = {
#     "10:00:00": ("Kitchen", "Cook"),
#     "10:30:00": ("Dining", "Eat"),
#     "11:00:00": (None, "Drink"),
#     "12:00:00": ("Kitchen", "Cook"),
#     "12:30:00": ("Dining", "Eat"),
#     "13:00:00": (None, "Drink"),
#     "14:00:00": (None, "Drink"),
#     "15:00:00": (None, "Drink"),
#     "16:00:00": (None, "Drink"),
#     "17:00:00": (None, "Drink"),
#     "18:00:00": ("Kitchen", "Cook"),
#     "18:30:00": ("Dining", "Eat")
# }

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

spurious_actions = ["Readbook", "Usecomputer", "Usephone", "Usetablet", "WatchTV"]

ds = Sims4ActionDataset(data_path="/media/arjun/Shared/research_projects/temporal_weight_gating/Sims4ActionVideos",
                        timed_actions=timed_actions,
                        event_driven_actions=event_driven_actions,
                        spurious_actions=spurious_actions,
                        maxtime="12:00:00")

id = 0
for dp in ds:
    id += 1
    time, room, camera_angle, action = dp
    print("{} {} fC{} {}".format(time, room, camera_angle, action))
    # if id > 1000:
    #     break
