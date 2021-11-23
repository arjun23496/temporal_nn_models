import torch
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

import os


class Sims4ActionDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, timed_actions, event_driven_actions, spurious_actions, current_subject=1,
                 spurious_action_probability=0.9, current_house=1, fixed_camera=True, idle_action="WatchTV"):
        super(Sims4ActionDataset).__init__()
        self.timed_actions = timed_actions
        self.event_driven_actions = event_driven_actions
        self.spurious_actions = spurious_actions
        self.spurious_action_probability = spurious_action_probability
        self.fixed_camera = fixed_camera
        self.data_path = data_path

        self.time_zero = datetime.strptime("00:00:00", "%H:%M:%S")
        self.current_time = datetime.strptime("00:00:00", "%H:%M:%S")
        self.current_house = current_house
        self.current_subject = current_subject
        self.current_room = "Living"
        self.idle_action = idle_action
        self.action_queue = []
        self.action_df = pd.read_csv(os.path.join(self.data_path, "SimsSplitsCompleteVideos.csv"), sep=";")
        self.camera_angles = {
            "Dining1": [1, 2, 3, 4],
            "Kitchen1": [5, 6, 7, 8],
            "Living1": [9, 10, 11, 12],
            "Dining2": [13, 14, 15, 16],
            "Kitchen2": [17, 18, 19, 20],
            "Living2": [21, 22, 23, 24]
        }
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

    def get_duration(self, subject, room, action, camera, house):
        # print(self.action_df.columns)
        print("VideoName.str.contains(\"{}_S{}{}{}_f\")".format(self.action_map[action],
                                                                subject,
                                                                room[0],
                                                                house))
        action_info = self.action_df.query("VideoName.str.contains(\"{}_S{}{}{}_f\")".format(self.action_map[action],
                                                                                             subject,
                                                                                             room[0],
                                                                                             house),
                                           engine="python").reset_index(drop=True)
        return datetime.strptime(action_info["Duration"][0], "%H:%M:%S")

    def initialize_action_queue(self):
        time = datetime.strptime("00:00:00", "%H:%M:%S")
        current_room = self.current_room
        timed_action_keys = np.sort(list(self.timed_actions.keys()))
        timed_event_idx = 0
        next_event_time = datetime.strptime(timed_action_keys[timed_event_idx], "%H:%M:%S")
        while time < datetime.strptime("20:00:00", "%H:%M:%S"):
            print(time)
            # logic for current action
            if np.random.rand() < self.spurious_action_probability:
                current_action = np.random.choice(self.spurious_actions)
            else:
                current_action = self.idle_action

            # logic for timed action
            if time >= next_event_time:
                self.action_queue.append((current_room, self.idle_action))
                try:
                    duration = self.get_duration(self.current_subject,
                                                 current_room,
                                                 self.idle_action,
                                                 np.random.choice(self.camera_angles["{}{}".format(current_room,
                                                                                                   self.current_house)]),
                                                 self.current_house)
                    time += timedelta(hours=duration.hour, minutes=duration.minute, seconds=duration.second)
                except IndexError:
                    pass
                current_room, current_action = timed_actions[next_event_time.strftime("%H:%M:%S")]

                if current_room == None:
                    current_room = self.current_room

                timed_event_idx += 1
                try:
                    next_event_time = datetime.strptime(timed_action_keys[timed_event_idx], "%H:%M:%S")
                except IndexError:
                    next_event_time = datetime.strptime("23:59:59", "%H:%M:%S")
                self.action_queue.append((current_room, current_action))
                time = time - self.time_zero + self.get_duration(self.current_subject,
                                                                 current_room,
                                                                 current_action,
                                                                 np.random.choice(
                                                                     self.camera_angles[
                                                                         "{}{}".format(
                                                                             current_room,
                                                                             self.current_house)]),
                                                                 self.current_house)
                continue

            try:
                duration = self.get_duration(self.current_subject,
                                             current_room,
                                             current_action,
                                             np.random.choice(self.camera_angles["{}{}".format(current_room,
                                                                                               self.current_house)]),
                                             self.current_house)
                time = time + timedelta(hours=duration.hour,
                                        minutes=duration.minute,
                                        seconds=duration.second)
                self.action_queue.append((current_room, current_action))
            except IndexError:
                pass

    def __iter__(self):
        while True:
            self.initialize_action_queue()
            for action in self.action_queue:
                yield action


timed_actions = {
    "10:00:00": ("Kitchen", "Cook"),
    "10:30:00": ("Dining", "Eat"),
    "11:00:00": (None, "Drink"),
    "12:00:00": ("Kitchen", "Cook"),
    "12:30:00": ("Dining", "Eat"),
    "13:00:00": (None, "Drink"),
    "14:00:00": (None, "Drink"),
    "15:00:00": (None, "Drink"),
    "16:00:00": (None, "Drink"),
    "17:00:00": (None, "Drink"),
    "18:00:00": ("Kitchen", "Cook"),
    "18:30:00": ("Dining", "Eat")
}

event_driven_actions = {}

spurious_actions = ["Readbook", "Usecomputer", "Usephone", "Usetablet", "WatchTV"]

ds = Sims4ActionDataset(data_path="/media/arjun/Shared/research_projects/temporal_weight_gating/Sims4ActionVideos",
                        timed_actions=timed_actions,
                        event_driven_actions=event_driven_actions,
                        spurious_actions=spurious_actions)

id = 0
for i in ds:
    id += 1
    print(i)
    if id > 20:
        break
