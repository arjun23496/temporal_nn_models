import numpy as np
from datetime import datetime
from datetime import timedelta

from datasets.modules.utils import get_random_camera_angle
from datasets.modules.utils import get_duration
from datasets.modules.utils import get_filename

import datasets.modules.constants as constants
import copy


class SpuriousActionsManager:
    def __init__(self, action_df, available_rooms, room_change_probability=0.2):
        self._spurious_actions = {}
        self._action_df = action_df
        self._spurious_actions_list = None
        self._action_probs = None
        self.rooms = available_rooms
        self.room_change_probability = room_change_probability

    def add_action(self, action, unnorm_probability):
        self._spurious_actions[action] = unnorm_probability

    def reset(self):
        self._spurious_actions_list = list(self._spurious_actions.keys())
        self._action_probs = [self._spurious_actions[action] for action in self._spurious_actions_list]

        # normalization of action probabilities
        self._action_probs = np.array(self._action_probs) / np.sum(self._action_probs)

    def update_timeline(self, house, subject, current_room, current_time, timeline):
        add_status = True
        time0 = copy.deepcopy(current_time)

        room = current_room
        if np.random.rand() < self.room_change_probability:
            room = np.random.choice(self.rooms)

        try:
            action = np.random.choice(self._spurious_actions_list, p=self._action_probs)

            camera_angle = get_random_camera_angle(self._action_df,
                                                   subject,
                                                   room,
                                                   action,
                                                   house)

            current_time = current_time - constants.MIN_TIME + get_duration(self._action_df,
                                                                            subject,
                                                                            room,
                                                                            camera_angle,
                                                                            action,
                                                                            house)
            filename = get_filename(self._action_df,
                                    subject,
                                    room,
                                    camera_angle,
                                    action,
                                    house)
            timeline.append((time0, room, camera_angle, action, filename))
        except (IndexError, ValueError):
            add_status = False

        return add_status, current_time, room, timeline
