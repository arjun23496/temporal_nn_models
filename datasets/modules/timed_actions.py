import numpy as np
from datetime import datetime
from datetime import timedelta
from datasets.modules.utils import get_random_camera_angle
from datasets.modules.utils import get_duration

import datasets.modules.constants as constants
import copy


class TimedActionsManager:
    def __init__(self, action_df):
        self._timed_actions = {}
        self._timed_event_idx = None
        self._timed_action_keys = None
        self._next_event_time = None
        self._action_df = action_df

    def add_action(self, action, time):
        self._timed_actions[time] = action

    def reset(self):
        self._timed_action_keys = np.sort(list(self._timed_actions.keys()))
        self._timed_event_idx = 0
        self._next_event_time = datetime.strptime(self._timed_action_keys[self._timed_event_idx],
                                                  constants.TIME_FORMAT)

    def update_timeline(self, house, subject, current_room, current_time, timeline):
        add_status = False
        room = current_room
        if current_time >= self._next_event_time:
            time0 = copy.deepcopy(current_time)
            room, action = self._timed_actions[self._next_event_time.strftime(constants.TIME_FORMAT)]

            if room is None:
                room = current_room

            self._timed_event_idx += 1

            try:
                self._next_event_time = datetime.strptime(self._timed_action_keys[self._timed_event_idx],
                                                          constants.TIME_FORMAT)
            except IndexError:
                self._next_event_time = constants.MAX_TIME

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
            timeline.append((time0, room, camera_angle, action))
            add_status = True

        return add_status, current_time, room, timeline
