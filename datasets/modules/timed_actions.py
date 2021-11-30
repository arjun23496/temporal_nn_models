import numpy as np
from datetime import datetime
from datetime import timedelta
import constants

class TimedActionsManager:
    def __init__(self, action_df):
        self._timed_actions = {}
        self._timed_event_idx = None
        self._next_event_time = None
        self._action_df = action_df
        self.reset()

    def add_action(self, action, time):
        self._timed_actions[time] = action

    def reset(self):
        timed_action_keys = np.sort(list(self._timed_actions.keys()))
        self._timed_event_idx = 0
        self._next_event_time = datetime.strptime(timed_action_keys[self._timed_event_idx], constants.TIME_FORMAT)

    def update_timeline(self, current_time, current_room, timeline):
        if current_time >= self._next_event_time:
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
                current_time += timedelta(hours=duration.hour, minutes=duration.minute, seconds=duration.second)
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

