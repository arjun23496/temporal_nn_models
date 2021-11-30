import numpy as np
import re
from datetime import datetime


def get_random_camera_angle(self, action_df, subject, room, action, house):
    action_info = action_df.query("VideoName.str.contains(\"{}_S{}{}{}_f\")".format(self.action_map[action],
                                                                                    subject,
                                                                                    room[0],
                                                                                    house),
                                  engine="python").reset_index(drop=True)
    indices = action_info.index
    filename = action_info["VideoName"][np.random.choice(indices)]
    m = re.match(r"(.*)_fC(.*).avi", filename)

    return int(m.group(2))


def get_duration(self, action_df, subject, room, camera_angle, action, house):
    action_info = action_df.query("VideoName.str.contains(\"{}_S{}{}{}_fC{}\")".format(self.action_map[action],
                                                                                       subject,
                                                                                       room[0],
                                                                                       house,
                                                                                       camera_angle),
                                  engine="python").reset_index(drop=True)
    return datetime.strptime(action_info["Duration"][0], "%H:%M:%S")
