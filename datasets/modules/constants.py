from datetime import datetime

TIME_FORMAT = "%H:%M:%S"
MAX_TIME = datetime.strptime("12:00:00", TIME_FORMAT)
MIN_TIME = datetime.strptime("00:00:00", TIME_FORMAT)
