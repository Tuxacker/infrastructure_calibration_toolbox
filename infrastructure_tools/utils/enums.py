from enum import Enum, IntEnum


class Singleton(Enum):
    DUMMY = 1


class TimestampType(Enum):
    SEC = 1
    NANOSEC = 2


class Classification(IntEnum):
    UNCLASSIFIED = 0
    PEDESTRIAN = 1
    BICYCLE = 2
    MOTORBIKE = 3
    CAR = 4
    TRUCK = 5
    VAN = 6
    BUS = 7
    ANIMAL = 8
    ROAD_OBSTACLE = 9
    TRAIN = 10
    TRAFFIC_LIGHT_NONE = 30
    TRAFFIC_LIGHT_RED = 31
    TRAFFIC_LIGHT_RED_YELLOW = 32
    TRAFFIC_LIGHT_YELLOW = 33
    TRAFFIC_LIGHT_GREEN = 34
    TRAFFIC_SIGN_NONE = 40
    CAR_UNION = 50  # CAR or VAN
    TRUCK_UNION = 51  # TRUCK or BUS or TRAIN
    BIKE_UNION = 52  # BICYCLE or MOTORBIKE
    UNKNOWN = 100