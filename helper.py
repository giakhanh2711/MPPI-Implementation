import math


def normalize_angle(angle):
    """
    Normalize angle (radian) to [-pi, pi)
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi
