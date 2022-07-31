import math
from typing import Tuple

from haversine import haversine


class CoordinatesUtils:
    @classmethod
    def calc_haversine_in_km(cls, coord1: Tuple, coord2: Tuple):
        return haversine(coord1, coord2)

    @classmethod
    def dms2km(cls, r):
        A = 6.371
        L = 2 * math.pi * r * A / 360
        return L

    @classmethod
    def dec2dms(cls, degree):
        mnt, sec = divmod(degree * 3600, 60)
        deg, mnt = divmod(mnt, 60)
        return deg, mnt, sec

    @classmethod
    def dms2dec(cls, degree, minute, second):
        return int(degree) + float(abs(minute)) / 60 + float(abs(second)) / 3600


# 13737.958776315085
L = CoordinatesUtils.dms2km(0.00833333333333334)
print(L)
