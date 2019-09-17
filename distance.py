"""
    Method for calculating the distance between two latitudes and longitudes

    @author: Wangmei
    @date: 2019/05/16

"""

import math
from numpy import mat

# 6378.137 is the radius of the earth with kilometers;
EARTH_REDIUS = 6378.137

# Define the circumference calculation method
def rad(d):
    rad = float(d) * math.pi / 180.0
    return rad

#Define the distance between two latitudes and longitudes (using the Google Map Distance calculation method)
def getDistance(lat1, lng1, lat2, lng2):      #  Lat1 Lung1 represents location A ，Lat2 Lung2 represents location B
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    #print(a)
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b/2), 2)))
    s = s * EARTH_REDIUS
    return s

