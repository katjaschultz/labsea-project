# Based on https://github.com/voto-ocean-knowledge/votoutils/blob/main/votoutils/utilities/utilities.py
import re
import numpy as np
import pandas as pd
import logging
import datetime
import xarray as xr

import numpy as np
import math

def ll2km(lon, lat, bbox):
    """returns xkm, ykm from the origin (lon0, lat0) of a given box [lon0, lon1, lat0, lat1])
       for given lon/lat coordinates """
    rearth = 6370800  # Earth radius [m].
    deg2rad = np.pi/180
    ykm = rearth * (lat - bbox[2]) * deg2rad / 1000
    xkm = rearth * ((deg2rad * (lon - bbox[0])) *
                    np.cos(deg2rad * 0.5 * (lat + bbox[2])) / 1000)

    return xkm, ykm

# AR7W intersection at the coast
bbox = [-55.73, -44, 53.517, 68]  

def rotate_point(x, y, angle_rad=np.radians(60.17555927225667)):
    """Rotate point (x, y) by a given angle in radians.
       The rotation is counter-clockwise around the origin (0, 0)."""
    new_x = x * math.cos(angle_rad) + y * math.sin(angle_rad)
    new_y = - x * math.sin(angle_rad) + y * math.cos(angle_rad)
    
    return new_x, new_y

def rotate_point_corr(x, y, angle_rad=np.radians(60.17555927225667)):
    """Rotate point (x, y) in km by a given angle in radians.
       The rotation is counter-clockwise around the origin (0, 0).
       This version corrects for an offsets of -21.0366 km and +11.7872 km, to provide
       distances along the AR7W line comparable to literature values."""
    new_x = x * math.cos(angle_rad) + y * math.sin(angle_rad)
    new_y = - x * math.sin(angle_rad) + y * math.cos(angle_rad)
    
    return new_x - 21.0366, new_y + 11.7872 

