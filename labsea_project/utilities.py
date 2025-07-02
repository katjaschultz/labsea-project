import numpy as np
import math
from geopy.distance import geodesic

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

# above
p01 = [ 53.77, -56.8 ]
p02 = [ 61.35, -48.83]

# beneath
p11 = [ 60.97, -46.4]
p12 = [ 52.38, -55.52]

# AR7W
pA1 = [53.517, -55.73]
pA2 = [60.57, -48.23] 

# Ar7W (new)
pa1 = [55.185, -53.957]
pa2 = [60.57, -48.23]

def generate_coordinates_along_line(start_coords, end_coords, interval_km):
    distance = geodesic(start_coords, end_coords).kilometers
    num_points = int(distance / interval_km)

    coordinates = [start_coords]
    for i in range(1, num_points):
        fraction = i / num_points
        interpolated_coords = [
            start_coords[0] + fraction * (end_coords[0] - start_coords[0]),
            start_coords[1] + fraction * (end_coords[1] - start_coords[1])
            ]
        coordinates.append(interpolated_coords)

    coordinates.append(end_coords)
    
    coord = np.array(coordinates)
    lat_line = coord[:,0]
    lon_line = coord[:,1]

    return lat_line, lon_line

interval_distance_km = 10

start_coordinates_0 = p01 
end_coordinates_0   = p02  

start_coordinates_1 = p11 
end_coordinates_1   = p12 

start_coordinates_A = pA1 
end_coordinates_A   = pA2 

lat_line_0, lon_line_0 = generate_coordinates_along_line(start_coordinates_0, end_coordinates_0, interval_distance_km)
lat_line_1, lon_line_1 = generate_coordinates_along_line(start_coordinates_1, end_coordinates_1, interval_distance_km)
lat_line_A, lon_line_A = generate_coordinates_along_line(start_coordinates_A, end_coordinates_A, interval_distance_km)

lat_line_a, lon_line_a = generate_coordinates_along_line(pa1, pa2, interval_distance_km)
