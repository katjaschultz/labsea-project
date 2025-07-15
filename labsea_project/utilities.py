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

def fit_of_running_mean(x, y, window_size=5, degree=5):
    """ Fit a polynomial of degree n to the running mean of x and y.
        Returns the polynomial function."""

    x_smooth = []
    y_smooth = []

    for i in range(len(x) - window_size + 1):
        x_smooth.append(np.mean(x[i:i+window_size]))
        y_smooth.append(np.mean(y[i:i+window_size]))
        
    x_smoothR, y_smoothR = np.array(x_smooth), np.array(y_smooth)
        
    # Fit a polynomial of degree n
    coefficientsA = np.polyfit(x_smoothR, y_smoothR, degree)
    poly_functionA = np.poly1d(coefficientsA)

    #poly_functionA = scipy.interpolate.interp1d(x_smoothR, y_smoothR, kind='linear', fill_value='extrapolate')

    return poly_functionA

def select_datapoints_by_season_and_year(dataset, season, years):
    """
    Selects profiles from the dataset based on the given season and year.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing datetime information.
    season (str or list): Season to filter profiles by ('spring', 'summer', 'winter', 'all_year', [list of months]).
    years (list): List of years to filter profiles by.
    
    Returns:
    mask (numpy.ndarray): Boolean mask indicating selected profiles.
    """
    
    mask = np.full(dataset.datetime.shape, False)  # Initialize mask
    for year in years:
            if season == 'winter':
                mask_1 = np.full(dataset.datetime.shape, False)
                mask_2 = np.full(dataset.datetime.shape, False)
                mask_1 |= (dataset.datetime.dt.year == year) & (
                    (dataset.datetime.dt.month == 1) | 
                    (dataset.datetime.dt.month == 2) | 
                    (dataset.datetime.dt.month == 3)
                )
                mask_2 |= (dataset.datetime.dt.year == year-1) & (
                    (dataset.datetime.dt.month == 12)
                )
                mask |= mask_1 | mask_2
        
            elif season == 'spring':
                mask |= (dataset.datetime.dt.year == year) & (
                    (dataset.datetime.dt.month == 4) | 
                    (dataset.datetime.dt.month == 5) | 
                    (dataset.datetime.dt.month == 6)
                )
        
            elif season == 'summer':
                mask |= (dataset.datetime.dt.year == year) & (
                    (dataset.datetime.dt.month == 7) | 
                    (dataset.datetime.dt.month == 8) | 
                    (dataset.datetime.dt.month == 9) |  
                    (dataset.datetime.dt.month == 10) | 
                    (dataset.datetime.dt.month == 11)
                )
            elif season == 'all_year':
                mask |= (dataset.datetime.dt.year == year)
            elif isinstance(season, list):
                # If season is a list of months, filter by those months
                mask |= (dataset.datetime.dt.year == year) & (dataset.datetime.dt.month.isin(season))

    return mask

