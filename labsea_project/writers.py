import numpy as np
import xarray as xr
import pathlib
import sys
import gsw

script_dir = pathlib.Path().parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

from labsea_project import reference_func as ref
from labsea_project import utilities, plotters, tools


''' Function to load data from files and return a dataset suitable for plotting
    Input parameters:
    - case: string, name of the case to load (specify e.g. the season)
    - file_case: string, specify time frame and whether to load only profiles within 1000db isobars or not
      default is '2004_to_2023_1000db_isobars'
    - omega: float, omega value used in gaussian filter to calc. weighted mean
    - xstart: float, x-coordinate of the start point of the AR7W line
    - xend: float, x-coordinate of the end point of the AR7W line
    - spacing_z: float, spacing in z-direction for the grid
    - spacing_x: float, spacing in x-direction for the grid
    - mask_sigma: boolean, if True, mask the sigma0 values below 27.8 kg/m^3

'''
def create_dataset(case, file_case, omega, xstart, xend, spacing_z=25, spacing_x=10, mask_sigma=True):
    # Load data from the specified file

    file_path = parent_dir / f"data/weighted data/weighted_data_{file_case}_{case}_omega{int(omega)}_xstart{str(xstart)}_xend{xend}.npy"
    specvol_anom, sigma0, SA, CT = np.load(file_path)
    season = case.split('_')[1]  # Extract season from the case name
    print('season:', season) # small check
    x = np.arange(xstart, xend + spacing_x, spacing_x)
    z = np.arange(0, 2000 + spacing_z, spacing_z) * -1 # negative depth
    # Create a new xarray dataset
    ds = xr.Dataset(
        {
            'specvol_anom': (['z', 'x'], specvol_anom),
            'sigma0': (['z', 'x'], sigma0),
            'SA': (['z', 'x'], SA),
            'CT': (['z', 'x'], CT)
        },
        coords={
            'x': (['x'], np.arange(xstart, xend + spacing_x, spacing_x)),
            'z': (['z'], np.arange(0, 2000 + spacing_z, spacing_z))
        },
        attrs={
            'description': f'Weighted data for {case} with omega={omega}, xstart={xstart}, xend={xend}',
            'units': {
                'specvol_anom (specific volume anomaly)': 'kg/m^3',
                'sigma0 (potential density)': 'kg/m^3',
                'SA (absolute salinity)': 'g/kg',
                'CT (conservative temperature)': 'degree_Celsius'
            }
        })
    
    # calculate overtruning and horizontal transports

    # load reference velocity
    # raise error if file_case is not '2004_to_2023_1000db_isobars'
    input_file = parent_dir / 'data/yomaha_velocities_referenced_to_1000dbar_new.nc'

    if file_case != '2004_to_2023_1000db_isobars':
        print("Caution: file_case is not default '2004_to_2023_1000db_isobars', please enter start and end time")
        # ask user get keyboard input for start and end time
        start_time = input("Enter start time (YYYY-MM-DD): ")
        end_time = input("Enter end time (YYYY-MM-DD): ")
        poly_func = ref.derive_poly_func(input_file, start_time=start_time, end_time=end_time, cut_season=True, season=season, returnXY=False)

    else:
        poly_func = ref.derive_poly_func(input_file, start_time='2004-01-01', end_time='2013-12-31', cut_season=True, season=season, returnXY=False)

    xhalf = (x[1:] + x[:-1])/2
    _, Zhalf = np.meshgrid(xhalf, z)

    argo_ref = poly_func(xhalf) / 100 

    lon_ar7w, lat_ar7w = np.load(parent_dir / f'data/coordinates_AR7W_xstart{xstart}.npy') # represent coordinates of chosen grid (10km spacing in x)
    _, Z  = np.meshgrid(x, z)
    P = gsw.p_from_z(Z, lat_ar7w)

    v, sigma0half    = tools.derive_abs_geo_v(specvol_anom, sigma0, P, argo_ref, lon_ar7w, lat_ar7w, xhalf, Zhalf, mask_sigma=mask_sigma)
    strf_z, strf_x, imbalance, mask_xa = tools.derive_strf(v, xhalf*1e3, z*-1)

    # add variables to dataset (coords: xhalf, z)
    # add xhalf as a coordinate
    ds = ds.assign_coords(xhalf=('xhalf', xhalf))

    ds['v'] = (['z', 'xhalf'], v)
    ds['sigma0half'] = (['z', 'xhalf'], sigma0half)
    ds['strf_z'] = (['z'], strf_z)
    ds['strf_x'] = (['xhalf'], strf_x)
    ds['mask_xa'] = (['xhalf'], mask_xa.data)
    ds['argo_ref'] = (['xhalf'], argo_ref)
    ds['imbalance'] = imbalance

    return ds







