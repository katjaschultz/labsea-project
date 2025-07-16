import numpy as np
import xarray as xr
import pathlib
import sys
import gsw

script_dir = pathlib.Path().parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

from labsea_project import tools


def create_dataset(case, file_case, omega, xstart, xend, season, years, spacing_z=25, spacing_x=10, mask_sigma=True):

    ''' Function to load data from files, calculate overturning and return a dataset with all relevant variables
    Input parameters:

    - case: string, name of the case to load (specify e.g. the season)
    - file_case: string, specify time frame and whether to load only profiles within 1000db isobars or not
      default is '2004_to_2023_1000db_isobars'
    - omega: float, omega value used in gaussian filter to calc. weighted mean
    - xstart: float, x-coordinate of the start point of the AR7W line
    - xend: float, x-coordinate of the end point of the AR7W line
    - spacing_z: float, spacing in z-direction for the grid
    - spacing_x: float, spacing in x-direction for the grid
    - season: string, season to filter profiles by (e.g. 'spring', 'summer', 'winter', 'all_year')
    - years: list of years to filter profiles by
    - mask_sigma: boolean, if True, mask the sigma0 values below 27.8 kg/m^3

'''
    # Load data from the specified file
    file_path = parent_dir / f"data/weighted data/weighted_data_{file_case}_{case}_omega{int(omega)}_xstart{str(xstart)}_xend{xend}.npy"

    specvol_anom, sigma0, SA, CT = np.load(file_path)
    
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
            'x': (['x'], x),
            'z': (['z'], z)
        },
        attrs={
            'description': f'Weighted data for {case} with omega={omega}, xstart={xstart}, xend={xend}',
            'units': {
                'specvol_anom (specific volume anomaly)': 'kg/m^3 \n',
                'sigma0 (potential density)': 'kg/m^3 \n',
                'SA (absolute salinity)': 'g/kg \n',
                'CT (conservative temperature)': 'degree_Celsius'
            }
        })

    # calculate overtruning and horizontal transports

    # load reference velocity
    input_file = parent_dir / 'demo data/yomaha_velocities_referenced_to_1000dbar_new.nc'
    
    #poly_func = ref.derive_poly_func(input_file, start_time=start_time, end_time=end_time, cut_season=cut_season, season=season, returnXY=False)
    poly_func = tools.derive_poly_func(input_file,  years, season=season, degree=5, start_x=xstart, end_x=xend, cut_edges=True, returnXY=False)

    xhalf = (x[1:] + x[:-1])/2
    _, Zhalf = np.meshgrid(xhalf, z)

    argo_ref = poly_func(xhalf) / 100 

    lon_ar7w, lat_ar7w = np.load(parent_dir / f'demo data/coordinates_AR7W_xstart{xstart}.npy') # represent coordinates of chosen grid (10km spacing in x)
    _, Z  = np.meshgrid(x, z)
    P = gsw.p_from_z(Z, lat_ar7w)

    v, sigma0half    = tools.derive_abs_geo_v(specvol_anom, sigma0, P, argo_ref, lon_ar7w, lat_ar7w, xhalf, Zhalf, mask_sigma=mask_sigma)
    strf_z, strf_x, imbalance, mask_xa, piecewise_trapz_z, piecewise_trapz_x = tools.derive_strf(v, xhalf*1e3, z*-1)

    # add variables to dataset (coords: xhalf, z)
    # add xhalf as a coordinate
    ds = ds.assign_coords(xhalf=('xhalf', xhalf))
    ds['z0'] = (['z0'], z[1:]) # exclude 0 point from z-coordinate
    ds['x0'] = (['x0'], xhalf[1:]) # exclude 0 point from x-coordinate

    ds['v'] = (['z', 'xhalf'], v)
    ds['sigma0half'] = (['z', 'xhalf'], sigma0half)
    ds['strf_z'] = (['z'], strf_z)
    ds['strf_x'] = (['xhalf'], strf_x)
    ds['mask_xa'] = (['xhalf'], mask_xa.data)
    ds['piecewise_trapz_z'] = (['z0'], piecewise_trapz_z)
    ds['piecewise_trapz_x'] = (['x0'], piecewise_trapz_x)
    ds['argo_ref'] = (['xhalf'], argo_ref)
    ds['imbalance'] = imbalance

    return ds
