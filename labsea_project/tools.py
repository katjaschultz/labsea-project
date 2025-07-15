import numpy as np
import xarray as xr
import gsw
from labsea_project.utilities import ll2km, rotate_point_corr
from labsea_project import utilities
import scipy

import pathlib
import sys

"""
This module provides core computational functions for processing and analyzing oceanographic data 
in the labsea-project. It includes routines for calculating dynamic height, absolute geostrophic 
velocity, overturning streamfunctions, transport in density space, profile selection and interpolation, 
and other utilities needed for the analysis of Argo and CTD datasets.
"""

script_dir = pathlib.Path().parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

def calc_dyn_h(specvol_anom, p, p_ref=0):

    """ Calculate dynamic height anomaly from specific volume anomaly.
    Args:
        specvol_anom (numpy.ndarray): Specific volume anomaly.
        p (numpy.ndarray): Pressure levels.
        p_ref (float): Reference pressure level. Default is 0."""
    
    db2Pa   = 1e4
    B       = np.zeros([specvol_anom.shape[0],specvol_anom.shape[1]])
    idx_ref = np.argmin(np.abs(p[:,0]-p_ref))

    for k in range(specvol_anom.shape[1]): 
        B[1:,k] = 0.5 * (specvol_anom[:-1,k] + specvol_anom[1:,k]) * (p[1:,k] - p[:-1,k])* db2Pa
 
    D0    = - np.cumsum(B, axis=0)
    D_ref = D0[idx_ref,:]
    dyn_h = D0 - D_ref 
    
    return dyn_h
    
def calc_abs_geo_v(geo_v, ref_vel, p):

    """ Calculate absolute geostrophic velocity from relative geostrophic velocity 
        and reference velocity at 1000 dbar.
    Args:
        geo_v (numpy.ndarray): Relative geostrophic velocity.
        ref_vel (numpy.ndarray): Reference velocity.
        p (numpy.ndarray): Pressure levels."""
    
    Vp    =  np.zeros(geo_v.shape[1])
    
    for i in range(0,geo_v.shape[1]):
        idx_p = np.argmin(abs(p[:,i]-1000))
        Vp[i] = geo_v[idx_p,i]
    
    # reference to V at p=1000
    v_0    = ref_vel - Vp    

    return geo_v  + v_0


def derive_abs_geo_v(specvol_anom, sigma0, p, ref_vel, lon_ar7w, lat_ar7w, xhalf, Z, p_ref=0, mask_sigma=True):
   
    """ Calculate absolute geostrophic velocity from specific volume anomaly 
        directly using the functions above. 
        Masking out below densities of 27.8 kg/m^3 and at the topography.
    Args: 
        specvol_anom (numpy.ndarray): Specific volume anomaly.
        sigma0 (numpy.ndarray): Potential Density.
        ref_vel (numpy.ndarray): Reference velocity.
        xhalf (numpy.ndarray): x-coordinates of derived geostrophic velocity.
        Z (numpy.ndarray): Depth levels."""
    
    dyn_h = calc_dyn_h(specvol_anom, p, p_ref)
    geo_v = gsw.geostrophy.geostrophic_velocity(dyn_h, lon_ar7w[:], lat_ar7w[:], axis=0, p=0)
    geo_v = geo_v[0] # relative geostrophic velocity on axis xhalf, z
    
    v      = calc_abs_geo_v(geo_v, ref_vel, p)
    
    # mask topo and density
    sigma0half = (sigma0[:,1:] + sigma0[:,:-1])/2
    mask    = sigma0half <= 27.8
  
    x_topo, topo = np.load(parent_dir / 'demo data/corrected_topography.npy')

    ind = [np.argmin(abs(x_topo - d)) for d in xhalf]
    mask_topo = Z <= topo[ind]*-1

    v[mask_topo] = np.nan
    sigma0half[mask_topo] = np.nan

    if mask_sigma:
        v[~mask] = np.nan
        sigma0half[~mask] = np.nan
      
    return v, sigma0half

def derive_strf(v, x, z, sensitivity = 0,  onlyEast=False, returnDelta=False):

    """ Calculate the horizontal and vertical transport streamfunction from the velocity field. 
        option to add a sensitivity to the velocity field for the whole section or only at eastern part
        to minimize the imbalance of the streamfunction.
    Args:
        v (numpy.ndarray): Velocity field.
        x (numpy.ndarray): x-coordinates.
        z (numpy.ndarray): Depth levels.
        sensitivity (float): Sensitivity to be added to the velocity field.
        onlyEast (bool): If True, apply sensitivity only to the eastern part.
        returnDelta (bool): If True, return delta_x and delta_z.
    Returns:
        strf_z (numpy.ndarray): Vertical transport streamfunction.
        strf_x (numpy.ndarray): Horizontal transport streamfunction.
        imbalance (float): Imbalance of the streamfunction.
        mask_x (numpy.ndarray): Mask for horizontal transport streamfunction.
    Optional:
        delta_x (numpy.ndarray): Delta x-coordinates.
        delta_z (numpy.ndarray): Delta depth levels.
        """
    
    # define grid
    spacing_z, spacing_x = 25, 10

    v_copy = v.copy()

    if onlyEast == False:  
        mask = ~np.isnan(v_copy)
        vel    = xr.DataArray(data = v_copy + sensitivity,
                        dims = ["depth", "distance"],
                        coords = dict(depth = z,
                                        distance = x))
    elif onlyEast == True:
        # add sensitivity for x >= xend-100km
        v_copy[:,-10:] = v_copy[:,-10:] + sensitivity 
        mask = ~np.isnan(v_copy) 
        vel    = xr.DataArray(data = v_copy,
                        dims = ["depth", "distance"],
                        coords = dict(depth = z,
                                        distance = x))
        
    delta_x = np.array([len(mask[i])-len(np.where(mask[i] == False)[0]) for i in range(len(mask))])*spacing_x
    delta_z = np.array([len(mask[:,i])-len(np.where(mask[:,i] == False)[0]) for i in range(mask.shape[1])])*spacing_z
    
    strf_z  = scipy.integrate.cumulative_trapezoid(vel.mean(dim="distance", skipna=True)*delta_x*1000, x=vel.depth, initial=0)*10**(-6) # in m, in Sv
            
    vp = (vel - vel.mean(dim='distance', skipna=True) )
    vh = vp.mean(dim='depth', skipna=True)
    mask_x = np.isnan(vh)
    strf_x = scipy.integrate.cumulative_trapezoid(vh[~np.isnan(vh)]*delta_z[~np.isnan(vh)], x=vel.distance[~np.isnan(vh)], initial=0)*10**(-6)
    
    imbalance = strf_x[-1]

    piecewise_trapz_z = np.array([
    scipy.integrate.trapezoid(vel.mean(dim="distance", skipna=True)[i:i+2] * delta_x[i:i+2]*1e3, x=vel.depth[i:i+2]*-1) * 10**(-6) 
    for i in range(len(vel.depth) - 1)])

    piecewise_trapz_x = np.array([
    scipy.integrate.trapezoid(vh[~np.isnan(vh)][i:i+2] * delta_z[i:i+2], x=vel.distance[~np.isnan(vh)][i:i+2]) * 10**(-6)
    for i in range(len(vel.distance) - 1)])

    if returnDelta:
        return strf_z, strf_x, imbalance, mask_x, piecewise_trapz_z, piecewise_trapz_x, delta_x, delta_z
    else:
        return strf_z, strf_x, imbalance, mask_x, piecewise_trapz_z, piecewise_trapz_x
    

    
def derive_transport_in_density_space(v, sigma0half, sigma_bins, sensitivity = 0, onlyEast=False):

    if onlyEast is False:
        v = v + sensitivity
    else:
        v[:,-10:] = v[:,-10:] + sensitivity
    
    A = 10000*25 # area of a grid cell in depth space
    T = v * A    # tranport of each grid cell

    bin_indices = np.digitize(sigma0half, sigma_bins)

    Q0 = np.zeros(len(sigma_bins)) # last bin contains nan values
    Q = np.zeros(len(sigma_bins))

    # calc transport in each bin
    # bin_indices corresponds to bin smaller than the value of the bin, e.g the first bin is 26.8, so all values smaller than 26.8 are in bin 0
    for i in range(0,len(sigma_bins)):
        bin_mask = bin_indices == i
        Q0[i] = np.nansum(T[bin_mask])*10**(-6) # in Sv

    # cumulative sum of the transport
    for i in range(0,len(sigma_bins)):
        Q[i] = np.nansum(Q0[:i+1])

    return Q0, Q

def find_adjustment_velocity(v, x, z, onlyEast=False):

    """ Find the adjustment velocity to minimize the vertical imbalance of the streamfunction.
    Args:
        v (numpy.ndarray): Velocity field.
        x (numpy.ndarray): x-coordinates.
        z (numpy.ndarray): Depth levels.
        onlyEast (bool): If True, apply sensitivity only to the eastern part.
        """
        
    imbalance_z = 0.1 #initialize

    if onlyEast:
        eps = 1e-3
        sensitivity = np.arange(-1, 1, 0.1)
        precision   = 0.1
    else:
        eps = 1e-4
        sensitivity = np.arange(-0.2,0.2, 0.01)
        precision   = 0.01

    while imbalance_z >= eps:
        
        imb_x = np.zeros(len(sensitivity))
        imb_z = np.zeros(len(sensitivity))

        for i, sens in enumerate(sensitivity):
            strf_z1, strf_x1, imbalance, _, _, _ = derive_strf(v, x, z, sensitivity=sens, onlyEast=onlyEast)
            imb_z[i] = strf_z1[-1]-strf_z1[0]
            imb_x[i] = imbalance


        imbalance_z = abs(imb_z).min()
        const = sensitivity[np.argmin(abs(imb_z))].round(6)

        sensitivity = np.arange(const-precision, const+precision, precision*0.1)
        precision = precision*0.1
    
    print(f'Adjustment velocity of {const} m/s determined at {imbalance_z.round(6)} Sv vertical imbalance')

    return const

    

def load_selected_profiles(filename, mask_profiles=np.array([])):
    """
    Loads selected profiles from the Argo dataset.
    
    Args:
        mask_profiles (array of int): Array of profile indices (saved beforehand for specific experiments).
        default is set to all profiles
    
    Returns:
        x_data: Rotated x-coordinates (shape: [n_profiles, M])
        z_data: Rotated depth levels (shape: [n_profiles, M])
        specvol_anom: Specific volume anomaly (shape: [n_profiles, M])
        sigma0: Potential density (shape: [n_profiles, M])
        SA: Absolute salinity (shape: [n_profiles, M])
        CT: Conservative temperature (shape: [n_profiles, M])
    """
    print(filename)
    argo_ds = xr.open_dataset(filename, engine='netcdf4')

    
    if mask_profiles.size==0:
        argo_sel = argo_ds
    else:
        argo_sel = argo_ds.where(mask_profiles, drop=True)
        
    M = len(argo_sel.N_LEVELS.values)
    
    # Calculate specific volume anomaly
    specvol_anom = gsw.density.specvol_anom_standard(
        argo_sel.SA.values, argo_sel.CT.values, argo_sel.PRES.values
    )
    
    sigma0 = gsw.density.sigma0(argo_sel.SA.values, argo_sel.CT.values)
    
    # Rotate coordinates
    bbox = [-55.73, -44, 53.517, 68]  # AR7W intersection at the coast
    lon_data = np.tile(argo_sel.LONGITUDE.values[:, np.newaxis], (1, M))
    lat_data = np.tile(argo_sel.LATITUDE.values[:, np.newaxis], (1, M))
    x_data, _ = rotate_point_corr(*ll2km(lon_data, lat_data, bbox))

    z_data     = gsw.z_from_p(argo_sel.PRES.values, lat_data)*10**(-3) # convert to km
    
    argo_ds.close()
    return x_data, z_data, specvol_anom, sigma0, argo_sel.SA.values, argo_sel.CT.values


def select_profiles_and_save_masks( filename, case_str, season, years ):

    """
    Selects profiles based on the given season and year, and saves the masks to a file.
    
    Parameters:
    filename (str): Path to the input netCDF file containing Argo data.
    case_str (str): Case string to append to the output file name to indicate what years and season is selected.
    season (str): Season to filter profiles by ('spring', 'summer', 'winter', 'all_year', [list of months]).
                    default (used in my analysis) 
                            spring contains April, May, June
                            summer contains July, August, September, October, November
                            winter contains January, February, March, December
    years (list): Year to filter profiles by given as a list
    
    """
    
    # Load the dataset
    argo_ds = xr.open_dataset(filename)
    mask = np.full(argo_ds.TIME.shape, False)  # Initialize mask

    for year in years:
        if season == 'winter':
            mask_1 = np.full(argo_ds.TIME.shape, False)
            mask_2 = np.full(argo_ds.TIME.shape, False)
            mask_1 |= (argo_ds.TIME.dt.year == year) & (
                (argo_ds.TIME.dt.month == 1) | 
                (argo_ds.TIME.dt.month == 2) | 
                (argo_ds.TIME.dt.month == 3)
            )
            mask_2 |= (argo_ds.TIME.dt.year == year-1) & (
                (argo_ds.TIME.dt.month == 12)
            )
            mask |= mask_1 | mask_2
    
        elif season == 'spring':
            mask |= (argo_ds.TIME.dt.year == year) & (
                (argo_ds.TIME.dt.month == 4) | 
                (argo_ds.TIME.dt.month == 5) | 
                (argo_ds.TIME.dt.month == 6)
            )
    
        elif season == 'summer':
            mask |= (argo_ds.TIME.dt.year == year) & (
                (argo_ds.TIME.dt.month == 7) | 
                (argo_ds.TIME.dt.month == 8) | 
                (argo_ds.TIME.dt.month == 9) |  
                (argo_ds.TIME.dt.month == 10) | 
                (argo_ds.TIME.dt.month == 11)
            )
        elif season == 'all_year':
            mask |= (argo_ds.TIME.dt.year == year)
        elif isinstance(season, list):
            # If season is a list of months, filter by those months
            mask |= (argo_ds.TIME.dt.year == year) & (argo_ds.TIME.dt.month.isin(season))

    # save mask to data path
    script_dir = pathlib.Path().parent.absolute()
    parent_dir = script_dir.parents[0]

    # make sure the directory exists
    mask_dir = parent_dir / 'data/profile masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask.to_netcdf(parent_dir / f'data/profile masks/mask_{case_str}.nc','w')
    print(f"Mask saved to {parent_dir / f'data/profile masks/mask_{case_str}.nc'}")

def derive_poly_func(input_file,  years, season='spring', degree=5, start_x=200, end_x=860, cut_edges=False, returnXY=False): 
    """
    Derives a polynomial function from the dataset based on the specified season and years.
    Here we use preprocessed Yomaha data that is referenced to 1000 dbar and rotated to be in cross direction of the AR7W line.
    """
    
    dataset    = xr.open_dataset(input_file)
    mask = utilities.select_datapoints_by_season_and_year(dataset, season, years)
    Ds = dataset.where(mask, drop=True)
    
    if cut_edges:
        Ds = Ds.where((Ds['x'] >= start_x) & (Ds['x'] <= end_x), drop=True)
    
    xA0, yA0 = Ds['x'].values, Ds['vs_adj'].values*100

    iS = np.argsort(xA0)
    xA0, yA0 = xA0[iS], yA0[iS]
    
    iN = np.isnan(yA0)
    x = xA0[~iN]
    y = yA0[~iN]

    poly_function = utilities.fit_of_running_mean(x, y, window_size=5, degree=degree)

    if returnXY:
        return poly_function, x, y
        
    return poly_function
    


def interpolate_profiles(values, z_data, z):
    """ Interpolate profiles to a common depth grid.
    Args:
        z_data (numpy.ndarray): Depth data.
        values (numpy.ndarray): Values to interpolate.
        z (numpy.ndarray): Common depth grid."""
    
    # Create empty arrays for interpolated values
    interpolated_values = np.empty((values.shape[0], len(z)))

    for k in range(values.shape[0]):  
        # Mask valid (non-NaN) values
        valid = ~np.isnan(z_data[k, :]) & ~np.isnan(values[k, :])  
        
        if np.sum(valid) > 1:  # Ensure at least two valid points for interpolation
            # Define maximum valid depth
            max_valid_z = np.min(z_data[k, valid])  
                
            f = scipy.interpolate.interp1d(
                z_data[k, valid], values[k, valid], kind='linear', bounds_error=False, fill_value="extrapolate")  
            
            # Apply interpolation only where z >= max_valid_z, otherwise set NaN with exception of max depth values close to 2000m (to generate some values there)
            if max_valid_z >= -1.93:
                interpolated_values[k, :] = np.where(z >= max_valid_z, f(z), np.nan)
            else:
                interpolated_values[k, :] = f(z)
        else:
            interpolated_values[k, :] = np.nan  # Assign NaN if no valid data points

    return interpolated_values


def convert_to_density_space(sigma0_grid, sigma0_int, var):
    """
    Convert variables to density space for 2D arrays.
    
    Args:
        sigma0_grid (1D array): Target density grid.
        sigma0_int (2D array): Input density values (2D grid, axis 1 is the sigma/z axis).
        var (2D array): Variable values corresponding to sigma0_int (2D grid, axis 1 is the sigma/z axis).
    
    Returns:
        2D array: Interpolated variable values on the sigma0_grid for each row.
    """
    # Initialize an array to store the interpolated results
    var_sigma = np.empty((sigma0_int.shape[0], len(sigma0_grid)))

    # Loop over rows
    for i in range(sigma0_int.shape[0]):
        # Sort sigma0_int and var for the current row
        sort_idx = np.argsort(sigma0_int[i, :])
        sigma0_sorted = sigma0_int[i, sort_idx]
        var_sorted = var[i, sort_idx]

        # Remove duplicates in sigma0_sorted
        _, unique_idx = np.unique(sigma0_sorted, return_index=True)
        sigma0_unique = sigma0_sorted[unique_idx]
        var_unique = var_sorted[unique_idx]

        # Interpolate to the target sigma0_grid
        f = scipy.interpolate.interp1d(sigma0_unique, var_unique, bounds_error=False, fill_value=np.nan)
        var_sigma[i, :] = f(sigma0_grid)

    return var_sigma



