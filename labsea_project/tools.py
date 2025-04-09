import numpy as np
import xarray as xr
import gsw
from labsea_project.utilities import ll2km, rotate_point_corr
import scipy
import xarray as xr
import scipy.integrate

import pathlib
import sys

script_dir = pathlib.Path().parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

# define grid
spacing_z, spacing_x = 25, 10
x_topo, topo = np.load(parent_dir / 'data/corrected_topography.npy')

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


def derive_abs_geo_v(specvol_anom, sigma0, p, ref_vel, lon_ar7w, lat_ar7w, xhalf, Z, p_ref=0):
   
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
    ind = [np.argmin(abs(x_topo - d)) for d in xhalf]
    mask_topo = Z <= topo[ind]*-1e-3

    v[mask_topo] = np.nan
    v[~mask] = np.nan
    sigma0half[mask_topo] = np.nan
    sigma0half[~mask] = np.nan
      
    return v, sigma0half

def derive_strf(v, x, z, returnDelta=False):

    """ Calculate the horizontal and vertical transport streamfunction from the velocity field.
    Args:
        v (numpy.ndarray): Velocity field.
        x (numpy.ndarray): x-coordinates.
        z (numpy.ndarray): Depth levels.
        returnDelta (bool): If True, return delta_x and delta_z.
    Returns:
        strf_z (numpy.ndarray): Vertical transport streamfunction.
        strf_x (numpy.ndarray): Horizontal transport streamfunction.
        imbalance (float): Imbalance of the streamfunction.
        mask_x (numpy.ndarray): Mask for horizontal transport streamfunction.
    Optional:
        delta_x (numpy.ndarray): Delta x-coordinates.
        delta_z (numpy.ndarray): Delta depth levels."""
      
    mask = ~np.isnan(v)
    vel    = xr.DataArray(data = v,
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

    if returnDelta:
        return strf_z, strf_x, imbalance, mask_x, delta_x, delta_z
    else:
        return strf_z, strf_x, imbalance, mask_x
    

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





# ------------------------------------------------------------------------------------------------
# template code for analysing Seaglider data 
# ------------------------------------------------------------------------------------------------
'''
# Various conversions from the key to units_name with the multiplicative conversion factor
unit_conversion = {
    'cm/s': {'units_name': 'm/s', 'factor': 0.01},
    'cm s-1': {'units_name': 'm s-1', 'factor': 0.01},
    'm/s': {'units_name': 'cm/s', 'factor': 100},
    'm s-1': {'units_name': 'cm s-1', 'factor': 100},
    'S/m': {'units_name': 'mS/cm', 'factor': 0.1},
    'S m-1': {'units_name': 'mS cm-1', 'factor': 0.1},
    'mS/cm': {'units_name': 'S/m', 'factor': 10},
    'mS cm-1': {'units_name': 'S m-1', 'factor': 10},
    'dbar': {'units_name': 'Pa', 'factor': 10000},
    'Pa': {'units_name': 'dbar', 'factor': 0.0001},
    'dbar': {'units_name': 'kPa', 'factor': 10},
    'degrees_Celsius': {'units_name': 'Celsius', 'factor': 1},
    'Celsius': {'units_name': 'degrees_Celsius', 'factor': 1},
    'm': {'units_name': 'cm', 'factor': 100},
    'm': {'units_name': 'km', 'factor': 0.001},
    'cm': {'units_name': 'm', 'factor': 0.01},
    'km': {'units_name': 'm', 'factor': 1000},
    'g m-3': {'units_name': 'kg m-3', 'factor': 0.001},
    'kg m-3': {'units_name': 'g m-3', 'factor': 1000},
}

# Specify the preferred units, and it will convert if the conversion is available in unit_conversion
preferred_units = ['m s-1', 'dbar', 'S m-1']

# String formats for units.  The key is the original, the value is the desired format
unit_str_format = {
    'm/s': 'm s-1',
    'cm/s': 'cm s-1',
    'S/m': 'S m-1',
    'meters': 'm',
    'degrees_Celsius': 'Celsius',
    'g/m^3': 'g m-3',
}


##-----------------------------------------------------------------------------------------------------------
## Calculations for new variables
##-----------------------------------------------------------------------------------------------------------
def calc_Z(ds):
    """
    Calculate the depth (Z position) of the glider using the gsw library to convert pressure to depth.
    
    Parameters
    ----------
    ds (xarray.Dataset): The input dataset containing 'PRES', 'LATITUDE', and 'LONGITUDE' variables.
    
    Returns
    -------
    xarray.Dataset: The dataset with an additional 'DEPTH' variable.
    """
    # Ensure the required variables are present
    if 'PRES' not in ds.variables or 'LATITUDE' not in ds.variables or 'LONGITUDE' not in ds.variables:
        raise ValueError("Dataset must contain 'PRES', 'LATITUDE', and 'LONGITUDE' variables.")

    # Initialize the new variable with the same dimensions as dive_num
    ds['DEPTH_Z'] = (['N_MEASUREMENTS'], np.full(ds.dims['N_MEASUREMENTS'], np.nan))

    # Calculate depth using gsw
    depth = gsw.z_from_p(ds['PRES'], ds['LATITUDE'])
    ds['DEPTH_Z'] = depth

    # Assign the calculated depth to a new variable in the dataset
    ds['DEPTH_Z'].attrs = {
        "units": "meters",
        "positive": "up",
        "standard_name": "depth",
        "comment": "Depth calculated from pressure using gsw library, positive up.",
    }
    
    return ds

def split_by_unique_dims(ds):
    """
    Splits an xarray dataset into multiple datasets based on the unique set of dimensions of the variables.

    Parameters:
    ds (xarray.Dataset): The input xarray dataset containing various variables.

    Returns:
    tuple: A tuple containing xarray datasets, each with variables sharing the same set of dimensions.
    """
    # Dictionary to hold datasets with unique dimension sets
    unique_dims_datasets = {}

    # Iterate over the variables in the dataset
    for var_name, var_data in ds.data_vars.items():
        # Get the dimensions of the variable
        dims = tuple(var_data.dims)
        
        # If this dimension set is not in the dictionary, create a new dataset
        if dims not in unique_dims_datasets:
            unique_dims_datasets[dims] = xr.Dataset()
        
        # Add the variable to the corresponding dataset
        unique_dims_datasets[dims][var_name] = var_data

    # Convert the dictionary values to a dictionary of datasets
    return {dims: dataset for dims, dataset in unique_dims_datasets.items()}



def reformat_units_var(ds, var_name, unit_format=unit_str_format):
    """
    Renames units in the dataset based on the provided dictionary for OG1.

    Parameters
    ----------
    ds (xarray.Dataset): The input dataset containing variables with units to be renamed.
    unit_format (dict): A dictionary mapping old unit strings to new formatted unit strings.

    Returns
    -------
    xarray.Dataset: The dataset with renamed units.
    """
    old_unit = ds[var_name].attrs['units']
    if old_unit in unit_format:
        new_unit = unit_format[old_unit]
    else:
        new_unit = old_unit
    return new_unit

def convert_units_var(var_values, current_unit, new_unit, unit_conversion=unit_conversion):
    """
    Convert the units of variables in an xarray Dataset to preferred units.  This is useful, for instance, to convert cm/s to m/s.

    Parameters
    ----------
    ds (xarray.Dataset): The dataset containing variables to convert.
    preferred_units (list): A list of strings representing the preferred units.
    unit_conversion (dict): A dictionary mapping current units to conversion information.
    Each key is a unit string, and each value is a dictionary with:
        - 'factor': The factor to multiply the variable by to convert it.
        - 'units_name': The new unit name after conversion.

    Returns
    -------
    xarray.Dataset: The dataset with converted units.
    """
    if current_unit in unit_conversion and new_unit in unit_conversion[current_unit]['units_name']:
        conversion_factor = unit_conversion[current_unit]['factor']
        new_values = var_values * conversion_factor
    else:
        new_values = var_values
        print(f"No conversion information found for {current_unit} to {new_unit}")
#        raise ValueError(f"No conversion information found for {current_unit} to {new_unit}")
    return new_values

def convert_qc_flags(dsa, qc_name):
    # Must be called *after* var_name has OG1 long_name
    var_name = qc_name[:-3] 
    if qc_name in list(dsa):
        # Seaglider default type was a string.  Convert to int8.
        dsa[qc_name].values = dsa[qc_name].values.astype("int8")
        # Seaglider default flag_meanings were prefixed with 'QC_'. Remove this prefix.
        if 'flag_meaning' in dsa[qc_name].attrs:
            flag_meaning = dsa[qc_name].attrs['flag_meaning']
            dsa[qc_name].attrs['flag_meaning'] = flag_meaning.replace('QC_', '')
        # Add a long_name attribute to the QC variable
        dsa[qc_name].attrs['long_name'] = dsa[var_name].attrs.get('long_name', '') + ' quality flag'
        dsa[qc_name].attrs['standard_name'] = 'status_flag'
        # Mention the QC variable in the variable attributes
        dsa[var_name].attrs['ancillary_variables'] = qc_name
    return dsa

def find_best_dtype(var_name, da):
    input_dtype = da.dtype.type
    if "latitude" in var_name.lower() or "longitude" in var_name.lower():
        return np.double
    if var_name[-2:].lower() == "qc":
        return np.int8
    if "time" in var_name.lower():
        return input_dtype
    if var_name[-3:] == "raw" or "int" in str(input_dtype):
        if np.nanmax(da.values) < 2**16 / 2:
            return np.int16
        elif np.nanmax(da.values) < 2**32 / 2:
            return np.int32
    if input_dtype == np.float64:
        return np.float32
    return input_dtype

def set_fill_value(new_dtype):
    fill_val = 2 ** (int(re.findall(r"\d+", str(new_dtype))[0]) - 1) - 1
    return fill_val

def set_best_dtype(ds):
    bytes_in = ds.nbytes
    for var_name in list(ds):
        da = ds[var_name]
        input_dtype = da.dtype.type
        new_dtype = find_best_dtype(var_name, da)
        for att in ["valid_min", "valid_max"]:
            if att in da.attrs.keys():
                da.attrs[att] = np.array(da.attrs[att]).astype(new_dtype)
        if new_dtype == input_dtype:
            continue
        _log.debug(f"{var_name} input dtype {input_dtype} change to {new_dtype}")
        da_new = da.astype(new_dtype)
        ds = ds.drop_vars(var_name)
        if "int" in str(new_dtype):
            fill_val = set_fill_value(new_dtype)
            da_new[np.isnan(da)] = fill_val
            da_new.encoding["_FillValue"] = fill_val
        ds[var_name] = da_new
    bytes_out = ds.nbytes
    _log.info(
        f"Space saved by dtype downgrade: {int(100 * (bytes_in - bytes_out) / bytes_in)} %",
    )
    return ds

'''
