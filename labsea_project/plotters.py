import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame

import matplotlib.pyplot as plt
import pathlib
import sys

script_dir = pathlib.Path().parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))



x_topo, topo = np.load(parent_dir / 'data/corrected_topography.npy')

# load colorbar ------------------------------------------------------------------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, Normalize

sampled_colors = np.load(parent_dir / 'data/colorbars/colorbar_pink_yellow_blue_green.npy')
vel_cmap = LinearSegmentedColormap.from_list('custom_cmap', sampled_colors, N=256)
value_range = [-0.3, -2.00000000e-01, -1.75000000e-01, -1.50000000e-01,
       -1.25000000e-01, -1.00000000e-01, -7.50000000e-02, -5.00000000e-02,
       -2.50000000e-02, 0.0, 2.50000000e-02,  5.00000000e-02,
        7.50000000e-02,  1.00000000e-01,  1.25000000e-01,  1.50000000e-01,
        1.75000000e-01,  2.00000000e-01, 0.3]
norm = BoundaryNorm(value_range, ncolors=vel_cmap.N, clip=True)
vticks_vel=[-0.2, -0.1, 0, 0.1, 0.2]
#cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=vel_cmap), ax=ax, ticks=vticks_vel, extend='both')


# -------------------------------------------------------------------------------------------------------------
# density contours
#axes[2].contour(dis, z, V, levels=np.arange(-0.3, 0.4, 0.1), colors='w', linewidth=0.8)
#axes[2].contour(x[1:-1], z, rho.T[:, 1:-1], levels=np.arange(27.6, 27.82, 0.02), colors='brown', linewidths=0.5, zorder=2)
#axes[2].contour(x[:], z, rho.T[:, :], [27.8], colors='k', linewidths=1.2, zorder=1)

def plot_abs_geo_v(v, sigma0half, Xhalf, Z, title_string, fig_string, saving=False):

    fig, ax = plt.subplots(1,1, figsize=(10,5))
    pc = ax.pcolormesh( Xhalf, Z*1e3, v, cmap=vel_cmap, norm=norm, zorder=0)
    ax.contour(Xhalf, Z*1e3, sigma0half, levels=np.arange(27.6, 27.82, 0.02), colors='k', linewidths=0.5)
    ax.contour(Xhalf, Z*1e3, sigma0half, [27.8] )
    ax.set_title(title_string, fontsize=12)
    ax.plot(x_topo, topo * -1, color=[.3,.3,.3], linewidth=0.5, zorder=3)
    ax.fill_betweenx(topo*-1, x_topo, where=(x_topo <= 205.622405357364), color=[.8,.8,.8], zorder=2)
    right_mask = (x_topo >= 855.0984735257368)
    ax.fill_betweenx(topo[right_mask]*-1, x_topo[right_mask], x2=1000, color=[.8,.8,.8], zorder=2)
    ax.set_xlim([155,925])
    ax.set_ylim([-2000,0])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=vel_cmap), ax=ax, ticks=vticks_vel, extend='both')
    cbar.set_label('velocity m/s')
    ax.set_xlabel('distance km')
    ax.set_ylabel('depth m')
    ax.axvline(x=205, color='y', linestyle=':', linewidth=2)
    ax.axvline(x=856, color='y', linestyle=':', linewidth=2)
    plt.tight_layout()
    if saving:
        plt.savefig(f'figures/abs_geo_v_{fig_string}.png')
    plt.show()





#----------------------------------------------------------------------------------------------------
# Template code for plotting functions for Seaglider data
#-----------------------------------------------------------------------------------------------------

'''

##------------------------------------------------------------------------------------
## Views of the ds or nc file
##------------------------------------------------------------------------------------
def show_contents(data, content_type='variables'):
    """
    Wrapper function to show contents of an xarray Dataset or a netCDF file.
    
    Parameters:
    data (str or xr.Dataset): The input data, either a file path to a netCDF file or an xarray Dataset.
    content_type (str): The type of content to show, either 'variables' (or 'vars') or 'attributes' (or 'attrs'). Default is 'variables'.
    
    Returns:
    pandas.io.formats.style.Styler or pandas.DataFrame: A styled DataFrame with details about the variables or attributes.
    """
    if content_type in ['variables', 'vars']:
        if isinstance(data, str):
            return show_variables(data)
        elif isinstance(data, xr.Dataset):
            return show_variables(data)
        else:
            raise TypeError("Input data must be a file path (str) or an xarray Dataset")
    elif content_type in ['attributes', 'attrs']:
        if isinstance(data, str):
            return show_attributes(data)
        elif isinstance(data, xr.Dataset):
            return show_attributes(data)
        else:
            raise TypeError("Attributes can only be shown for netCDF files (str)")
    else:
        raise ValueError("content_type must be either 'variables' (or 'vars') or 'attributes' (or 'attrs')")

def show_variables(data):
    """
    Processes an xarray Dataset or a netCDF file, extracts variable information, 
    and returns a styled DataFrame with details about the variables.
    
    Parameters:
    data (str or xr.Dataset): The input data, either a file path to a netCDF file or an xarray Dataset.
    
    Returns:
    pandas.io.formats.style.Styler: A styled DataFrame containing the following columns:
        - dims: The dimension of the variable (or "string" if it is a string type).
        - name: The name of the variable.
        - units: The units of the variable (if available).
        - comment: Any additional comments about the variable (if available).
    """
    from pandas import DataFrame
    from netCDF4 import Dataset

    if isinstance(data, str):
        print("information is based on file: {}".format(data))
        dataset = Dataset(data)
        variables = dataset.variables
    elif isinstance(data, xr.Dataset):
        print("information is based on xarray Dataset")
        variables = data.variables
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(variables):
        var = variables[key]
        if isinstance(data, str):
            dims = var.dimensions[0] if len(var.dimensions) == 1 else "string"
            units = "" if not hasattr(var, "units") else var.units
            comment = "" if not hasattr(var, "comment") else var.comment
        else:
            dims = var.dims[0] if len(var.dims) == 1 else "string"
            units = var.attrs.get("units", "")
            comment = var.attrs.get("comment", "")
        
        info[i] = {
            "name": key,
            "dims": dims,
            "units": units,
            "comment": comment,
            "standard_name": var.attrs.get("standard_name", ""),
            "dtype": str(var.dtype) if isinstance(data, str) else str(var.data.dtype),
        }

    vars = DataFrame(info).T

    dim = vars.dims
    dim[dim.str.startswith("str")] = "string"
    vars["dims"] = dim

    vars = (
        vars.sort_values(["dims", "name"])
        .reset_index(drop=True)
        .loc[:, ["dims", "name", "units", "comment", "standard_name", "dtype"]]
        .set_index("name")
        .style
    )

    return vars

def show_attributes(data):
    """
    Processes an xarray Dataset or a netCDF file, extracts attribute information, 
    and returns a DataFrame with details about the attributes.
    
    Parameters:
    data (str or xr.Dataset): The input data, either a file path to a netCDF file or an xarray Dataset.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the following columns:
        - Attribute: The name of the attribute.
        - Value: The value of the attribute.
    """
    from pandas import DataFrame
    from netCDF4 import Dataset

    if isinstance(data, str):
        print("information is based on file: {}".format(data))
        rootgrp = Dataset(data, "r", format="NETCDF4")
        attributes = rootgrp.ncattrs()
        get_attr = lambda key: getattr(rootgrp, key)
    elif isinstance(data, xr.Dataset):
        print("information is based on xarray Dataset")
        attributes = data.attrs.keys()
        get_attr = lambda key: data.attrs[key]
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(attributes):
        dtype = type(get_attr(key)).__name__
        info[i] = {
            "Attribute": key,
            "Value": get_attr(key),
            "DType": dtype
        }

    attrs = DataFrame(info).T

    return attrs

def show_variables_by_dimension(data, dimension_name='trajectory'):
    """
    Processes an xarray Dataset or a netCDF file, extracts variable information,
    and returns a styled DataFrame with details about the variables filtered by a specific dimension.
    
    Parameters:
    data (str or xr.Dataset): The input data, either a file path to a netCDF file or an xarray Dataset.
    dimension_name (str): The name of the dimension to filter variables by.
    
    Returns:
    pandas.io.formats.style.Styler: A styled DataFrame containing the following columns:
        - dims: The dimension of the variable (or "string" if it is a string type).
        - name: The name of the variable.
        - units: The units of the variable (if available).
        - comment: Any additional comments about the variable (if available).
    """

    if isinstance(data, str):
        print("information is based on file: {}".format(data))
        dataset = Dataset(data)
        variables = dataset.variables
    elif isinstance(data, xr.Dataset):
        print("information is based on xarray Dataset")
        variables = data.variables
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(variables):
        var = variables[key]
        if isinstance(data, str):
            dims = var.dimensions[0] if len(var.dimensions) == 1 else "string"
            units = "" if not hasattr(var, "units") else var.units
            comment = "" if not hasattr(var, "comment") else var.comment
        else:
            dims = var.dims[0] if len(var.dims) == 1 else "string"
            units = var.attrs.get("units", "")
            comment = var.attrs.get("comment", "")
        
        if dims == dimension_name:
            info[i] = {
                "name": key,
                "dims": dims,
                "units": units,
                "comment": comment,
            }

    vars = DataFrame(info).T

    dim = vars.dims
    dim[dim.str.startswith("str")] = "string"
    vars["dims"] = dim

    vars = (
        vars.sort_values(["dims", "name"])
        .reset_index(drop=True)
        .loc[:, ["dims", "name", "units", "comment"]]
        .set_index("name")
        .style
    )

    return vars

'''
