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
    - mask_sigma: boolean, if True, mask the sigma values below 27.8 kg/m^3

'''
    # Load data from the specified file
    file_path = parent_dir / f"data/weighted data/weighted_data_{file_case}_{case}_omega{int(omega)}_xstart{str(xstart)}_xend{xend}.npy"

    specvol_anom, sigma, SA, CT = np.load(file_path)
    
    x = np.arange(xstart, xend + spacing_x, spacing_x)
    z = np.arange(0, 2000 + spacing_z, spacing_z) * -1 # negative depth
    # Create a new xarray dataset
    ds = xr.Dataset(
        {
            'specvol_anom': (['z', 'x'], specvol_anom),
            'sigma': (['z', 'x'], sigma),
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
                'sigma (potential density)': 'kg/m^3 \n',
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

    v, sigmahalf    = tools.derive_abs_geo_v(specvol_anom, sigma, P, argo_ref, lon_ar7w, lat_ar7w, xhalf, Zhalf, mask_sigma=mask_sigma)
    strf_z, strf_x, imbalance, mask_xa, piecewise_trapz_z, piecewise_trapz_x = tools.derive_strf(v, xhalf*1e3, z*-1)

    # add variables to dataset (coords: xhalf, z)
    # add xhalf as a coordinate
    ds = ds.assign_coords(xhalf=('xhalf', xhalf))
    ds['z0'] = (['z0'], z[1:]) # exclude 0 point from z-coordinate
    ds['x0'] = (['x0'], xhalf[1:]) # exclude 0 point from x-coordinate

    ds['v'] = (['z', 'xhalf'], v)
    ds['sigmahalf'] = (['z', 'xhalf'], sigmahalf)
    ds['strf_z'] = (['z'], strf_z)
    ds['strf_x'] = (['xhalf'], strf_x)
    ds['mask_xa'] = (['xhalf'], mask_xa.data)
    ds['piecewise_trapz_z'] = (['z0'], piecewise_trapz_z)
    ds['piecewise_trapz_x'] = (['x0'], piecewise_trapz_x)
    ds['argo_ref'] = (['xhalf'], argo_ref)
    ds['imbalance'] = imbalance

    return ds


def derive_adjusted_parameters(Ds, mean_sigma):
    # barotropic adj.
    # calc adujustment to balance transport in depth space
    v = Ds['v'].values.copy()
    xhalf = Ds['xhalf'].values
    z = Ds['z'].values
    sigmahalf = Ds['sigmahalf'].values.copy()

    c0 = tools.find_adjustment_velocity(v.copy(), xhalf*1e3, -1*z, onlyEast=False)
    strf_z_adj0, strf_x_adj0, imbalance_adj0, mask_xa_adj0, pw_trapz_z0, pw_trapz_x_adj0 = tools.derive_strf(v.copy(), xhalf*1e3, -1*z, sensitivity=c0, onlyEast=False)
    Ds['c0'] = c0

    max_depth = np.argwhere(z <=-1500)[0]
    MOC = strf_z_adj0[:max_depth[0]].max()
    depth_MOC = z[:max_depth[0]][np.argmax(strf_z_adj0[:max_depth[0]])]
    MOC_W = np.max(abs(strf_x_adj0)[:16]).round(3)
    MOC_E = np.max(abs(strf_x_adj0)[-16:]).round(3)

    # calculate accuracy by letting c0 vary +-1cm
    strf_z_adj0_cplus, _, _, _, _, _ = tools.derive_strf(v.copy(), xhalf*1e3, -1*z, sensitivity=c0+1*1e-2, onlyEast=False)
    MOC_cplus = strf_z_adj0_cplus[:max_depth[0]][np.argmax(strf_z_adj0[:max_depth[0]])]

    strf_z_adj0_cminus, _, _, _, _, _ = tools.derive_strf(v.copy(), xhalf*1e3, -1*z, sensitivity=c0-1*1e-2, onlyEast=False)
    MOC_cminus = strf_z_adj0_cminus[:max_depth[0]][np.argmax(strf_z_adj0[:max_depth[0]])]

    # declare new dataset to store the results
    Ds_adj = xr.Dataset({
            'v': (['z', 'xhalf'], v.copy()+c0),
            'c0': c0,
            'strf_z': (['z'], strf_z_adj0),
            'strf_x': (['xhalf'], strf_x_adj0),
            'imbalance': imbalance_adj0,
            'mask_xa': (['xhalf'], mask_xa_adj0.data),
            'piecewise_trapz_z': (['z0'], pw_trapz_z0),
            'piecewise_trapz_x': (['x0'], pw_trapz_x_adj0),
            'MOC': MOC,
            'depth_MOC': depth_MOC,
            'MOC_cplus': MOC_cplus,
            'MOC_cminus': MOC_cminus,
            'MOC_W': MOC_W,
            'MOC_E': MOC_E,
        },
        coords={
            'xhalf': ('xhalf', xhalf),
            'z': ('z', z),
            'z0': ('z0', z[1:]),
            'x0': ('x0', xhalf[1:]),
        }
    )

    # adj. at eastern edge
    c_east = tools.find_adjustment_velocity(v.copy(), xhalf*1e3, z*-1, onlyEast=True)
    Ds['c_east'] = c_east

    strf_z_adj_east, strf_x_adj_east, imbalance_adj_east, mask_xa_adj_east, pw_trapz_z_east, pw_trapz_x_adj_east = tools.derive_strf(v.copy(), xhalf*1e3, z*-1, sensitivity=c_east, onlyEast=True)

    MOC_east = strf_z_adj_east[:max_depth[0]].max()
    depth_MOC_east =z[:max_depth[0]][np.argmax(strf_z_adj_east[:max_depth[0]])]
    MOC_W_east = np.max(abs(strf_x_adj_east)[:16]).round(3)
    MOC_E_east = np.max(abs(strf_x_adj_east)[-16:]).round(3)

    # calculate accuracy by letting c0 vary +-1cm
    strf_z_adjeast_cplus, _, _, _, _, _ = tools.derive_strf(v.copy(), xhalf*1e3, -1*z, sensitivity=c_east+1*1e-2, onlyEast=True)
    MOC_cplus_east = strf_z_adjeast_cplus[:max_depth[0]][np.argmax(strf_z_adj_east[:max_depth[0]])]

    strf_z_adjeast_cminus, _, _, _, _, _ = tools.derive_strf(v.copy(), xhalf*1e3, -1*z, sensitivity=c_east-1*1e-2, onlyEast=True)
    MOC_cminus_east = strf_z_adjeast_cminus[:max_depth[0]][np.argmax(strf_z_adj_east[:max_depth[0]])]

    v_east = v.copy()
    v_east[:,-10:] = v_east[:,-10:] + c_east

    # Declare new dataset to store the results
    Ds_adj_east = xr.Dataset(
        {
            'v': (['z', 'xhalf'], v_east),
            'c_east': c_east,
            'strf_z': (['z'], strf_z_adj_east),
            'strf_x': (['xhalf'], strf_x_adj_east),
            'strf_z_cplus': (['z'], strf_z_adjeast_cplus),
            'imbalance': imbalance_adj_east,
            'mask_xa': (['xhalf'], mask_xa_adj_east.data),
            'piecewise_trapz_z': (['z0'], pw_trapz_z_east),
            'piecewise_trapz_x': (['x0'], pw_trapz_x_adj_east),
            'MOC': MOC_east,
            'depth_MOC': depth_MOC_east,
            'MOC_cplus': MOC_cplus_east,
            'MOC_cminus': MOC_cminus_east,
            'MOC_W': MOC_W_east,
            'MOC_E': MOC_E_east,
        },
        coords={
            'xhalf': ('xhalf', xhalf),
            'z': ('z', z),
            'z0': ('z0', z[1:]),
            'x0': ('x0', xhalf[1:]),
        }
    )

    # derive transport in density space, define sigma_bins (griding of transport in density space / bins)
    step = 0.005
    sigma_bins = np.linspace(27.2, 27.8, np.int64((27.8-27.2)/step)+1)

    # default: no adj.
    # Q0, Q = tools.derive_transport_in_density_space(v, sigmahalf, sigma_bins)
    Q0_adj, Q_adj = tools.derive_transport_in_density_space(v.copy(), sigmahalf, sigma_bins, sensitivity=c0)    
    Q0_east, Q_east = tools.derive_transport_in_density_space(v.copy(), sigmahalf, sigma_bins, sensitivity=c_east, onlyEast=True)

    scaled_depth = np.interp(sigma_bins, mean_sigma, z)
    max_scaled_depth = np.where(scaled_depth >= -100)[0][-1]
    sigma_dMOC = sigma_bins[max_scaled_depth:][np.argmax(Q_adj[max_scaled_depth:])]
    depth_dMOC = scaled_depth[max_scaled_depth:][np.argmax(Q_adj[max_scaled_depth:])]
    dMOC = Q_adj[max_scaled_depth:].max()

    sigma_dMOC_east = sigma_bins[max_scaled_depth:][np.argmax(Q_east[max_scaled_depth:])]
    depth_dMOC_east = scaled_depth[max_scaled_depth:][np.argmax(Q_east[max_scaled_depth:])]
    dMOC_east = Q_east[max_scaled_depth:].max()

    _, Q_adj_cplus = tools.derive_transport_in_density_space(v.copy(), sigmahalf, sigma_bins, sensitivity=c0+0.01)    
    _, Q_adj_cminus = tools.derive_transport_in_density_space(v.copy(), sigmahalf, sigma_bins, sensitivity=c0-0.01)  

    _, Q_east_cplus = tools.derive_transport_in_density_space(v.copy(), sigmahalf, sigma_bins, sensitivity=c_east+0.01, onlyEast=True)
    _, Q_east_cminus = tools.derive_transport_in_density_space(v.copy(), sigmahalf, sigma_bins, sensitivity=c_east-0.01, onlyEast=True)

    dMOC_cplus = Q_adj_cplus[max_scaled_depth:][np.argmax(Q_adj[max_scaled_depth:])]
    dMOC_cminus = Q_adj_cminus[max_scaled_depth:][np.argmax(Q_adj[max_scaled_depth:])]

    dMOC_east_cplus = Q_east_cplus[max_scaled_depth:][np.argmax(Q_east[max_scaled_depth:])]
    dMOC_east_cminus = Q_east_cminus[max_scaled_depth:][np.argmax(Q_east[max_scaled_depth:])]

    var_density = {'depth_dMOC': depth_dMOC,
                    'dMOC':dMOC,
                    'sigma_dMOC': sigma_dMOC,
                    'sigma_dMOC_east': sigma_dMOC_east,
                    'depth_dMOC_east': depth_dMOC_east,
                    'dMOC_east':  dMOC_east,
                    'dMOC_cplus': dMOC_cplus,
                    'dMOC_cminus': dMOC_cminus,
                    'dMOC_east_cplus': dMOC_east_cplus,
                    'dMOC_east_cminus': dMOC_east_cminus,
                    }

    return Ds_adj, Ds_adj_east, Q0_adj, Q_adj, Q0_east, Q_east, var_density

