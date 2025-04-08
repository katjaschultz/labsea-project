# to run this: python calc_weighted_mean.py data/argo_profiles_75kmAR7W_2004_to_2023.nc n_profiles_test.npy specvol_anom_weighted_test.npy
import argparse
import numpy as np
import gsw
import xarray as xr
import scipy.interpolate
import tqdm
import h5py

from tools import load_selected_profiles, interpolate_profiles

def main(filename, mask_profiles, output_file, folder_path, spacing_z, spacing_x, sigma, xstart, xend):
    
    # Load profiles
    x_data, z_data, specvol_anom, sigma0, SA, CT = load_selected_profiles(filename) # loads all profiles as default when mask_profiles is empty
   
    N, M = specvol_anom.shape  # number of profiles, levels
    
    # Define grid
    z = np.arange(0, 2000 + spacing_z, spacing_z) * -1e-3  # in km
    x = np.arange(xstart, xend + spacing_x, spacing_x)
    Mg, Ng = len(z), len(x)  # grid dimensions
    X, Z = np.meshgrid(x, z)
    grid_points = np.stack((X, Z))  # shape: (2, Mg, Ng)
    
    
    # Initialize output matrix for weighted specific volume anomaly
    specvol_anom_weighted = np.empty([Mg, Ng])
    sigma0_weighted = np.empty([Mg, Ng])
    SA_weighted = np.empty([Mg, Ng])
    CT_weighted = np.empty([Mg, Ng])

    # Interpolate on common z grid
    specvol_int = interpolate_profiles(specvol_anom, z_data, z)
    sigma0_int = interpolate_profiles(sigma0, z_data, z)
    SA_int = interpolate_profiles(SA, z_data, z)
    CT_int = interpolate_profiles(CT, z_data, z)    
    
    # Distance Matrix A[i, j] = |x_data[i,0] - grid_points[0,0,j]|
    A = np.abs(x_data[:, 0][:, None] - grid_points[0, 0, :][None, :])
    
    # For each grid point, store the indices of profiles that are within 50 km
    profiles_in_range = {j: np.where(A[:, j] <= 50)[0] for j in range(A.shape[1])}
  
    # Loop over grid points
    for j in tqdm.tqdm(range(Ng), desc='Processing Gridpoints'):
        n_profiles = np.where(mask_profiles)[0]
        n_selected = np.intersect1d(n_profiles, profiles_in_range[j]) # profiles provided by 'n_profiles' that are within 50 km 
        
        # load distances  
        file_path = folder_path + f'distances_gridpoint_{j}.h5'
        with h5py.File(file_path, 'r') as f:
            profile_dist = f[f'ngrid_{j}'][:]

        mask = np.intersect1d(n_selected, np.where(~np.isnan(profile_dist)))
        valid_dist = profile_dist[mask]
        valid_specvol = specvol_int[mask,:]
        valid_sigma0 = sigma0_int[mask,:]
        valid_SA = SA_int[mask,:]
        valid_CT = CT_int[mask,:]
    
        if np.all(np.isnan(valid_specvol)):
            specvol_anom_weighted[:, j] = np.nan
            sigma0_weighted[:, j] = np.nan
            SA_weighted[:, j] = np.nan
            CT_weighted[:, j] = np.nan
        else:
            for i in range(Mg):
                valid_z = ~np.isnan(valid_specvol[:,i])
                # gaussian weighting 
                weights = np.exp(-valid_dist[valid_z]**2 / (2 * sigma**2))
                weights /= np.sum(weights)
                if weights.size == 0:
                    specvol_anom_weighted[i, j] = np.nan
                    sigma0_weighted[i, j] = np.nan
                    SA_weighted[i, j] = np.nan
                    CT_weighted[i, j] = np.nan
                else:
                    specvol_anom_weighted[i, j] = np.nansum(valid_specvol[valid_z,i]* weights[:], axis=0)
                    sigma0_weighted[i, j] = np.nansum(valid_sigma0[valid_z,i] * weights[:], axis=0)
                    SA_weighted[i, j] = np.nansum(valid_SA[valid_z,i] * weights[:], axis=0)
                    CT_weighted[i, j] = np.nansum(valid_CT[valid_z,i] * weights[:], axis=0)
            
        
    # Save the result as a NumPy binary file
    np.save(output_file, [specvol_anom_weighted, sigma0_weighted, SA_weighted, CT_weighted], 'w')
    print(f"Weighted data saved to {output_file}")
  
if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(
        description="Compute weighted specific volume anomaly matrix from selected profiles."
    )
    parser.add_argument("filename", type=str, help='Input filename')
    parser.add_argument("mask_profiles_file", type=str,
                        help="Filename containing the NumPy array of profile indices (e.g., 'mask_profiles_default.npy')")
    parser.add_argument("output_file", type=str,
                        help="Output file name for the weighted matrix (e.g., 'specvol_anom_weighted.npy')")
    
    # Optional arguments with default values
    parser.add_argument("--spacing_z", type=float, default=25,
                        help="Vertical grid spacing in meters (default: 25)")
    parser.add_argument("--spacing_x", type=float, default=10,
                        help="Horizontal grid spacing in kilometers (default: 10)")
    parser.add_argument("--sigma", type=float, default=30.0,
                        help="Gaussian weighting sigma value (default: 30.0)")
    parser.add_argument("--xstart", type=float, default=200,
                        help="Starting x-coordinate in kilometers (default: 200)")
    parser.add_argument("--xend", type=float, default=860,
                        help="Ending x-coordinate in kilometers (default: 860)")

    args = parser.parse_args()
    
    # Load the array from file:
    mask_profiles = xr.open_dataarray(args.mask_profiles_file)
    
    if args.mask_profiles_file[-6:] == 'all.nc':
        folder_path = './data/distances_all/' 
        print('check')
    else:
        folder_path = './data/distances/'
        
    # Pass optional arguments to the main function
    main(args.filename, mask_profiles, args.output_file, folder_path, spacing_z=args.spacing_z, spacing_x=args.spacing_x, sigma=args.sigma, xstart=args.xstart, xend=args.xend)