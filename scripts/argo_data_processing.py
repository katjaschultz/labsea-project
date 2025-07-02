import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.path import Path
from labsea_project import utilities
import pathlib
import sys

script_dir = pathlib.Path().parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

""" 
    This script processes Argo profile data by applying spatial filters.
    It can filter profiles based on their proximity to a specified line (AR7W line and 75km within) and bathymetric
    constraints within 1000isobars. It will automatically generate nc-files. """

class ArgoDataProcessor:
    
    def __init__(self, filename):
        self.dataset = xr.open_dataset(filename)
        self.filtered_indices = None

    def find_profiles_near_line(self, lon0, lat0, bbox, distance=75):
        lon_km, lat_km = utilities.ll2km(self.dataset.LONGITUDE.values, self.dataset.LATITUDE.values, bbox)
        lon_r, lat_r = utilities.rotate_point_corr(lon_km, lat_km)

        I = np.where(lat_km.ravel() >= -60)
        J = np.where(lon_km.ravel() >= 0)
        i75 = np.where((lat_r >= -distance) & (lat_r <= distance))
        K = np.intersect1d(i75, I, J)

        bool_array = np.zeros_like(self.dataset.LONGITUDE.values, dtype=bool)
        bool_array[K] = True
        self.filtered_indices = bool_array

    def apply_filter(self):
        if self.filtered_indices is not None:
            self.dataset = self.dataset.isel(N_PROF=self.filtered_indices)

class BathymetryPlotter:
    def __init__(self, topo_filename):
        topo_data = np.load(topo_filename)
        self.lon_topo = topo_data['lon_topo']
        self.lat_topo = topo_data['lat_topo']
        self.bathy = topo_data['bathy']

    def plot_bathymetry_and_argo(self, argo_dataset):
        ranges = np.arange(0, 5400, 500)
        fig, ax = plt.subplots(figsize=(6, 7))
        plt.contour(self.lon_topo, self.lat_topo, self.bathy, ranges, colors='grey', linewidths=1)
        plt.contour(self.lon_topo, self.lat_topo, self.bathy, [0], colors='k', linewidths=1)
        plt.contour(self.lon_topo, self.lat_topo, self.bathy, [1000], colors='b', linewidths=1, zorder=2)

        plt.plot(argo_dataset.LONGITUDE.values, argo_dataset.LATITUDE.values, '.r', zorder=1)
        
        contour_paths = [p.vertices for p in plt.gca().collections[2].get_paths()]
        contour1000_vertices = contour_paths[0]  # Assuming one merged path
        distances = np.sqrt(np.diff(contour1000_vertices[:, 0])**2 + np.diff(contour1000_vertices[:, 1])**2)
        
        split_indices = np.where(distances > 1)[0] + 1
        contour_segments = np.split(contour1000_vertices, split_indices)
        contour_segments = [seg for seg in contour_segments if len(seg) > 5]
        
        points_within_contour = np.zeros_like(argo_dataset.LONGITUDE.values, dtype=bool)
        for segment in contour_segments:
            contour1000_path = Path(segment)
            mask = contour1000_path.contains_points(np.column_stack((
                argo_dataset.LONGITUDE.values.flatten(),
                argo_dataset.LATITUDE.values.flatten()
            )))
            points_within_contour |= mask  
        
        plt.plot(argo_dataset.LONGITUDE.values[points_within_contour], argo_dataset.LATITUDE.values[points_within_contour], 'og', markersize=3, zorder=2)
        plt.plot(utilities.lon_line_A, utilities.lat_line_A, 'k', zorder=3)
        plt.plot(utilities.lon_line_1, utilities.lat_line_1, ':k', zorder=3)
        plt.plot(utilities.lon_line_0, utilities.lat_line_0, ':k', zorder=3)   
        plt.show()
    
        bool_array_2 = np.zeros_like(argo_dataset.LONGITUDE.values, dtype=bool)
        bool_array_2[points_within_contour] = True
        return bool_array_2

if __name__ == "__main__":
    argo_processor = ArgoDataProcessor(parent_dir /'data/LabSea_Argo_2004_2023.nc')

    Lon0, Lat0 = -55.73, 53.517        
    bbox = [Lon0, -44 , Lat0, 68]
    argo_processor.find_profiles_near_line(Lon0, Lat0, bbox)
    argo_processor.apply_filter()

    bathymetry_plotter = BathymetryPlotter(parent_dir / 'demo data/LabSea_topo_Argo.npz')
    filter_indices = bathymetry_plotter.plot_bathymetry_and_argo(argo_processor.dataset)

    ag_filtered = argo_processor.dataset.isel(N_PROF=filter_indices)
    ag = argo_processor.dataset

    ag.attrs['description'] = 'This dataset contains Argo profile data for the Labrador Sea for 2004-2023. Locations of profiles are filtered for the AR7W line and within 75 km distance of it.'
    ag_filtered.attrs['description'] = 'This dataset contains Argo profile data for the Labrador Sea for 2004-2023. Locations of profiles are filtered for the AR7W line and within 75 km distance of it and within 1000dB-isobars.'

    # filter dataset for duplicates (at same time)
    # Convert to DataFrame for easier manipulation
    df = ag[['TIME', 'N_PROF']].to_dataframe().reset_index()
    df_filtered = ag_filtered[['TIME', 'N_PROF']].to_dataframe().reset_index()

    duplicates = df[df.duplicated(subset='N_PROF', keep=False)]
    duplicates_filtered = df_filtered[df_filtered.duplicated(subset='N_PROF', keep=False)]

    duplicates_grouped = duplicates.groupby('N_PROF')['TIME'].nunique()
    duplicates_filtered_grouped = duplicates_filtered.groupby('N_PROF')['TIME'].nunique()

    # Identify N_PROF values where all duplicates have the same timestamp
    same_timestamp_n_prof = duplicates_grouped[duplicates_grouped == 1].index.tolist()
    same_timestamp_n_prof_filtered = duplicates_filtered_grouped[duplicates_filtered_grouped == 1].index.tolist()

    # Apply mask to drop unwanted N_PROF values
    mask = ~ag.N_PROF.isin(same_timestamp_n_prof)
    mask_filtered = ~ag_filtered.N_PROF.isin(same_timestamp_n_prof_filtered)

    # Apply filter
    ag_masked = ag.where(mask, drop=True)
    ag_filtered_masked = ag_filtered.where(mask_filtered, drop=True)

    ag_masked.to_netcdf(parent_dir / 'data/argo_profiles_75kmAR7W_2004_to_2023.nc')
    ag_filtered_masked.to_netcdf(parent_dir / 'data/argo_profiles_75kmAR7W_2004_to_2023_1000db_isobars.nc')

    ag.close()
    ag_filtered.close()
    ag_masked.close()
    ag_filtered_masked.close()

    print("Argo data processing complete. Datasets saved.")

