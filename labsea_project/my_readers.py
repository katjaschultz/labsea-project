import pathlib
import sys
import time

script_dir = pathlib.Path().parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

import argopy

argopy.set_options(mode='research')
from argopy import DataFetcher as ArgoDataFetcher

# readers.py: Will only read files.  Not manipulate them.


def fetch_argo_data_per_year(start_year=2004, end_year=2023, retries=5, wait_seconds=10):
    
    """
    Fetches Argo data for a specified region and time period.
    
    Parameters:
    start_year (int): The starting year for data collection (default is 2004).
    end_year (int): The ending year for data collection (default is 2023).
    Returns:
    ag (xarray.Dataset): An xarray dataset containing the Argo data for the specified region and time period.
    The region is defined as the Labrador Sea, covering latitudes from 45 to 68 and longitudes from -66 to -44, with a depth range of 0 to 2000 meters.
    """
      
    data_dir = parent_dir / 'data'
    data_dir.mkdir(exist_ok=True)

    for year in range(start_year, end_year + 1):
        file_path = data_dir / f'ArgoFetched_{year}.nc'

        # Skip if file already exists
        if file_path.exists():
            print(f"{year}: File already exists, skipping.")
            continue

        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        attempt = 0
        success = False

        while attempt < retries and not success:
            try:
                # Fetch Argo data for the specified region and time
                fetcher = ArgoDataFetcher(mode='research', timeout=10000)
                ds_year = fetcher.region([-66, -44, 45, 68, 0, 2000, year_start, year_end]).to_xarray()
                ds_points = ds_year.argo.point2profile()
                ds_points.argo.teos10(['SA', 'CT', 'PV'])

                # Save data to file
                ds_points.to_netcdf(file_path)
                print(f"{year}: Saved to {file_path.name}")
                ds_points.close()
                success = True

            except FileNotFoundError as e:
                attempt += 1
                print(f"{year}: Attempt {attempt} failed - {e}")
                time.sleep(wait_seconds)
                if attempt == retries:
                    print(f"{year}: Failed to fetch data after {retries} attempts. Retry later by running this script again. Note: it will automatically skip years that have already been downloaded.")
                    return

            except Exception as e:
                print(f"{year}: Unexpected error - {e}")
                break

    print("Finished downloading available yearly data.")


