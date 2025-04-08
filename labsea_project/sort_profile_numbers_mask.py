import xarray as xr
import pandas as pd
import numpy as np
import argparse
import pathlib

def main(filename, case_appendix):
    # Load dataset
    argo_ds = xr.open_dataset(filename, engine="netcdf4")
    #argo_ds = argo_ds.where(argo_ds.TIME.dt.year >= 2004, drop=True)
    
    for year in [2013, 2023]:
        for season in ['winter', 'summer', 'spring', 'mayjunjul']:

            if season == 'winter':                
                mask = (argo_ds.TIME.dt.year >= 2004) & (argo_ds.TIME.dt.year <= year) & \
                           ((argo_ds.TIME.dt.month == 1) | (argo_ds.TIME.dt.month == 2) | (argo_ds.TIME.dt.month == 12) | (argo_ds.TIME.dt.month == 3))
    
            elif season == 'spring':
                mask = (argo_ds.TIME.dt.year >= 2004) & (argo_ds.TIME.dt.year <= year) & \
                           ((argo_ds.TIME.dt.month == 4) | (argo_ds.TIME.dt.month == 5) | (argo_ds.TIME.dt.month == 6))
            
            elif season == 'summer':
                mask = (argo_ds.TIME.dt.year >= 2004) & (argo_ds.TIME.dt.year <= year) & \
                           ((argo_ds.TIME.dt.month == 7) | (argo_ds.TIME.dt.month == 8) | (argo_ds.TIME.dt.month == 9) | \
                                 (argo_ds.TIME.dt.month == 10) | (argo_ds.TIME.dt.month == 11))
            
            elif season == 'mayjunjul':
                mask = (argo_ds.TIME.dt.year >= 2004) & (argo_ds.TIME.dt.year <= year) & \
                           ((argo_ds.TIME.dt.month == 5) | (argo_ds.TIME.dt.month == 6) | (argo_ds.TIME.dt.month == 7))


            # Save masks
            if case_appendix == 'all':
                case = f'04{year-2000}{season}_{case_appendix}'
            else:
                case = f'04{year-2000}{season}'

            script_dir = pathlib.Path().parent.absolute()
            parent_dir = script_dir.parents[0]
        
            print(f"Processing: {case}")
            mask.to_netcdf(parent_dir / f'data/profile masks/mask_{case}.nc','w')
            

    strong_years = {2008, 2014, 2015, 2016, 2017, 2018, 2012, 2019, 2020, 2022}
    weak_years= {2004, 2005, 2006, 2007, 2009, 2010, 2011, 2013, 2021, 2023}
    
    for season in ['winter', 'summer', 'spring']:
        mask_strong = np.full(argo_ds.TIME.shape, False)  # Initialize strong mask
        mask_weak = np.full(argo_ds.TIME.shape, False)  # Initialize weak mask
        
        for year in strong_years:
            if season == 'winter':
                mask_strong |= (argo_ds.TIME.dt.year == year) & (
                    (argo_ds.TIME.dt.month == 1) | 
                    (argo_ds.TIME.dt.month == 2) | 
                    (argo_ds.TIME.dt.month == 12) | 
                    (argo_ds.TIME.dt.month == 3)
                )
        
            elif season == 'spring':
                mask_strong |= (argo_ds.TIME.dt.year == year) & (
                    (argo_ds.TIME.dt.month == 4) | 
                    (argo_ds.TIME.dt.month == 5) | 
                    (argo_ds.TIME.dt.month == 6)
                )
        
            elif season == 'summer':
                mask_strong |= (argo_ds.TIME.dt.year == year) & (
                    (argo_ds.TIME.dt.month == 7) | 
                    (argo_ds.TIME.dt.month == 8) | 
                    (argo_ds.TIME.dt.month == 9) |  
                    (argo_ds.TIME.dt.month == 10) | 
                    (argo_ds.TIME.dt.month == 11)
                )

        
        for year in weak_years:
            if season == 'winter':
                mask_weak |= (argo_ds.TIME.dt.year == year) & (
                    (argo_ds.TIME.dt.month == 1) | 
                    (argo_ds.TIME.dt.month == 2) | 
                    (argo_ds.TIME.dt.month == 12) | 
                    (argo_ds.TIME.dt.month == 3)
                )
        
            elif season == 'spring':
                mask_weak |= (argo_ds.TIME.dt.year == year) & (
                    (argo_ds.TIME.dt.month == 4) | 
                    (argo_ds.TIME.dt.month == 5) | 
                    (argo_ds.TIME.dt.month == 6)
                )
        
            elif season == 'summer':
                mask_weak |= (argo_ds.TIME.dt.year == year) & (
                    (argo_ds.TIME.dt.month == 7) | 
                    (argo_ds.TIME.dt.month == 8) | 
                    (argo_ds.TIME.dt.month == 9) |  
                    (argo_ds.TIME.dt.month == 10) | 
                    (argo_ds.TIME.dt.month == 11)
                )
            
        # Save masks
        if case_appendix == 'all':
            case_w = f'weak_{season}_{case_appendix}'
            case_s = f'strong_{season}_{case_appendix}'
        else:
            case_w = f'weak_{season}'
            case_s = f'strong_{season}'
    
        print(f"Processing: {case_s}")
        print(f"Processing: {case_w}")
        script_dir = pathlib.Path().parent.absolute()
        parent_dir = script_dir.parents[0]

        mask_weak.to_netcdf(parent_dir / f'data/profile masks/mask_{case_w}.nc','w')
        mask_strong.to_netcdf(parent_dir / f'data/profile masks/mask_{case_s}.nc','w')           
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute weighted specific volume anomaly matrix from selected profiles."
    )
    parser.add_argument("filename", type=str, help='input filename')
    
    parser.add_argument("case_appendix", type=str)
    args = parser.parse_args()
    
    main(args.filename, args.case_appendix)


