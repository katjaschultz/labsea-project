import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

def fit_of_running_mean(x, y, window_size=5, degree=5):

    x_smooth = []
    y_smooth = []

    for i in range(len(x) - window_size + 1):
        x_smooth.append(np.mean(x[i:i+window_size]))
        y_smooth.append(np.mean(y[i:i+window_size]))
        
    x_smoothR, y_smoothR = np.array(x_smooth), np.array(y_smooth)
        
    # Fit a polynomial of degree n
    coefficientsA = np.polyfit(x_smoothR, y_smoothR, degree)
    poly_functionA = np.poly1d(coefficientsA)

    #poly_functionA = scipy.interpolate.interp1d(x_smoothR, y_smoothR, kind='linear', fill_value='extrapolate')

    return poly_functionA

def map_month_to_season(month):
    if month in [4]:
        return 'April'
    elif month in [5]:
        return 'May'
    elif month in [8,9,10,11]:
        return 'Summer'
    elif month in [12, 1, 2, 3]:
        return 'Winter'
    elif month in [6]:
        return 'Jun'
    elif month in [7]:
        return 'Jul'

def map_year_to_intensity(year):
    if year in [2008, 2014, 2015, 2016, 2017, 2018, 2012, 2019,2020, 2022]:
        return 'strong'
    else:
        return 'weak'
        
def derive_poly_func(input_file, start_time, end_time, cut_season=False, season='spring', cut_intense=False, intensity='strong', degree=5, start_x=200, end_x=860, cut_edges=False, returnXY=False): #'2016-04-30'
    
    dataset       = xr.open_dataset(input_file)
    start_date = pd.to_datetime(start_time)
    end_date   = pd.to_datetime(end_time)
    time_mask = (dataset['datetime'] >= start_date) & (dataset['datetime'] <= end_date)
    ds = dataset.where(time_mask, drop=True)

    if cut_season:
        ds['season'] = xr.apply_ufunc(map_month_to_season, ds.datetime.dt.month, vectorize=True)
        seasonal_datasets = ds.groupby('season')

        if season == 'spring':
            ds1 = seasonal_datasets['April']
            ds2 = seasonal_datasets['May']
            ds3 = seasonal_datasets['Jun']
            ds = xr.concat([ds1, ds2, ds3], dim="datetime")
        elif season == 'summer':
            ds1 = seasonal_datasets['Summer']
            ds2 = seasonal_datasets['Jul']
            ds = xr.concat([ds1, ds2], dim="datetime")
        elif season == 'mayjunjul':
            ds1 = seasonal_datasets['May']
            ds2 = seasonal_datasets['Jun']
            ds3 = seasonal_datasets['Jul']
            ds = xr.concat([ds1, ds2, ds3], dim="datetime")
        else:
            ds = seasonal_datasets['Winter']
    else:
        Ds = ds

    if cut_intense:
        ds['intensity'] = xr.apply_ufunc(map_year_to_intensity, ds.datetime.dt.year, vectorize=True)
        intense_datasets = ds.groupby('intensity')

        if intensity == 'strong':
            Ds = intense_datasets['strong']
        else:
            Ds = intense_datasets['weak']
    else:   
        Ds = ds
    
    if cut_edges:
        Ds = Ds.where((Ds['x'] >= start_x) & (Ds['x'] <= end_x), drop=True)

    #plt.figure()
    #plt.scatter(ds.datetime.dt.year, ds.datetime.dt.month)
    #plt.show()
                
    xA0, yA0 = Ds['x'].values, Ds['vs_adj'].values*100

    iS = np.argsort(xA0)
    xA0, yA0 = xA0[iS], yA0[iS]
    
    iN = np.isnan(yA0)
    x = xA0[~iN]
    y = yA0[~iN]

    poly_function = fit_of_running_mean(x, y, window_size=5, degree=degree)

    if returnXY:
        return poly_function, x, y
        
    return poly_function
    

# yomaha       = xr.open_dataset('data/yomaha_dataset_refrenced_to_1000bar.nc')

'''
poly_func_2002_2016         = derive_poly_func( start_time='2002-03-01', end_time='2016-04-30')
poly_func_2002_2016_spring  = derive_poly_func(yomaha, start_time='2002-03-01', end_time='2016-04-30', cut_season=True, season='spring')
poly_func_2002_2016_summer  = derive_poly_func(yomaha, start_time='2002-03-01', end_time='2016-04-30', cut_season=True, season='summer')
poly_func_2002_2016_winter  = derive_poly_func(yomaha, start_time='2002-03-01', end_time='2016-04-30', cut_season=True, season='winter')

poly_func_2004_2023         = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31')
poly_func_2004_2023_spring  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31', cut_season=True, season='spring')
poly_func_2004_2023_summer  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31', cut_season=True, season='summer')
poly_func_2004_2023_winter  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31', cut_season=True, season='winter')

poly_func_2004_2023_strong         = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31', cut_intense=True, intensity='strong' )
poly_func_2004_2023_spring_strong  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31', cut_season=True, season='spring', cut_intense=True, intensity='strong')
poly_func_2004_2023_summer_strong  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31', cut_season=True, season='summer', cut_intense=True, intensity='strong')
poly_func_2004_2023_winter_strong  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31', cut_season=True, season='winter', cut_intense=True, intensity='strong')

poly_func_2004_2023_weak         = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31', cut_intense=True, intensity='weak' )
poly_func_2004_2023_spring_weak  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31', cut_season=True, season='spring', cut_intense=True, intensity='weak')
poly_func_2004_2023_summer_weak  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31', cut_season=True, season='summer', cut_intense=True, intensity='weak')
poly_func_2004_2023_winter_weak  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2023-12-31', cut_season=True, season='winter', cut_intense=True, intensity='weak')

poly_func_2004_2023_early         = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2013-12-31')
poly_func_2004_2023_spring_early  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2013-12-31', cut_season=True, season='spring')
poly_func_2004_2023_summer_early  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2013-12-31', cut_season=True, season='summer')
poly_func_2004_2023_winter_early  = derive_poly_func(yomaha, start_time='2004-01-01', end_time='2013-12-31', cut_season=True, season='winter')

poly_func_2004_2023_late         = derive_poly_func(yomaha, start_time='2014-01-01', end_time='2023-12-31')
poly_func_2004_2023_spring_late  = derive_poly_func(yomaha, start_time='2014-01-01', end_time='2023-12-31', cut_season=True, season='spring')
poly_func_2004_2023_summer_late  = derive_poly_func(yomaha, start_time='2014-01-01', end_time='2023-12-31', cut_season=True, season='summer')
poly_func_2004_2023_winter_late  = derive_poly_func(yomaha, start_time='2014-01-01', end_time='2023-12-31', cut_season=True, season='winter')
'''