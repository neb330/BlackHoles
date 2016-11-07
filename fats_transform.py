
import matplotlib.pyplot as plt
from FATS import *
import seaborn
import numpy as np
import pandas as pd
from ipy_table import *
import os
# import seaborn for prettiness of plots; you don't have to do that!
import seaborn as sns

import astropy.io.fits as fits


def read_initial_data(path):
    '''Args: path to FITS data files.
    
       Returns: a list of time readings, a list of magnitude readings and a list of error readings.
    '''
    count = 0
    times = []
    rates = []
    errors = []
    file_handles = []
    for filename in os.listdir(path):
        with fits.open(path+filename, memmap=False) as example:
            hdu = example[1]
            time = hdu.data.field("TIME")
            rate = hdu.data.field("RATE")
            error = hdu.data.field("ERROR")
            times.append(time)
            rates.append(rate)
            errors.append(error)
            del example
    return times, rates, errors



def fats_transform(times, rates, errors):
    '''Args: 
            times - list of time readings
            rates - list of magnitude readings
            errors - list of error readings
       Returns:
            Pandas dataframe with FATS features for each time series
    '''
    features = []
    for time, error, rate in zip(times, errors, rates):
        ts = np.array([rate,time,error])
        a = FeatureSpace(Data=['magnitude','time','error'], featureList = None, excludeList=['interp1d','Color','Eta_color','Q31_color','StetsonJ','StetsonL'])
        a=a.calculateFeature(ts)
        feature = a.result(method = 'array')
        features.append(feature)
    labels = a.result(method = 'features')
    df = pd.DataFrame(features, columns = labels) 
    return df
      
    
def main():
    data_path = "nicedata_for_daniela/"
    times, rates, errors = read_initial_data(data_path)
    df = fats_transform(times, rates, errors)
    df.to_csv("time_series_fats.csv")
    
 
    
if __name__ == '__main__':
    main()