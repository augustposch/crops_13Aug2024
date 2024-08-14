import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from pandas.tseries.holiday import USFederalHolidayCalendar

from scipy import optimize

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def clean_data(df):
    '''
    Expects data in the form of the 2019 raw APC data, right after the .csv gets read in as a pandas DataFrame.
    
    Adds threshold features and time features.
    
    Removes bogus observations. Requires routes_against_calc and abnormal_stadir_routes .csv files to be present in the /data folder
    
    Result is a 1208360 by 58 DataFrame.
    '''

    for col in ['route finish time','route start time','stop arrival time']:
        df[col] = pd.to_datetime(df[col])

        
    df['Crowded'] = df['passwithin']>74
    df['Supercrowded'] = df['passwithin']>134
    df['Capacity'] = df['passwithin']>194
    df['UnderNeg5'] = df['passwithin']<-5
    df['NegAFew'] = df['passwithin'].between(-5,-1)
    df['Over250'] = df['passwithin']>250
    df['AllObservations'] = True

    df['Crowded000'] = df['passwithin']>0
    df['Crowded010'] = df['passwithin']>10
    df['Crowded020'] = df['passwithin']>20
    df['Crowded030'] = df['passwithin']>30
    df['Crowded040'] = df['passwithin']>40
    df['Crowded050'] = df['passwithin']>50
    df['Crowded060'] = df['passwithin']>60
    df['Crowded070'] = df['passwithin']>70
    df['Crowded080'] = df['passwithin']>80
    df['Crowded090'] = df['passwithin']>90
    df['Crowded100'] = df['passwithin']>100
    df['Crowded110'] = df['passwithin']>110
    df['Crowded120'] = df['passwithin']>120
    df['Crowded130'] = df['passwithin']>130
    df['Crowded140'] = df['passwithin']>140
    df['Crowded150'] = df['passwithin']>150
    df['Crowded160'] = df['passwithin']>160
    df['Crowded170'] = df['passwithin']>170
    df['Crowded180'] = df['passwithin']>180
    df['Crowded190'] = df['passwithin']>190
    df['Crowded200'] = df['passwithin']>200
    df['Crowded210'] = df['passwithin']>210
    df['Crowded220'] = df['passwithin']>220
    df['Crowded230'] = df['passwithin']>230
    df['Crowded240'] = df['passwithin']>240

    df['TOD'] = df['stop arrival time'].dt.time
    df['DOW'] = df['stop arrival time'].dt.dayofweek # 0 is Monday, 6 is Sunday
    df['DOW_name'] = df['stop arrival time'].dt.day_name()
    df['Date'] = df['stop arrival time'].dt.date
    df['Hour'] = df['stop arrival time'].dt.hour
    df['Minute'] = df['stop arrival time'].dt.minute
    df['Minute_od'] = df['Hour'] * 60 + df['Minute']
    df['Month'] = df['stop arrival time'].dt.month
    df['Month_name'] = df['stop arrival time'].dt.month_name()
    df['Season'] = df['stop arrival time'].dt.quarter
    df['DOY'] = df['stop arrival time'].dt.dayofyear
    df['WOY'] = df['stop arrival time'].dt.isocalendar().week

    # Create a station ID
    names = ['Zero','Angle','SeaTac','Tukwila','Rainier','Othello',
             'Columbia','Baker','Beacon','SODO','Stadium','Intl District',
             'Pioneer','University','Westlake','Capitol Hill ','UW ']

    for idx, name in enumerate(names):
        df.loc[df['station name']==name, 'sta_ID'] = idx
        df.loc[df['next station']==name, 'nxsta_ID'] = idx

    df['sta_ID'] = df['sta_ID'].astype('int32')
    df['nxsta_ID'] = df['nxsta_ID'].astype('int32')
    df['stadir_ID'] = df['sta_ID'] * 100 + df['nxsta_ID']


    # df01 filters out routedones that have >20 stops
    rtd = df.groupby('routedone').count()['railcar ID']
    rtd.name = 'count'
    overmuch = rtd[rtd>20]
    df01 = df[~df['routedone'].isin(overmuch.index)]


    # df02 is df01 but without >210 observations
    df02 = df01[~df01['Crowded210']]


    # df03 is df02 but without any against-equation trips
    against_calc = pd.read_csv('../data/routes_against_calc.csv').iloc[:,0]
    df03 = df02[~df02['routedone'].isin(against_calc)]

    
    # df04 is df03 but without any trips where the train visited stations out of sequence.
    # In other words, df04 consists of only normal stadir ID values.
    abnormal = pd.read_csv('../data/abnormal_stadir_routes.csv').iloc[:,0]
    df04 = df03[~df03['routedone'].isin(abnormal)]
    
    return df04


def create_byday(doy_avg, weekly_avg, year_prd_cycle):
    
    weekly_avg.name = 'WkMeanPassw'
    year_prd_cycle.name = 'Sinu01Passw'

    byday = pd.DataFrame()
    byday['Date'] = pd.date_range("2019-01-01", periods=365, freq="D")
    byday['DOW'] = byday['Date'].dt.day_of_week
    byday['WOY'] = byday['Date'].dt.isocalendar().week
    byday['DOY'] = byday['Date'].dt.day_of_year
    byday = byday.set_index('DOY')

    byday['DayMeanPassw'] = doy_avg
    byday = byday.join(weekly_avg, on='WOY')
    byday = byday.join(year_prd_cycle, on='WOY')
    byday['Removed01'] = byday['DayMeanPassw'] - byday['Sinu01Passw']

    return byday


def create_bydow(byday):
    # take the by-day df and group by day of week
    bydow = pd.DataFrame()
    bydow['Cycle02A'] = byday.groupby('DOW')['Removed01'].mean()
    bydow['Cycle02B'] = np.where(bydow.index<5, np.mean(bydow.loc[:4,'Cycle02A']),
                                np.mean(bydow.loc[5:,'Cycle02A']))
    return bydow


def improve_byday(byday, bydow):
    byday = byday.join(bydow, on='DOW')
    byday['Removed02A'] = byday['Removed01'] - byday['Cycle02A']
    byday['Removed02B'] = byday['Removed01'] - byday['Cycle02B']    
    return byday


def create_df05_train(df04_train, byday, bydow):
    df05_train = df04_train.copy()
    df05_train = df05_train.join(byday['Sinu01Passw'], on='WOY')
    df05_train = df05_train.join(bydow, on='DOW')
    df05_train['Removed01'] = df05_train['passwithin'] - df05_train['Sinu01Passw']
    df05_train['Removed02A'] = df05_train['Removed01'] - df05_train['Cycle02A']
    df05_train['Removed02B'] = df05_train['Removed01'] - df05_train['Cycle02B']
    return df05_train
    

def sinu(x, a, c, d):
    b = 6.28/52 # forces the period to be 1 year
    return a * np.sin(b * (x - c)) + d


def do_cycles_processing(df04_train, return_all_frames=False):

    pw_agg_doy = df04_train.groupby(['DOY'])['passwithin'].mean()
    pw_agg_woy = df04_train.groupby(['WOY'])['passwithin'].mean()

    params, params_covariance = optimize.curve_fit(sinu, pw_agg_woy.index, pw_agg_woy,
                                                   p0=[3, 13, 33])

    x_data = pw_agg_woy.index
    y_sinu_year = pd.Series(data=sinu(x_data, params[0], params[1], params[2]),
                       index=x_data)

    byday = create_byday(pw_agg_doy, pw_agg_woy, y_sinu_year)

    bydow = create_bydow(byday)

    byday = improve_byday(byday, bydow)

    df05_train = create_df05_train(df04_train, byday, bydow)

    if return_all_frames==True:
        return df05_train, byday, bydow

    if return_all_frames==False:
        return df05_train


    