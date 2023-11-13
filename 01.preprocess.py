########################################################################################################################
#    프로그램명    : preprocess.py
#    작성자        : misol lee
#    작성일자      : 2023.08.29
#    파라미터      : None
#    설명          : 가속도 데이터를 이용한 낙상 탐지를 위한 전처리
#       데이터 수집 경로 : https://github.com/Fall-Prevention-Team/sisfallData
########################################################################################################################

# set up ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import math
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, lfilter, freqz, filtfilt

pd.set_option('display.max_rows', None)
os.chdir('C:/Users/YJMS/Documents/미솔/fall/')

# set parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# butterworth filer
fs = 200  # Sampling frequency
fc = 5  # Cut-off frequency of the filter
order = 4
w = fc / (0.5 * fs)


# modules ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def uni_features(x):
    uni_stats = pd.DataFrame({
    'skew'  : [skew(x)],
    'kurt'  : [kurtosis(x)],
    'mean'  : [np.mean(x)],
    'med'   : [np.median(x)],
    'q25'   : [np.quantile(x, 0.25)],
    'q75'   : [np.quantile(x, 0.75)],
    'min'   : [np.min(x)],
    'max'   : [np.max(x)],
    'std'   : [np.std(x)],
    'iqr'   : [np.quantile(x, 0.75)-np.quantile(x, 0.25)],
    'range' : [np.max(x) - np.min(x)],
    'mad'   : [np.median(abs(x - np.median(x)))],
    'abs_std' : [np.std(abs(x))],
    'abs_range' : [np.max(abs(x))-np.min(abs(x))],
    'abs_rms' : [math.sqrt(np.mean(abs(x)**2))],
    'abs_overall' : [np.mean(abs((abs(x)**2) - (np.mean(abs(x)**2))))],
    'diff_mean' : [np.mean(np.diff(x))],
    })
    return(uni_stats)

def magnitude(data):
    return([math.sqrt(n) for n in (data ** 2).sum(1).values])

def sliding_window(data, window, overlap):
    start = list(range(0, data.shape[0], overlap))
    end = [np.min([x + window, data.shape[0]]) for x in start]
    return zip(start, end)

def feature_extract(dat, accl):
    x, y, z = accl[0], accl[1], accl[2]

    c1 = np.sqrt((dat[accl] ** 2).apply(sum, axis=1))
    c2 = np.sqrt((dat[[x, z]] ** 2).apply(sum, axis=1))
    c3 = np.sqrt(np.max(c1) - np.min(c1))
    c4 = np.arctan2(np.sqrt(dat[x] ** 2 + dat[z] ** 2), - dat[y]) * 180 / np.pi
    c5 = np.std(c4)
    c6 = np.mean(dat[accl].head(1)) * np.mean(dat[accl].tail(1))
    c7 = (dat[x].values[-1] - dat[x].values[0]) / (dat.time.values[-1] - dat.time.values[0])
    c8 = np.sqrt(dat[[x, z]].apply(np.var).sum())
    c9 = np.sqrt(dat[accl].apply(np.var).sum())
    d = [np.cumsum(((abs(dat[i]).rolling(2).sum() / 2) * (1 / fs))) for i in accl]
    c10 = ((d[0] + d[1] + d[2]) / dat['time']).fillna(0)
    c11 = ((d[0] + d[1]) / dat['time']).fillna(0)
    d = [np.cumsum(((dat[i].rolling(2).sum() / 2) * (1 / fs))) for i in accl]
    c14 = (np.sqrt(d[0] ** 2 + d[2] ** 2) / dat['time']).fillna(0)

    stat_name = uni_features(c1).columns
    stats = pd.DataFrame([c3, c5, c6, c7, c8, c9]).transpose()
    stats.columns = ['c3', 'c5', 'c6', 'c7', 'c8', 'c9']
    result = pd.concat([stats
                   , uni_features(c1).rename(columns=dict(zip(stat_name, 'c1_' + stat_name)))
                   , uni_features(c2).rename(columns=dict(zip(stat_name, 'c2_' + stat_name)))
                   , uni_features(c4).rename(columns=dict(zip(stat_name, 'c4_' + stat_name)))
                   , uni_features(c10).rename(columns=dict(zip(stat_name, 'c10_' + stat_name)))
                   , uni_features(c11).rename(columns=dict(zip(stat_name, 'c11_' + stat_name)))
                   , uni_features(c14).rename(columns=dict(zip(stat_name, 'c14_' + stat_name)))
                   , uni_features(dat[x]).rename(columns=dict(zip(stat_name, 'ax_' + stat_name)))
                   , uni_features(dat[y]).rename(columns=dict(zip(stat_name, 'ay_' + stat_name)))
                   , uni_features(dat[z]).rename(columns=dict(zip(stat_name, 'az_' + stat_name)))], axis=1)
    return result

# data preprocess ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dat_dir = 'data/sisfallData-main'
person_lis = np.setdiff1d(os.listdir(dat_dir), ['Readme.txt'])
#person = person_lis[0]

for person in person_lis:
    files = os.listdir(dat_dir+'/'+person)
    files = [f for f in files if person in f]
    feature_df = pd.DataFrame()
    #file = files[0]
    for file in files:
        # load data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        df = pd.read_csv(dat_dir+'/'+person+'/'+ file, sep=",|;", header = None, engine = 'python')
        df = df.iloc[:,0:9]
        df.columns = ['ax','ay','az','ax2','ay2','az2','zx','zy','zz']
        df = df[['ax','ay','az','zx','zy','zz']]
        df['time'] = df.index/fs
        # butterworth filter ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        features = np.setdiff1d(df.columns, 'time')
        b, a = butter(order, w, 'low')
        filtered = pd.DataFrame()
        for x in features:
            filtered = pd.concat([filtered, pd.DataFrame(lfilter(b, a, df[x]))], axis = 1)
        filtered.columns = 'f' + features
        df= pd.concat([df, filtered], axis = 1)

        # plt.plot(dt['acc_x'][1:200], label='original')
        # plt.plot(output[1:200], label='filtered')
        # plt.legend()
        # plt.savefig('out/butterfilter_example3.png')
        # plt.close()

        # extract features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        win = sliding_window(data=df, window=fs, overlap=round(fs*0.5))
        for e in win:
            dat = df.iloc[e[0]:e[1]]

            raw = feature_extract(dat = dat, accl = ['ax','ay','az'])
            filtered = feature_extract(dat = dat, accl = ['fax','fay','faz'])

            filtered.columns = 'f' + filtered.columns
            
            stat_df = pd.concat([raw,filtered], axis = 1)

            stat_df['window'] = dat.time.values[0]
            stat_df['id'] = person
            stat_df['activity'] = file.split('_')[0]
            stat_df['trial'] = file.split('_')[2].split('.')[0]
            feature_df = pd.concat([feature_df, stat_df])

    print(person, ':', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    feature_df.to_csv('data/feature_extract/'+person+'.csv')




