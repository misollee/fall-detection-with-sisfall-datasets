########################################################################################################################
#    프로그램명    : eda.py
#    작성자        : misol lee
#    작성일자      : 2023.08.29
#    파라미터      : None
#    설명          : 낙상에 대한 EDA
########################################################################################################################

# import modules ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os 
import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
os.chdir('/home/mslee/fall_detection')
pd.set_option('display.max_rows', None)

# load data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
person = 'SA02'

person_list = [s for s in os.listdir('data') if "csv" in s] 
person_list = [x.replace('.csv','') for x in person_list]

# total_df = pd.DataFrame()
# for person in person_list:
#     code = pd.read_excel('data/code_description.xlsx', engine = 'openpyxl')
#     df = pd.read_csv('data/'+person+'.csv', index_col = 0)

#     # timeseries plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#     for act in df['activity'].unique():
#         sub_df = df[df['activity']==act]
#         sub_df.index = range(sub_df.shape[0])
#         plt.plot(sub_df[['ax_mean', 'ay_mean','az_mean']], label=['x axis', 'y axis', 'z axis'])
#         plt.title(code[code['code']==act].iloc[0,1]+ '\n' + person)
#         plt.legend(loc = 'best')
#         plt.savefig('output/pattern/'+act+'_'+person+'.png',bbox_inches = 'tight')
#         plt.close()

#     total_df=pd.concat([total_df, df])

# total_df.shape
# total_df.to_csv('data/total_df.csv', index = False)

total_df = pd.read_csv('data/total_df.csv')
total_df.shape

total_df.head(50)
total_df['activity'][0][0]

total_df['fall'] = [x[0] for x in total_df['activity']]
total_df['fall'].value_counts()/total_df.shape[0] # 비율
total_df.isna().apply(sum, 0) # 결측 발생 원인 파악 

# boxplot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

features = total_df.dtypes
features =  features.index[features != 'object']
features = [x.replace('.1','') for x in features]
features = list(set(features))
features = np.setdiff1d(features, ['fall','window'])


total_df['fall'] = np.where(total_df.activity.isin(['F01','F02','F03','F04','F05','F06','F07']), 1,
                            np.where(total_df.activity.isin(['F08','F09','F10','F11','F12','F13','F14','F15']),1,0))

anova = pd.DataFrame()
for col in features:
    
    # col = features[1]
    
    # anova 
    model = ols(col+' ~ C(fall)', total_df).fit()
    pval = anova_lm(model)['PR(>F)'][0]
    f = round(anova_lm(model)['F'][0],3)
    
    anova = pd.concat([anova,pd.DataFrame({'feature': [col], 'F': [f], 'p' : [pval]})])
     
    # # boxplot
    # fig, ax = plt.subplots()
    # fig.set_figwidth(10)
    # fig.set_figheight(4)
    # sns.boxplot(data= total_df, x = 'activity', y = col).set(title = col + '(F : ' + str(f) + ')')
    # plt.xticks(rotation=45)
    # plt.savefig('output/boxplot/'+str(int(round(f)))+'_'+col+'.png',bbox_inches = 'tight')
    # plt.close()
    
anova = anova.sort_values('F', ascending=False)
anova.head(20)
anova.tail(20)
