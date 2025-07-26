import numpy as np
# import pymc as pm
# import arviz as az
import pandas as pd 
import pickle
import json
# import requests
from scipy import signal
from datetime import date,timedelta
# import yaml
import copy

import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

start = pd.to_datetime('2023-01-01')
eval_dates = [start + datetime.timedelta(days=7*j) for j in range(0,52)]


# # option 1: use CDPH data
# cases = pd.read_csv('data/covid19cases_test.csv',index_col=0)
# cases = cases[cases['area'] =='San Diego']
# cases = cases[(cases.index>='2021-06-01') & (cases.index<='2022-07-01')]
# cases.index = pd.to_datetime(cases.index)

# option 2: use SD county data, resolved by catchment. 
cases = pd.read_csv('data/cases.csv',index_col=0)
cases = cases[cases['catchment'] =='PointLoma']
cases = cases[(cases.index>='2021-04-01') & (cases.index<='2022-07-01')]
cases.index = pd.to_datetime(cases.index)
cases = cases.groupby(pd.Grouper(freq='D'))[['new_cases']].sum()
cases['cases'] = cases['new_cases'].rolling(window=7, center=True, min_periods=0).mean()
# cases['positivity'] = cases['cases'].rolling(window=7, center=True, min_periods=0).mean()/cases['total_tests'].rolling(window=7, center=True, min_periods=0).mean()
ww_dict = {'Point Loma':['data/PointLoma_sewage_seqs.csv','data/PointLoma_sewage_qPCR.csv']}
for site, files in zip(ww_dict.keys(),ww_dict.values()):
    df = pd.read_csv(f'{files[0]}')

    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = [dfc.split(' (')[0] for dfc in df.columns]
    df =df.set_index('Date')
    df = df[df.index>='2021-04-01']
    df = df[df.index<='2022-07-01']
    df = df.dropna(axis = 0, how = 'all')
    df = df.fillna(0)
    df = df/100.

    df = df.drop(columns=['Other'])
    df = df[df.columns[df.sum(axis=0) > 0.01]]

    cdf = pd.read_csv(f'{files[1]}')
    cdf['Sample_Date'] = pd.to_datetime(cdf['Sample_Date'])
    cdf =cdf.set_index('Sample_Date')

    cdf = cdf.resample('D').asfreq()
    cdf = cdf.rolling(window=7, center=True, min_periods=0).mean()
    sharedInds = np.sort(list(set(cdf.index) & set(cases.index)))
    cdf = cdf.loc[sharedInds]
    # df = df.loc[sharedInds]
    # df = df[~df.index.duplicated(keep='last')]
    # scaleddf = df.mul(cdf['Mean viral gene copies/L'],axis=0)



minInd = np.max([cdf.index.min(),cases.index.min()])
maxInd = np.min([cdf.index.max(),cases.index.max()])

cdf = cdf[cdf.index>=minInd]
cdf = cdf[cdf.index<=maxInd]

cases = cases[cases.index>=minInd]
cases = cases[cases.index<=maxInd]

cdf = pd.concat([cdf,cases],axis=1)
cdf = cdf.dropna(how='any')

N = 42
F = 14
# first, let's learn the shedding kernel
X = np.array([cdf['cases'].values[(j-F):(j+N-F)] for j in range(F,(cdf.shape[0]-N+F))])#/cdf['Mean viral gene copies/L'].mean()
Y = np.array(cdf['Mean viral gene copies/L'].values[F:(len(cdf['Mean viral gene copies/L'])-N+F)])#/cdf['Mean viral gene copies/L'].mean()

import cvxpy as cp
# least squares problem
x = cp.Variable(N)
cost = cp.norm(X @ x - Y,2)
## add some constraints on the shape of the curve. 
constraints = [x >= 0, x[0]==0,cp.diff(x[0:F])>=0., cp.abs(cp.diff(x)) <= 1000,cp.abs(cp.diff(x,2)) <= 50,cp.diff(x[N-20:N])<=0.]
prob = cp.Problem(cp.Minimize(cost),constraints)
prob.solve(verbose=True,solver=cp.CLARABEL)
lags = np.arange(N) - F
fig,ax = plt.subplots()
ax.plot(lags,x.value)
ax.set_xlabel("Days after detection")
fig.savefig('test_frequentist.pdf')

fig,ax = plt.subplots()
ax.plot(cdf.index,cdf['cases'],color='black')
ax2 = ax.twinx()
ax2.plot(cdf.index,cdf['Mean viral gene copies/L'])
ax.set_ylabel('Clinical cases')
ax2.set_ylabel('Mean viral gene copies/L')
fig.savefig('trajects.pdf')

shed_df = pd.DataFrame(np.array([lags,x.value]).T,columns=['lags','load'])
shed_df.to_csv('inferred_kernel.csv')


# X = np.array([cdf['cases'].values[(j-F):(j+N-F)] for j in range(F,(cdf.shape[0]-N+F))])#/cdf['cases'].mean()
# Y = np.array(cdf['Mean viral gene copies/L'].values[F:(len(cdf['Mean viral gene copies/L'])-N+F)])/cdf['Mean viral gene copies/L'].mean()

# norm = cdf['Mean viral gene copies/L'].mean()

# with pm.Model() as shedding_model:
#     # S = pm.Gamma("S", alpha = 10, beta=0.1, shape=N) 
#     S = pm.HalfFlat("S",shape=N)
#     mu = pm.math.dot(X, S)
#     # Y_obs = pm.LogNormal("Y_obs", mu=mu, observed=Y) #alpha = mu, beta=100, observed=Y)
#     Y_obs = pm.Gamma("Y_obs", alpha = mu/50, beta=50, observed=Y)
#     trace = pm.sample(1000, return_inferencedata=True)


# lags = np.arange(N) - F
# fig,ax = plt.subplots()
# az.plot_hdi(lags, trace.posterior["S"].values, hdi_prob=0.95,ax=ax)
# ax.plot(lags, np.median(trace.posterior["S"].values,axis=(0,1)), color="blue")
# ax.set_xlabel("Days before detection")
# fig.savefig('test_bayesian.pdf')

# plt.close('all')




