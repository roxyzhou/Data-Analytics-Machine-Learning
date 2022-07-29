#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:36:43 2022

@author: roxyzhou
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#1.1
dt=pd.read_csv('/Users/roxyzhou/Downloads/imports-85.csv')
dt.dropna(subset=['price','horsepower','body-style'],inplace=True)
#original
fig,axs=plt.subplots(1,2,figsize=(15,6),sharey=True)
axs[0].scatter(dt['body-style'],dt['price'])
axs[1].scatter(dt['horsepower'],dt['price'])
plt.show()
#squared price
fig,axs=plt.subplots(1,2,figsize=(15,6),sharey=True)
axs[0].scatter(dt['body-style'],dt['price']**2)
axs[1].scatter(dt['horsepower'],dt['price']**2)
plt.show()
#log(price)
fig,axs=plt.subplots(1,2,figsize=(15,6),sharey=True)
axs[0].scatter(dt['body-style'],np.log(dt['price']))
axs[1].scatter(dt['horsepower'],np.log(dt['price']))
plt.show()
#Horsepower has a stonger relationship to log(price)

#1.2
import statsmodels.formula.api as sm
def ols_coef(x,formula):
    return sm.ols(formula,data=x).fit().params
result=ols_coef(dt,'np.log(price)~horsepower')
residual=np.log(dt.price)-(result[0]+result[1]*dt.horsepower)
plt.scatter(residual.axes,residual)
plt.hist(residual,bins=np.arange(-1.1,1.1,0.1))
#test correlation of residuals and variable
plt.scatter(result[0]+result[1]*dt.horsepower,residual)
plt.scatter(dt.horsepower,residual)
np.corrcoef(dt.horsepower,residual)
np.corrcoef(result[0]+result[1]*dt.horsepower,residual)
#ACf of residual
from statsmodels.tsa.stattools import acf
acf(residual)
sm.ols(data=dt,formula='np.log(price)~horsepower').fit().mse_resid
sm.acf(residual)

#1.3
plt.scatter(dt.horsepower,dt['city-mpg'])
dt.rename(columns={'city-mpg':'mpg'},inplace =True)
res=ols_coef(dt,'mpg ~ horsepower')
plt.plot(dt.horsepower,res[0]+res[1]*dt.horsepower)
plt.scatter(dt.horsepower,dt['mpg'])



#2.1
df=pd.read_csv('/Users/roxyzhou/Downloads/StockRetAcct_DT.csv')
df['exret']=np.exp(df.lnAnnRet)-np.exp(df.lnRf)
df.dropna(subset=['lnAnnRet','lnRf','lnIssue'],inplace=True)
df['decile']=df.groupby(['year'])['lnIssue'].transform(lambda x: pd.qcut(x,10,labels=range(1,11)))
averet=pd.DataFrame(index=set(df.year),columns=range(1,11))
def pfl_ave(y):
    for i in range(1,11):
        averet.loc[y][i]=sum(df[df.decile==i].MEwt/df[df.decile==i].MEwt.sum()*df[df.decile==i].exret)
for y in set(df.year):
    pfl_ave(y)
ave_ret_yr=averet.mean()

#2.2
plt.plot(range(1,11),ave_ret_yr)

#2.3
df['iss']=0
for i in df.index:
    if df['decile'][i]==1:
        df['iss'][i]=-1
    elif df['decile'][i]==10:
        df['iss'][i]=1
    else:
        df['iss'][i]=0
reg=df.groupby(['year']).apply(lambda x: ols_coef(x,'exret~iss'))
portfolio_ave_ret=reg['iss'].mean()
#Note that the mean for the transformed issuance-characteristic variable is 0. The final result(portfolio_ave_ret)\
#    denotes the average return over years to the portfolio where long stocks of 20% highest amount of issuance and short\
    # those of 20%lowest issuance. No position in the rest stocks belonging to Decile 2 to 9.
df.groupby(['year'])['iss'].mean()
df.iss.head(20)
#3.1 use value-weight inside a portfolio, equal weight across years
df['qtbm']=df.groupby(['year'])['lnBM'].transform(lambda x: pd.qcut(x, 5,labels=range(1,6)))
df['qtme']=df.groupby(['year'])['lnME'].transform(lambda x: pd.qcut(x, 5,labels=range(1,6)))
#3.2
def bmret(x):
    wt=x.MEwt
    er=x.exret
    return sum(er*wt/wt.sum())
a=df.groupby(['year','qtme','qtbm']).apply(bmret)
a=pd.DataFrame(a)
b=a.groupby(['qtme','qtbm'])[0].mean()
plt.plot(range(1,6),b[:5],label=('size=1'))
plt.plot(range(1,6),b[5:10],label=('size=2'))
plt.plot(range(1,6),b[10:15],label=('size=3'))
plt.plot(range(1,6),b[15:20],label=('size=4'))
plt.plot(range(1,6),b[20:25],label=('size=5'))
plt.legend()




