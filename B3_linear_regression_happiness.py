#import sklearn into venv
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib as mpl
from shapely.geometry import Point
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

happiness_df = pd.read_csv(r'C:\\Users\paton\documents\_CODE\EdUniv_Course\final_assessment\station_happiness_export.csv')
happiness_df['tMax_C'] = happiness_df['tMax_C'].str.extract('(\d+)', expand=False)
happiness_df['tMin_C'] = happiness_df['tMin_C'].str.extract('(\d+)', expand=False)
happiness_df['af_days'] = happiness_df['af_days'].str.extract('(\d+)', expand=False)
happiness_df['rain_mm'] = happiness_df['rain_mm'].str.extract('(\d+)', expand=False)
happiness_df['sun_hrs'] = happiness_df['sun_hrs'].str.extract('(\d+)', expand=False)

#convert to float
happiness_df[['tMax_C', 'tMin_C','af_days','rain_mm','sun_hrs']] = happiness_df[['tMax_C', 'tMin_C','af_days','rain_mm','sun_hrs']].apply(pd.to_numeric)
happiness_df = happiness_df.dropna()
#happiness_df.drop(columns = ['NAME','NUMBER0','POLYGON_ID','UNIT_ID', 'CODE', 'Area Codes', 'geometry','index_right'],inplace = True)
happiness_df

#select month and year March 2015 and removes rows with x instead of a value
happiness1_df = happiness_df.drop(happiness_df[happiness_df['0-4']=='x'].index)
happiness1_df[['0-4', '5-6','7-8','9-10','Average']] = happiness1_df[['0-4', '5-6','7-8','9-10','Average']].apply(pd.to_numeric)
happinessquery_df = happiness1_df.loc[happiness_df['Year'] == 2015 ]
happinessquery_df = happinessquery_df.loc[happinessquery_df['Month'] == 3]

#for reference 10:11 is 1-4, 11:12 is 5-6, 12:13 is 7-8, 13:14 is 9-10, 15:16 is average 
#for reference 5:6 is rain_mm, 6:7 is sun_hrs

#first attempt at linear regression looking at higher band of happiness vs sun hours
y = happinessquery_df.iloc[:,13:14]
x = happinessquery_df.iloc[:,6:7]
reg=LinearRegression().fit(x,y)
plt.scatter(x,y)
m = reg.coef_
c = reg.intercept_
ypred=reg.predict(x)
plt.scatter(x,y)
plt.plot(x,ypred, color='red')

#second attempt at linear regression looking at higher band of happiness vs rain
y = happinessquery_df.iloc[:,13:14]
x = happinessquery_df.iloc[:,5:6]
reg=LinearRegression().fit(x,y)
plt.scatter(x,y)
m = reg.coef_
c = reg.intercept_
ypred=reg.predict(x)
plt.scatter(x,y)
plt.plot(x,ypred, color='red')

#third attempt at linear regression looking at average happiness vs sun hours
y = happinessquery_df.iloc[:,15:16]
x = happinessquery_df.iloc[:,6:7]
reg=LinearRegression().fit(x,y)
plt.scatter(x,y)
m = reg.coef_
c = reg.intercept_
ypred=reg.predict(x)
plt.scatter(x,y)
plt.plot(x,ypred, color='red')

#fourth attempt at linear regression looking at average happiness vs rain
y = happinessquery_df.iloc[:,15:16]
x = happinessquery_df.iloc[:,5:6]
reg=LinearRegression().fit(x,y)
plt.scatter(x,y)
m = reg.coef_
c = reg.intercept_
ypred=reg.predict(x)
plt.scatter(x,y)
plt.plot(x,ypred, color='red')

#consideration and removal of record 10583 as an outlier for sun
y = happinessquery_df.iloc[:,15:16]
x = happinessquery_df.iloc[:,6:7]
x = x.drop(x[x['sun_hrs']==74].index)
y = y.drop(y[y['Average']==7.81].index)
reg=LinearRegression().fit(x,y)
plt.scatter(x,y)
m = reg.coef_
c = reg.intercept_
ypred=reg.predict(x)
plt.scatter(x,y)
plt.plot(x,ypred, color='red')

#consideration and removal of record 10583 as an outlier for sun
y = happinessquery_df.iloc[:,15:16]
x = happinessquery_df.iloc[:,5:6]
x = x.drop(x[x['rain_mm']==193].index)
y = y.drop(y[y['Average']==7.81].index)
reg=LinearRegression().fit(x,y)
plt.scatter(x,y)
m = reg.coef_
c = reg.intercept_
ypred=reg.predict(x)
plt.scatter(x,y)
plt.plot(x,ypred, color='red')

#Use of the above with outlier removal in a multivariate model
happinessquery_df = happinessquery_df.drop(happinessquery_df[happinessquery_df['Average']==7.81].index)
reg=LinearRegression().fit(happinessquery_df[['rain_mm', 'sun_hrs']],happinessquery_df['Average'])
reg.coef_
#array([-0.00048167,  0.00017061])
reg.intercept_
#7.550195054846266

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
a=-0.00048167
b=0.00017061
c=7.55
ax.scatter( happinessquery_df['rain_mm'], happinessquery_df['sun_hrs'],happinessquery_df['Average'], c='blue', s=60)
p1, p2 = np.mgrid[1:200, 0:200:50] 
plane= a*p1 + b*p2 + c
ax.plot_surface(p1,p2,plane,color='red',alpha=0.7)


