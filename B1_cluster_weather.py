#import sklearn into venv
import pandas as pd
import numpy as np
import re
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import Point
import sklearn as sk
import sklearn.datasets as skd
import sklearn.cluster as skc
import seaborn as sns

#station_df.columns = ['Year','Month', 'tMax_C', 'tMin_C', 'af_days', 'rain_mm', 'sun_hrs', 'name']
stations_df = pd.read_csv(r'C:\\Users\paton\documents\_CODE\EdUniv_Course\final_assessment\station_export.csv')


stations_df['tMax_C'] = stations_df['tMax_C'].str.extract('(\d+)', expand=False)
stations_df['tMin_C'] = stations_df['tMin_C'].str.extract('(\d+)', expand=False)
stations_df['af_days'] = stations_df['af_days'].str.extract('(\d+)', expand=False)
stations_df['rain_mm'] = stations_df['rain_mm'].str.extract('(\d+)', expand=False)
stations_df['sun_hrs'] = stations_df['sun_hrs'].str.extract('(\d+)', expand=False)

#convert to float
stations_df[['tMax_C', 'tMin_C','af_days','rain_mm','sun_hrs']] = stations_df[['tMax_C', 'tMin_C','af_days','rain_mm','sun_hrs']].apply(pd.to_numeric)
stations_df = stations_df.dropna()
stations_df

##RANDOM SAMPLING
#size = 20      # sample size
#replace = True  # with replacement
#fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
#samplestations_df = stations_df.groupby('name', as_index=False).apply(fn)
samplestations_df = stations_df

#BUILD ARRAY
f1=samplestations_df['Month'].values
f2=samplestations_df['tMax_C'].values
f3=samplestations_df['tMin_C'].values
f4=samplestations_df['af_days'].values #may not be much use
f5=samplestations_df['rain_mm'].values
f6=samplestations_df['sun_hrs'].values
f7=samplestations_df['easting'].values
f8=samplestations_df['northing'].values

#test of E and N clustering for 35 stations - resemble map of UK
X = np.array(list(zip(f7,f8)))
model = skc.KMeans(n_clusters=35)
model.fit(X).score(X)
plt.scatter(X[:,0],X[:,1], c=model.labels_)
model.fit(X).score(X)

#test of daylight vs northing for 35 stations
X = np.array(list(zip(f6,f8)))
model = skc.KMeans(n_clusters=35)
model.fit(X).score(X)
plt.scatter(X[:,0],X[:,1], c=model.labels_)

#test of month vs sun for 12 months
X = np.array(list(zip(f1,f6,f8)))
model = skc.KMeans(n_clusters=12)
model.fit(X).score(X)
plt.scatter(X[:,1],X[:,2], c=model.labels_)

#test of month vs sun vs northing for 12 months
X = np.array(list(zip(f1,f6,f8)))
model = skc.KMeans(n_clusters=12)
model.fit(X).score(X)
plt.scatter(X[:,1],X[:,2], c=model.labels_)

#test of sun vs northing for a given month
samplestations_df = stations_df.loc[stations_df['Month'] == 6]
f1=samplestations_df['Month'].values
f2=samplestations_df['tMax_C'].values
f3=samplestations_df['tMin_C'].values
f4=samplestations_df['af_days'].values #may not be much use
f5=samplestations_df['rain_mm'].values
f6=samplestations_df['sun_hrs'].values
f7=samplestations_df['easting'].values
f8=samplestations_df['northing'].values
X = np.array(list(zip(f6,f8)))
model = skc.KMeans(n_clusters=4)
model.fit(X).score(X)
plt.scatter(X[:,0],X[:,1], c=model.labels_)

#test of rain vs easting for a given month
X = np.array(list(zip(f5,f7)))
model = skc.KMeans(n_clusters=3)
model.fit(X).score(X)
plt.scatter(X[:,0],X[:,1], c=model.labels_)

#test of month vs minimum and maximum temperatures for 4 seasons
samplestations_df = stations_df
f1=samplestations_df['Month'].values
f2=samplestations_df['tMax_C'].values
f3=samplestations_df['tMin_C'].values
f4=samplestations_df['af_days'].values #may not be much use
f5=samplestations_df['rain_mm'].values
f6=samplestations_df['sun_hrs'].values
f7=samplestations_df['easting'].values
f8=samplestations_df['northing'].values
X = np.array(list(zip(f2,f3)))
model = skc.KMeans(n_clusters=4)
model.fit(X).score(X)
plt.scatter(X[:,0],X[:,1], c=model.labels_)

#test of minimum and maximum temperatures and rain and sun for 4 seasons
X = np.array(list(zip(f2,f3,f5,f6)))
model = skc.KMeans(n_clusters=4)
model.fit(X).score(X)
plt.scatter(X[:,2],X[:,3], c=model.labels_)

#test of rain and sun for 4 seasons
X = np.array(list(zip(f5,f6)))
model = skc.KMeans(n_clusters=4)
model.fit(X).score(X)
plt.scatter(X[:,0],X[:,1], c=model.labels_)

#CHECK THE BELOW!!!! MAy NOT BE COMPLETE!!!***************************************

#test of 4 climates in a cold/warm dry/wet matrix
samplestations = stations_df.loc[stations_df['Month'] == 6]
X = np.array(list(zip(f2,f3,f5)))#SELECT VALUES
model = skc.KMeans(n_clusters=2)
model.fit(X).score(X)
plt.scatter(X[:,0],X[:,1], c=model.labels_)

#test of  climates in a cold/dry vs warm/wet split
samplestations = stations_df.loc[stations_df['Month'] == 6]
X = np.array(list(zip(f2,f3,f5)))#SELECT VALUES
model = skc.KMeans(n_clusters=2)
model.fit(X).score(X)
plt.scatter(X[:,0],X[:,1], c=model.labels_)

#SEABORN
x_df = pd.DataFrame(X)
sns.set(style="ticks")
sns.pairplot(x_df)

sns.set(style="ticks")
sns.pairplot(stations_df,hue = f1)#name does not work, check df example

#DBSCAN
model = skc.DBSCAN(eps=0.5)
model.fit(X)
plt.scatter(X[:,0],X[:,1],c=model.labels_)




