#import sklearn into venv 36
import pandas as pd
import numpy as np
import re
import matplotlib as mpl
import geopandas as gpd
from shapely.geometry import Point
import sklearn as sk
import sklearn.datasets as skd
import sklearn as sk
import sklearn.datasets as skd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#station_df.columns = ['Year','Month', 'tMax_C', 'tMin_C', 'af_days', 'rain_mm', 'sun_hrs', 'name']
stations_df = pd.read_csv(r'C:\\Users\paton\documents\_CODE\EdUniv_Course\final_assessment\station_export.csv')

#tidies up strings after conversion from bytes
stations_df['tMax_C'] = stations_df['tMax_C'].str.extract('(\d+)', expand=False)
stations_df['tMin_C'] = stations_df['tMin_C'].str.extract('(\d+)', expand=False)
stations_df['af_days'] = stations_df['af_days'].str.extract('(\d+)', expand=False)
stations_df['rain_mm'] = stations_df['rain_mm'].str.extract('(\d+)', expand=False)
stations_df['sun_hrs'] = stations_df['sun_hrs'].str.extract('(\d+)', expand=False)

#convert to float and remove nulls
stations_df[['tMax_C', 'tMin_C','af_days','rain_mm','sun_hrs']] = stations_df[['tMax_C', 'tMin_C','af_days','rain_mm','sun_hrs']].apply(pd.to_numeric)
stations_df = stations_df.dropna()

#removes last five stations as per task instruction
pd.unique(stations_df['name'])
samplestations_df=stations_df.query('name != "Yeovilton" & name != "Wick Airport" & name != "Whitby Coastguard / Whitby(from 2000 and $sunshine)" & name != "Waddington" & name != "Valley"')
samplestations_df


#Sets up two boundaries to split the UK into 3 bands and attributes this to stations based on northings in a new column 'band' where 3 is most northerly
n_max = samplestations_df['northing'].max()
n_min = samplestations_df['northing'].min()
n_diffband = (n_max - n_min)/3
n_bound1 = n_min + n_diffband
n_bound2 = n_max - n_diffband
samplestations_df['n_band'] = ""
samplestations_df['n_band'] = [3 if x>n_bound2 else (2 if x>n_bound1 else 1) for x in samplestations_df['northing']]

#sets up one boundary with easting 305090 (Livingston) to split UK into two east/west bands with 1 as west and 2 as  east
e_bound = 305090
samplestations_df['e_band'] = ""
samplestations_df['e_band'] = [2 if x>e_bound else 1 for x in samplestations_df['easting']]

#plot to check
geometry = [Point(xy) for xy in zip(samplestations_df['easting'],samplestations_df['northing'])]
crs = {'init': 'epsg:27700'}
samplestations_df= gpd.GeoDataFrame(samplestations_df, crs=crs, geometry = geometry)
samplestations_df.plot(figsize = (18,16), column ='n_band'))
samplestations_df.plot(figsize = (18,16), column ='e_band'))

#Max Temp and Daylight Hrs vs Band (Northing)  
#Select the month of June (or December) or maybe an equinox month
#Weight Hours by dividing by 30 for June (or December) (daily daylight)

samplestations_climate = samplestations_df.loc[samplestations_df['Month'] == 9]
samplestations_climate['sun_hrs'] /= 30 


#classification process
X = np.column_stack((samplestations_climate['sun_hrs'], samplestations_climate['tMax_C'], samplestations_climate['tMin_C']))
X = np.array(X)
y = np.array(samplestations_climate['n_band'])
plt.scatter(X[:,0], X[:,1], c=y)
classifier = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
plt.scatter(X_test[:,0], X_test[:,1], c=pred)
print(accuracy_score(y_test,pred))
#optimum fit for 2 variables is Month 9 with weighting to daily hours and tMax_C with 66% fit
#optimum fit for 3 variables is Month 9 with weighting to daily hours and tMax_C with 71% fit
#adding rain_mm as a 3rd or 4th variable lessens the fit - rainfall may be less definitive 

#as above but for rainfall e/w
X = np.column_stack((samplestations_climate['rain_mm'], samplestations_climate['tMax_C'], samplestations_climate['tMin_C']))
X = np.array(X)
y = np.array(samplestations_climate['e_band'])
plt.scatter(X[:,0], X[:,1], c=y)
classifier = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
plt.scatter(X_test[:,0], X_test[:,1], c=pred)
print(accuracy_score(y_test,pred))
#70%