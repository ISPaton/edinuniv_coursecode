#activate my36env
#required in venv: conda install -c conda-forge descartes
import pandas as pd
import numpy as np
import re
import urllib.request
from urllib.request import urlopen, Request
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import Point
import requests
import io
from io import StringIO
%matplotlib inline

#Part 1 - pull in station data from list of URLs and create a dataframe/table of data plus eastings/northings

#OS grid reference regex
eastreg = re.compile('\d{5,7}E')
northreg = re.compile('\d{5,7}N')
eastreg2 = re.compile('\d{5,7}')
northreg2 = re.compile('\d{5,7}')
#empty arrays for station data (everything except easting/northing) and station details (name, easting, northing) to be joined by name once cleaned
station_data = []
station_details = []
#reads in cleaned CSV from directory - this should be changed to read list direct from website landing page
urls_df = pd.read_csv('cleaned_station_list.csv')
urls = urls_df.URL
#headers are needed for website functionality
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"}
#populates list with scraped data from url list and incrementally adds each time
for x in urls:
    req = Request(url=x, headers=headers)
    html = urlopen(req)
    data = html.read().splitlines()
    #station data and location are the first elements in each iteration
    #print(data)
    station_name = data[0]
    station_location = data[1]
    station_location = station_location.decode('utf-8')
    station_location = str(station_location)
    del data[0:7]
    #regex to find XY coords and retain numeric elements
    easting = eastreg.findall(station_location)
    easting = str(easting)
    easting = eastreg2.findall(easting)
    easting = int(easting[0])
    northing = northreg.findall(station_location)
    northing = str(northing)
    northing = northreg2.findall(northing)
    northing = int(northing[0])
    #splitting up text into list
    data = [line.split() for line in data]
    #variable lengths is a problem with a core data matrix of 8 columns sometimes inconsistently appended to within and across datasets
    for i in data:
       if len(i) <6:
        del i
       elif len(i) > 7:
        del i[7:]
    for i in data:
        #adding station name, X and Y coordinates
        if len(i) == 7:
            i.append(station_name)
    #adds station_data each time to list and converts to dataframe
    station_data = station_data + data
    station_df = pd.DataFrame(station_data)
    #decodes station_details from byte format each iteration - this is a significant problem preventing a straightforward merge and later use
    station_name = station_name.decode('utf-8')
    station_details.append([station_name, easting, northing])
#converts station_list to dataframe and does a bulk decode from byte format using stack and unstack
station_details= pd.DataFrame(station_details)
station_df = station_df.stack().str.decode('utf-8').unstack()
#tidies up column headers and joins on name
station_df.columns = ['Year','Month', 'tMax_C', 'tMin_C', 'af_days', 'rain_mm', 'sun_hrs', 'name']
station_details.columns = ['name', 'easting', 'northing']
station_df = station_df.merge(station_details, how = 'right', left_on='name', right_on='name')
station_df
#write to CSV for later use in station-only exercise
station_df.to_csv (r'station_export.csv',index = False, header=True)

#Part 2 - for linear regression exercise, plots geometry for stations and loads table for happiness, mapping this to UA/LA/County from Ordnance Survey Boundary Line
# then performing a spatial join to transfer the attributes for happiness to the table for stations

#Plot spatial data
geometry = [Point(xy) for xy in zip(station_df['easting'],station_df['northing'])]
crs = {'init': 'epsg:27700'}
station_df = gpd.GeoDataFrame(station_df, crs=crs, geometry = geometry)
station_df.plot(figsize = (18,16))
happiness_geom = gpd.read_file('district_borough_unitary_region.shp')
happiness_geom = happiness_geom.to_crs({'init': 'epsg:27700'})
ax = happiness_geom.plot(figsize = (18,16))
station_df.plot(ax=ax,color='red')

happiness_data = pd.read_excel('geographicbreakdownreferencetable_tcm77-417203.xls', sheet_name = 'Happiness', header=[5])
happiness_data.drop(columns = ['Area names', 'Unnamed: 2', 'Unnamed: 3'], inplace=True)
happiness_data.drop(happiness_data.iloc[:, 6:22], inplace=True, axis=1)
#The below drops nulls and xs in preference to retaining a full set of attributes
happiness_data.dropna(subset =['Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'],   inplace=True)
happiness_data = happiness_data.drop(happiness_data[happiness_data['Unnamed: 4']=='x'].index)
happiness_data.columns = ['Area Codes','0-4', '5-6', '7-8', '9-10', 'Average']
happiness_map = happiness_geom.merge(happiness_data, left_on='CODE', right_on='Area Codes')
happiness_map.drop(columns = ['FILE_NAME', 'AREA_CODE', 'NUMBER', 'HECTARES', 'AREA', 'TYPE_CODE', 'DESCRIPTIO','DESCRIPT0', 'TYPE_COD0', 'DESCRIPT1'], inplace=True)
happiness_map.head()
station_happiness = gpd.sjoin(station_df, happiness_map, how='inner', op ='intersects')
#visual check to see if all stations have corresponding regions
ax = happiness_map.plot(figsize = (18,16))
station_df.plot(ax=ax,color='red')

#check missed joins
station_dfcheck = station_df.drop_duplicates(subset=['name'])
stationhappiness_dfcheck = station_happiness.drop_duplicates(subset=['name'])
missing_stations = pd.merge(stationhappiness_dfcheck,station_dfcheck, how ='right', left_on='name', right_on='name')
missing_stations = missing_stations[missing_stations['northing_x'].isnull()]
missing_stations
#16 stations are missing so a spatial join is used to join these to the regions for identification
missing_stations = gpd.GeoDataFrame(missing_stations, crs=crs, geometry=  'geometry_y')
missing_stations = missing_stations.drop(['index_right',], axis=1)
missing_stations2 = gpd.sjoin(happiness_geom,missing_stations, how='inner', op ='intersects')
ax = missing_stations2.plot(figsize = (18,16))
missing_stations.plot(ax=ax,color='red')
#inserts new codes and merges
new_codes= ['E10000003','E10000006','E10000008','E10000009','E10000011','E10000016','E10000019','E10000023','E10000024','E10000025','E10000027','E10000029','S12000013','S12000034','S12000027']
missing_stations2.insert(8, 'new_code',new_codes)

#reload to accept Xs
happiness_data = pd.read_excel('geographicbreakdownreferencetable_tcm77-417203.xls', sheet_name = 'Happiness', header=[5])
happiness_data.drop(columns = ['Area names', 'Unnamed: 2', 'Unnamed: 3'], inplace=True)
happiness_data.drop(happiness_data.iloc[:, 6:22], inplace=True, axis=1)
missing_happinessdata = pd.merge(happiness_data,missing_stations2, left_on='Area Codes', right_on='new_code')

missing_happinessdata.drop(['AREA_CODE', 'Area Codes_x','DESCRIPTIO', 'NAME_left', 'new_code','FILE_NAME', 'NUMBER', 'NUMBER0_left', 'POLYGON_ID_left', 'UNIT_ID_left', 'CODE_left', 'HECTARES', 'AREA', 'TYPE_CODE', 'DESCRIPT0', 'TYPE_COD0', 'DESCRIPT1', 'index_right', 'Year_x', 'Month_x', 'tMax_C_x', 'tMin_C_x', 'af_days_x', 'rain_mm_x', 'sun_hrs_x', 'easting_x', 'northing_x', 
'NAME_right', 'NUMBER0_right', 'geometry_x', 'geometry','POLYGON_ID_right', 'UNIT_ID_right', 'CODE_right', 'Area Codes_y', '0-4', '5-6', '7-8', '9-10', 'Average', 'Year_y', 'Month_y', 'tMax_C_y', 'tMin_C_y', 'af_days_y', 'rain_mm_y', 'sun_hrs_y', 'easting_y', 'northing_y'], inplace=True, axis=1)

missing_happinessdata.columns = ['0-4', '5-6', '7-8', '9-10', 'Average', 'name']
missing_happinessdata

#station_happiness = gpd.sjoin(station_df, happiness_map, how='inner', op ='intersects')
#station_happiness = station_happiness.drop(station_happiness[station_happiness['0-4']=='x'].index)
#drop  'index_right', 'NAME', 'NUMBER0', 'POLYGON_ID', 'UNIT_ID', 'CODE', 'Area Codes'
station_happiness2 = pd.merge(station_df,missing_happinessdata, left_on='name', right_on='name')

station_happiness.drop(columns = ['index_right', 'NAME','NUMBER0', 'POLYGON_ID', 'UNIT_ID', 'CODE', 'Area Codes'],inplace=True)
#write out station_happiness_export for later use in happiness exercise

station_happiness = station_happiness.append(station_happiness2)
#station_happiness = station_happiness.drop(station_happiness[station_happiness['0-4']=='x'].index)
station_happiness.to_csv (r'station_happiness_export.csv',index = False, header=True)
station_happiness.plot()
#B1_cluster_weather.py is the next file in the series



