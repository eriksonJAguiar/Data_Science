import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gmplot
import gmaps
import ipywidgets as widgets
import matplotlib.patches as pc
import geopandas
from shapely.geometry import Point
import sys
import os
import math
from sklearn import preprocessing

args = sys.argv[1:]

sns.set()


def remove_objects(data):
    x = data['Primary Type'].value_counts()
    freq = pd.DataFrame()
    freq['index'] = x.index
    freq['values'] = x.values
    freq = freq.query('values > 800')


    return freq

def plot_hist(data):
    
    freq = data

    index = freq['index'].tolist()
    values = freq['values'].tolist()
    
    plt.rcParams['figure.figsize'] = (17,15)
    indx = np.arange(len(index)).tolist()
    b = sns.barplot(x=values, y=indx, alpha=1, orient='h')
    b.set_xticklabels(values, fontsize=10,rotation=90)
    b.set_yticklabels(indx, fontsize=8)
    Boxes = [item for item in b.get_children() if isinstance(item, pc.Rectangle)][:-1]
    legend_patches = [pc.Patch(color=C, label=L) for C, L in zip([item.get_facecolor() for item in Boxes], index )]  
    plt.title("Frequencia de crimes")
    plt.ylabel('Tipo de crime', fontsize=12)
    plt.xlabel('Numero de crimes', fontsize=12)
    plt.xlim(0, 1.05 * max(values))
    plt.legend(handles=legend_patches)
    #plt.show()
    plt.savefig('crimes_freq.png', dpi=200)
    

def drop_coord_nan(data):
    data = data.drop(columns=['X Coordinate', 'Y Coordinate'])
    #data[['X Coordinate', 'Y Coordinate']] = data[['X Coordinate', 'Y Coordinate']].replace(0, np.nan)
    data[['Latitude', 'Longitude']] = data[['Latitude', 'Longitude']].replace(0, np.nan)
    #data.dropna()
    #data = data.drop(columns=['X Coordinate','Y Coordinate'])
    
    data_rm = data[np.isnan(data['Latitude'])]
    data_rm = data[np.isnan(data['Longitude'])]

    index = data_rm.index.values.tolist()

    data = data.drop(index=index)

    return data

def draw_map(data):

    lat = data['Latitude'].values
    longi = data['Longitude'].values

    gmap = gmplot.GoogleMapPlotter(41.8781, -87.6298, 16,apikey='')
    #gmap.plot(lat, longi, 'cornflowerblue', edge_width=10)
    #gmap.scatter(lat, longi, '#3B0B39', marker=True)
    #gmap.plot(lat, longi, 'cornflowerblue')
    #gmap.scatter(lat, longi, 'k', marker=True)
    gmap.heatmap(lat, longi)
    #gmap.marker(lat,longi)
    gmap.draw("heat_map.html")


def correlation(data):
    c = data.corr()
    plt.rcParams['figure.figsize'] = (20,20)
    sns.heatmap(c, annot=True, linewidths=0.1)
    plt.title("Correlacao do tipo de crime")
    plt.savefig('heat_map.png', dpi=170)

def crimes_time(data):
    #data = data[data['Year'] != 2017]
    freq = data['Year'].value_counts()
    plt.rcParams['figure.figsize'] = (14,14)
    plt.plot(freq.index,freq.values)
    plt.title("Criminalidade durantes os anos")
    plt.ylabel('Numero de crimes', fontsize=12)
    plt.xlabel('Ano', fontsize=12)
    plt.savefig('crimes_ano2.png', dpi=200)
    #plt.show()

      
''' filter: NaN values '''

def nan_filter(data):
    
    data = data.dropna()
    
    
    return data

def obj_filter(data):
    
    obj_data = data.select_dtypes(include=['object']).copy()

    col = list(obj_data.columns)

    for i in range(len(col)):
        
        obj_data[col[i]] = obj_data[col[i]].astype('category')
        data[col[i]] = obj_data[col[i]].cat.codes
    
    return data

def normalization(data):
    
    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    res = pd.DataFrame(x_scaled, columns= data.columns, index=data.index)
    
    return res

def clean(data):
    new_data = nan_filter(data)
    new_data = obj_filter(new_data)
    new_data = normalization(new_data)

    return new_data


    


if __name__ == '__main__':
    
    crimes = pd.read_csv('Chicago_Crimes_2012_to_2017.csv')
    
    global df_crimes

    global remov 
    remov = {}
    remov["corr_filter"] = []
    remov["nan_filter"] = []
    remov["desc_filter"] = []

    print(crimes.count())

    if args[0] == 'correlacao':
        print('Calculando Correlacao ...')
        df_crimes = crimes.drop(columns=['Unnamed: 0','ID'])
        df_crimes = clean(df_crimes)
        correlation(df_crimes)
        print('Completo!')
    elif args[0] == 'hist':
        print('Plotando o grafico de barra ...')
        df_crimes = drop_coord_nan(crimes)
        freq = remove_objects(df_crimes)
        plot_hist(freq)
        print('Completo!')
    
    elif args[0] == 'maps':
        df_crimes = crimes.drop(columns=['Unnamed: 0','ID'])
        df_crimes = drop_coord_nan(crimes)
        draw_map(df_crimes)
    
    elif args[0] == 'time':
        df_crimes = crimes.drop(columns=['Unnamed: 0','ID'])
        df_crimes = drop_coord_nan(crimes)
        crimes_time(df_crimes)
    
    else:
        print('Processando tudo ...')
        
        df_crimes = crimes.drop(columns=['Unnamed: 0','ID'])
        correlation(df_crimes)
        
        df_crimes = drop_coord_nan(crimes)
        freq = remove_objects(df_crimes)
        plot_hist(crimes)

        print('Completo!')