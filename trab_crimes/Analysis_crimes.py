import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gmplot
import gmaps
import ipywidgets as widgets
import matplotlib.patches as pc
import sys

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
    plt.title("Frequency of Crimes")
    plt.ylabel('Type of crime', fontsize=12)
    plt.xlabel('Number of crimes', fontsize=12)
    plt.xlim(min(values)/4, 1.05 * max(values))
    plt.legend(handles=legend_patches)
    #plt.show()
    plt.savefig('crimes_freq.png', dpi=200)
    

def drop_coord_nan(data):
    data[['X Coordinate', 'Y Coordinate']] = data[['X Coordinate', 'Y Coordinate']].replace(0, np.nan)
    data[['Latitude', 'Longitude']] = data[['Latitude', 'Longitude']].replace(0, np.nan)
    data.dropna()
    #data = data.drop(columns=['X Coordinate','Y Coordinate'])
    

    return data

def draw_map(data):
    #latitude_list = [30.3358376, 30.307977, 30.3216419, 30.3427904, 30.378598, 30.3548185, 30.3345816, 30.387299, 30.3272198, 30.3840597, 30.4158, 30.340426, 30.3984348, 30.3431313, 30.273471] 
  
    #longitude_list = [77.8701919, 78.048457, 78.0413095, 77.886958,77.825396, 77.8460573, 78.0537813, 78.090614, 78.0355272, 77.9311923, 77.9663, 77.952092, 78.0747887, 77.9555512, 77.9997158] 
    
    #gmap1 = gmplot.GoogleMapPlotter()
    #gmap1.apikey = ""
    #data = gmaps.datasets.load_dataset('taxi_rides')
    #gmaps.heatmap(data)
    #gmap1.heatmap([41.8922738],[-87.815903],10,10)
    #gmap1.draw("/home/erjulioaguiar/Documentos/map14.html")
    #gmap1.display(h)

    #gmaps.configure(api_key='')
    #new_york_coordinates = [[40.75, -74.00]]

    #locations = fooDF[['latitude', 'longitude']]
    #fig = gmaps.figure()
    #fig = gmaps.figure(map_type='TERRAIN')
    #fig.add_layer(gmaps.heatmap_layer(new_york_coordinates))
    #fig

    lat = data['X Coordinate'].values
    longi = data['Y Coordinate'].values


    gmap = gmplot.GoogleMapPlotter(lat[0], longi[0], zoom=30)
    gmap.apikey = ''
    gmap.heatmap(lat[1:6], longi[1:6],radius=10)
    #gmaps.configure(api_key='')
    #gmap.plot(lat[1], longi[1], 'cornflowerblue')
    #gmap.scatter(lat[2:6], longi[2:6], '#3B0B39', size=40, marker=False)
    gmap.draw("crimes_map.html")

def correlation(data):
    c = data.corr()
    plt.rcParams['figure.figsize'] = (14,14)
    sns.heatmap(c, annot=True, linewidths=0.8)
    plt.title("Correlation Type of crime")
    plt.savefig('heat_map.png', dpi=170)
      
#def clean(data):
#    data = data.drop(columns=['Unnamed: 0','ID'])
#   data = drop_coord_nan(data)
#    data = remove_objects(data)

#    return data
    


if __name__ == '__main__':
    
    crimes = pd.read_csv('Chicago_Crimes_2012_to_2017.csv')
    
    global df_crimes
    
    if args[0] == 'correlacao':
        print('Calculando Correlacao ...')
        df_crimes = crimes.drop(columns=['Unnamed: 0','ID'])
        correlation(df_crimes)
        print('Completo!')
    elif args[0] == 'hist':
        print('Plotando o grafico de barra ...')
        df_crimes = drop_coord_nan(crimes)
        freq = remove_objects(df_crimes)
        plot_hist(freq)
        print('Completo!')
    else:
        print('Processando tudo ...')
        
        df_crimes = crimes.drop(columns=['Unnamed: 0','ID'])
        correlation(df_crimes)
        
        df_crimes = drop_coord_nan(crimes)
        freq = remove_objects(df_crimes)
        plot_hist(crimes)

        print('Completo!')