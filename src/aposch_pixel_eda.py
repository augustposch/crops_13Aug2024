## aposch_pixel_eda.py ##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import os


def list_granule_paths(tile='10SFH',
                       year=2020,
                       hls_dir='../data/hls_23feb23',
                       sats=['L30','S30']):
    
    tile_slashes = f'{tile[:2]}/{tile[2]}/{tile[3]}/{tile[4]}'
    granule_paths = []
    
    for sat in sats:
        path = f'{hls_dir}/{sat}/{year}/{tile_slashes}'
        these_granules = os.listdir(path)
        granule_paths += [path+'/'+granule for granule in these_granules]
        
    return granule_paths


def list_filtered_paths(granule_paths,
                        pixel,
                        remove_if_clouds=True,
                        remove_if_shadow=True,
                        remove_if_adjacent=True,
                        remove_if_snow=False):
    granules = [gp[-34:] for gp in granule_paths]
    nbr_g = len(granules)
    band = 'Fmask'
    filtered_paths = []
    
    for i in range(nbr_g):
        tiff = f'{granules[i]}.{band}.tif'
        path = f'{granule_paths[i]}/{tiff}'
        
        try:
            with rasterio.open(path) as src:
                arr = src.read(1)
        except Exception:
            print('A file could not be read:', path)
            print('Continuing as normal with the rest of the files.')
            continue
        
        has_snow_remove_it = False
        if remove_if_snow:
            band2 = 'B02'
            tiff2 = f'{granules[i]}.{band2}.tif'
            path2 = f'{granule_paths[i]}/{tiff2}'
            try:
                with rasterio.open(path2) as src2:
                    arr2 = src2.read(1)
            except Exception:
                print('A file could not be read:', path2)
                print('Continuing as normal with the rest of the files.')
                continue
            value2 = arr2[pixel[0],pixel[1]]
            has_snow = value2>2000
            has_snow_remove_it = (has_snow and remove_if_snow)
            
        
        value = arr[pixel[0],pixel[1]]
        binary = '{0:b}'.format(value).zfill(8)
        has_clouds = bool(int(binary[-2]))
        has_adjacent = bool(int(binary[-3]))
        has_shadow = bool(int(binary[-4]))
        remove = ((has_clouds and remove_if_clouds) or
                  (has_adjacent and remove_if_adjacent) or
                  (has_shadow and remove_if_shadow) or
                  (has_snow_remove_it))
        
        if remove == False: 
            filtered_paths.append(granule_paths[i])
            
    print(f'After filtering, there are {len(filtered_paths)} granules remaining.')

    return filtered_paths
   

def gather_data(granule_paths, band, pixel):
    
    L30_dict = {'Blue':'B02', 'Green':'B03', 'Red':'B04',
               'NIR':'B05', 'SWIR1':'B06', 'SWIR2':'B07'}
    S30_dict = {'Blue':'B02', 'Green':'B03', 'Red':'B04',
               'NIR':'B8A', 'SWIR1':'B11', 'SWIR2':'B12'}
    
    refls = []
    DOYs = []
    sats = []
    
    for gp in granule_paths:
        granule = gp[-34:]
        DOY = int(granule[-15:-12])
        sat = granule[4:7]
        if sat == 'L30':
            band_code = L30_dict[band]
        if sat == 'S30':
            band_code = S30_dict[band]
        tiff = f'{granule}.{band_code}.tif'
        path = f'{gp}/{tiff}'
        
        try:
            with rasterio.open(path) as src:
                arr = src.read(1)
        except Exception:
            print('A file could not be read:', path)
            print('Continuing as normal with the rest of the files.')
            continue
            
        value = arr[pixel[0],pixel[1]]
        refls.append(value)
        DOYs.append(DOY)
        sats.append(sat)
            
    return refls, DOYs, sats


def explore_data(refls, DOYs, sats, save_loc, tags,
                 save=True, show=False):
    title = f'Reflectances for {" ".join(str(t) for t in tags)}'
    file_name = f'EDA_{"_".join(str(t) for t in tags)}.png'

    la = sum([sat == 'L30' for sat in sats]) # last landsat
    
    plt.figure(figsize=(20, 6))
    plt.scatter(DOYs[:la],
            refls[:la],
            c='orange',
            label='L30')
    plt.scatter(DOYs[la:],
            refls[la:],
            c='purple',
            label='S30')
    plt.title(title, size=14)
    plt.legend(loc='best')
    plt.ylabel('Reflectance')
    plt.xlabel('Day of Year')
    if save:
        plt.savefig(f'{save_loc}/{file_name}', bbox_inches='tight')
    if show:
        plt.show()
    if not show:
        plt.close()