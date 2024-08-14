import numpy as np
import os
import rasterio
from scipy.interpolate import CubicSpline


def filter_granule(granule_path,
                        mark_if_clouds=True,
                        mark_if_shadow=True,
                        mark_if_adjacent=True,
                        mark_if_snow=False):

    granule = granule_path[-34:]
    
    fmask_tiff = f'{granule}.Fmask.tif'
    fmask_path = f'{granule_path}/{fmask_tiff}'

    try:
        with rasterio.open(fmask_path) as src:
            fmask_arr = src.read(1).flatten()
    except Exception:
        print('An fmask file could not be read:', fmask_path)
        print('Marking that granule as bad data.')
        # If we can't read the fmask, we need to ditch this granule
        fin = [np.full((3660*3660,), -9999,
                       dtype='int16') for _ in range(7)]
        return fin

    marked = []
    for el in fmask_arr:
        binary = '{0:b}'.format(el).zfill(8)
        has_clouds = bool(int(binary[-2]))
        has_adjacent = bool(int(binary[-3]))
        has_shadow = bool(int(binary[-4]))
        mark = ((has_clouds and mark_if_clouds) or
              (has_adjacent and mark_if_adjacent) or
              (has_shadow and mark_if_shadow))
        marked.append(mark)    
    cas_mask = np.array(marked)
    
    ### SNOW PROCESSING
    if mark_if_snow:
        band2 = 'B02'
        tiff2 = f'{granule}.{band2}.tif'
        path2 = f'{granule_path}/{tiff2}'
        try:
            with rasterio.open(path2) as src2:
                arr2 = src2.read(1).flatten()
        except Exception:
            print('A file could not be read:', path2)
            print('Cannot determine snow, so we cannot use this granule any further.')
            print('Marking that granule as bad data.')
            fin = [np.full((3660*3660,), -9999,
                       dtype='int16') for _ in range(7)]
            return fin
        snow_mask = arr2>2000
        final_mask = snow_mask | cas_mask
    else:
        final_mask = cas_mask
    
    ### For the 6 color bands,
    ### Read into arrays and flatten
    L30_dict = {'Blue':'B02', 'Green':'B03', 'Red':'B04',
               'NIR':'B05', 'SWIR1':'B06', 'SWIR2':'B07'}
    S30_dict = {'Blue':'B02', 'Green':'B03', 'Red':'B04',
               'NIR':'B8A', 'SWIR1':'B11', 'SWIR2':'B12'}
    bands = ['Blue','Green','Red','NIR','SWIR1','SWIR2']
    DOY = int(granule[-15:-12])
    sat = granule[4:7]
    
    filtered_granule = []
    
    for band in bands:
        if sat == 'L30':
            band_code = L30_dict[band]
        if sat == 'S30':
            band_code = S30_dict[band]
        
        band_tiff = f'{granule}.{band_code}.tif'
        band_path = f'{granule_path}/{band_tiff}'

        try:
            with rasterio.open(band_path) as src:
                band_arr = src.read(1).astype('int16').flatten()
        except Exception:
            print('A band file could not be read:', band_path)
            print('Marking that as bad data and continuing as normal with the rest of the files.')
            # If we can't read this color band, mark it as all -9999
            band_arr = np.full(final_mask.shape, -9999, dtype='int16')
        
        filtered_granule.append(band_arr)

    filtered_granule.append(final_mask)
    
    # return filtered_granule as a list of 7 flat arrays
    # the 7 arrays are: blue, greem, red, NIR, SWIR1, SWIR2, mask
    return filtered_granule 


def list_granule_paths(tile='10SFH',
                       year=2020,
                       hls_dir='../data/hls_23feb23',
                       sats=['L30','S30'],
                       period_start=1,
                       period_end=366):
    
    tile_slashes = f'{tile[:2]}/{tile[2]}/{tile[3]}/{tile[4]}'
    
    granule_paths = []
    
    for sat in sats:
        path = f'{hls_dir}/{sat}/{year}/{tile_slashes}'
        these_granules = os.listdir(path)
        
        this_sat_paths = []
        for granule in these_granules:
            DOY = int(granule[-15:-12])
            if DOY>=period_start and DOY<=period_end:
                this_sat_paths.append(path+'/'+granule)
            
        granule_paths += this_sat_paths
    
    return granule_paths


def composite_one_period(tile='10SFH',
                        year=2020,
                        period_start=91,
                        period_end=104,
                        hls_dir='../data/hls_23feb23',
                        sats=['L30','S30'],
                        mark_if_clouds=True,
                        mark_if_shadow=True,
                        mark_if_adjacent=True,
                        mark_if_snow=False):

    granules_in_period = list_granule_paths(period_start=period_start,
                                           period_end=period_end,
                                           year=year,
                                           tile=tile,
                                           hls_dir=hls_dir,
                                           sats=sats)

    ## Fill list_of_lists with all the data to be composited
    list_of_lists = [[] for _ in range(7)]
    for granule_path in granules_in_period:
        # Filter the granule
        filtered_granule = filter_granule(granule_path,
                                          mark_if_clouds=mark_if_clouds,
                                          mark_if_shadow=mark_if_shadow,
                                          mark_if_adjacent=mark_if_adjacent,
                                          mark_if_snow=mark_if_snow)

        # Record all the bands and the mask
        for i in range(7):
            list_of_lists[i].append(filtered_granule[i])

    # Now take the median of each color band
    fmask_filter = np.array(list_of_lists[-1])
    composited = []
    for i in range(6):
        marr = np.ma.array(list_of_lists[i])
        print('marr type:',type(marr))
        print('fmask_filter type:',type(fmask_filter))
        print('marr type:',marr.dtype)
        print('fmask_filter type:',fmask_filter.dtype)
        marr.mask = (marr==-9999) | fmask_filter
        this_band = np.ma.median(marr, axis=0).astype('int16')
        composited.append(this_band.filled(-9999))

    # Now composited contains the median-composited data for all bands
    # for just this period, and for just this tile.
    return composited


def get_period_schemes():

    first_pd = (1,90)
    last_pd = (301,366)
    meat_5day = [(i,i+4) for i in range(91,300,5)]
    meat_14day = [(i,i+13) for i in range(91,300,14)]

    scheme_5day = [first_pd] + meat_5day + [last_pd]
    scheme_14day = [first_pd] + meat_14day + [last_pd]
    
    return scheme_5day, scheme_14day


def composite_tile_year(tile,
                            year,
                            period_scheme,
                            hls_dir='../data/hls_23feb23',
                            sats=['L30','S30'],
                            mark_if_clouds=True,
                            mark_if_shadow=True,
                            mark_if_adjacent=True,
                            mark_if_snow=False):

    feature_names = []
    features = []

    for period in period_scheme:
        
        period_start = period[0]
        period_end = period[1]
        
        # NAMES
        name_ohne_band = f'Refl_{period_start}_{period_end}'
        bands = ['Blue','Green','Red','NIR','SWIR1','SWIR2']
        names_this_period = [f'{name_ohne_band}_{band}' for band in bands]
        feature_names += names_this_period
        
        # DATA
        composited_this_period = composite_one_period(tile=tile,
                            year=year,
                            period_start=period_start,
                            period_end=period_end,
                            hls_dir=hls_dir,
                            sats=sats,
                            mark_if_clouds=mark_if_clouds,
                            mark_if_shadow=mark_if_shadow,
                            mark_if_adjacent=mark_if_adjacent,
                            mark_if_snow=mark_if_snow)
        features += composited_this_period
        
        print(f'Finished period ending DOY {period_end}.')
        
        # features is a list of flat numpy arrays
        # feature_names is a list of strings
        
    return features, feature_names


def interpolate_cubic_spline(features):

    n_pixels = len(features[0])
    n_features = len(features)
    array_new = np.full(shape=(n_pixels,n_features),
                        fill_value=-9999,dtype='int16')

    for i_pixel in range(n_pixels): # for every pixel in the 14 million
        if i_pixel%1000000 == 0:
            print(f'Sit tight! Fitting cubic splines on the {i_pixel}th pixel...')
        for band in range(6): # for each band in the 6
            # gather the data for this pixel this band
            these_data = [features[j][i_pixel] for j in range(band, n_features, 6)]
            
            meat = these_data[1:-1]

            t = [t for t,refl in enumerate(meat) if refl!=-9999]
            r = [refl for refl in meat if refl!=-9999]

            f = CubicSpline(t,r)

            t_new = np.arange(len(meat))
            r_new = f(t_new).astype('int16')
            r_sandwich = np.concatenate(([these_data[0]],
                                        r_new,
                                        [these_data[-1]]),
                                        dtype='int16')

            # write the filled features into array_new in the same order
            for k,j in enumerate(range(band, n_features, 6)):
            # distribute each element of r_new out into array_new
                array_new[i_pixel,j] = r_sandwich[k]

    return array_new
