# USACE-KRI Sub-Project 4 Crop Mapping

Last edited by August Posch, 13Aug2024.

This Readme provides information about DATASETS and REPRODUCING THE STUDY. This entire repository contains open-source code produced for USACE-sponsored deliverables.

## DATASETS

All data used are publicly available. We accessed the publicly available data, processed these data for machine learning, performed machine learning, and analyzed the results. In this section are step-by-step instructions to reproduce the datasets.

### Access Harmonized Landsat-Sentinel (HLS) data
1. Create an account on NASA Earthdata website: https://urs.earthdata.nasa.gov
2. Establish a netrc file with your login credentials as explained in /hls-bulk-download/README.md
2. Run the Bash script hls-bulk-download/getHLS.sh through your shell from the hls-bulk-download directory, like this: "./getHLS.sh tmp.tileid.txt 2018-01-01 2022-12-31 ../data/hls_23feb23". Note that /hls-bulk-download/tmp.tileid.txt must contain the text "15TVG 10SFH" to specify the correct geographical tiles. Running the Bash script as indicated downloads HLS surface reflectance data for MGRS tiles 10SFH and 15TVG, for years 2018-2022, and saves these in /data/hls_23feb23.
3. In case you have unexpected difficulties on your system, check NASA's instructions within /hls-bulk-download/README.md and on GitHub: https://github.com/nasa/HLS-Data-Resources/tree/main/bash/hls-bulk-download

### Access Cropland Data Layer (CDL)
1. Go to this USDA webpage: https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php
2. Click the download link for each year (2014-2022) for National CDL. Unzip the downloads. Save them under /data. Name the folders NationalCDL_2014, ..., NationalCDL_2022. Each of these should contain a .tif file for the corresponding year, called 2014_30m_cdls.tif, ..., 2022_30m_cdls.tif. For example, there exists /data/NationalCDL_2014/2014_30m_cdls.tif, and likewise for every year.

### Process HLS data
1. Run /notebooks/eda-3B-composite-plots-as-for-loop.ipynb for each tile and period scheme. You need to set the tile and scheme parameters in the cell that says "Set Parameters Here". The output files are .npy files like /data/composited-interpolated/Refl_10SFH_2022_14day.npy, indicating Reflectance data, tile 10SFH, year 2022, and scheme 14day.

### Process CDL data
1. Run /notebooks/eda-0B-create-crop-data.ipynb. The results look like /data/processed_crop/Crop_10SFH_2014.npy, indicating Crop data, tile 10SFH, and year 2022.

### Create machine learning datasets
1. Run /notebooks/ml-1J-create-premade-train-val.ipynb and /notebooks/ml-4C-create-premade-train-val.ipynb. These combine CDL data with HLS data, and premake a set of files for the desired proportion of the dataset. Note that in our study, we used 0.1% training and 0.1% validation on the first pass, and these datasets are produced by /notebooks/ml-1J-create-premade-train-val.ipynb; then, the next part of our study used 100% training and 0.1% validation on the second pass, and these datasets are produced by /notebooks/ml-4C-create-premade-train-val.ipynb. The prepared data goes in a folder with a corresponding name; for example, the 0.1% training and 0.1% validation data goes in /data/premade_0.001_0.001, while the 100% training and 0.1% validation data goes in /data/premade_1.0_0.001. The files within will look like 10SFH_2018_14day_75_160_X_train.npy, indicating tile 10SFH, year 2018, scheme 14day, crop code 75, in-season cutoff day 160, and machine learning subset X_train.

### All data has been created.

## REPRODUCING THE STIDY
The deliverable of the study was ensemble machine learning methods and software for remote sensing based in-season mapping of crop type distribution. To produce this we accessed publicly available data, processed these data for machine learning, performed machine learning, and analyzed the results. In this section are step-by-step instructions to reproduce our work.

1. Access HLS data using /hls-bulk-download/getHLS.sh. (Detail above)
2. Access CDL data from the USDA website. (Detail above)
3. Process HLS data using /notebooks/eda-3B-composite-plots-as-for-loop.ipynb or similar. (Detail above)
4. Process CDL data using /notebooks/eda-0B-create-crop-data.ipynb. (Detail above)
5. Combine HLS and CDL into machine learning ready datasets using /notebooks/ml-1J-create-premade-train-val.ipynb and /notebooks/ml-4C-create-premade-train-val.ipynb. (Detail above)
6. Perform machine learning first pass. Run notebooks/ml-1L-parallel-and-premade. For running the models, use only the bottom of the two code cells You must specify where labeled "## SPECIFY HERE" for architecture and model. To reproduce our model specifics, please refer to the definition of return_model_object() function from notebooks/ml-4A-use-all-data.ipynb.
Outputs look like /data/results/ET041_0.001_0.001_15TVG_14day_1_230.csv indicating model ET041, 0.1% training, 0.1% validation, tile 15TVG, scheme 14day, crop code 1, in-season cutoff day 230. Note that based on your computing environment you may have to configure your Dask cluster differently (set of code cells under "Dask initialization") to most-efficiently run all the models in parallel.
7. Perform machine learning second pass. Run notebooks/ml-4A-use-all-data.ipynb. In the "Run models" code cell, you must specify model_names (a list of model codenames like 'LR036'), training_sample_size (like 1.0), and validation_sample_size (like 0.001). Results look like /data/results/LR036_1.0_0.001_10SFH_14day_75_160.csv, indicating model LR036, 100% training, 0.1% validation, tile 15TVG, scheme 14day, crop code 1, in-season cutoff day 230.
8. Analyze results using /notebooks/ml-3-analyze-results.ipynb. This organizes the results into a handy dataframe and lets you create useful visuals reporting best models.

### The above steps reproduce the deliverable: ensemble machine learning methods and software for remote sensing based in-season mapping of crop type distribution.

Optional activities explored during development use other notebooks in /notebooks, including: find a trusted pixel of a certain crop, perform pixel-level exploratory data analysis such as common land covers, create missing-data plots during interpolation, non-dask and alternative-dask methods of parallelization, and datasets that use no crop rotation history (NCRH) or only crop rotation history (0-day cutoff). For details about these side-investigations, please see the monthly reports.

