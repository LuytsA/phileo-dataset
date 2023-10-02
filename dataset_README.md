# General 
This is the description for the PhiLeo  dataset. It's a global dataset of Sentinel-2 images and labels for roads, buildings, landcover, coordinates, terrain and climate zones. This dataset is used to train a foundation model for Sentinel-2 images. It's divided in two main subsets "downstream" and "mini_foundation". As the name suggests the "downstream" data is used to test the trained models on downstream tasks (landcover classification, road segmentation and building detection). The "mini_foundation" is a small sample of the full dataset the foundation model is trained on.





# Downstream

## S2 images
The S2 images in the downstream dataset are sampled from different places around the globe. The different regions are:
denmark-1, denmark-2, east-africa, egypt-1, eq-guinea, europe, ghana-1, isreal-1, isreal-2, japan, nigeria, north-america, senegal, south-america, tanzania-1, tanzania-2, tanzania-3, tanzania-4, tanzania-5 and uganda-1. Each region consists up to 200 S2 tiles of different sizes. Some locations have been revisited up to 3 times, creating a timeseries. The naming convention of for an S2 tile is the following: 

    s2_name_downstream = <region>_<tile_nr>_<timeseries_id>  

The S2 images contain 11 bands in the following order: 0-SCL, 1-B02, 2-B03, 3-B04, 4-B08, 5-B05, 6-B06, 7-B07, 8-B8A, 9-B11, 10-B12
where SCL is the Scene Classification Layer. More info on S2 bands see https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a

## Labels downstream
The naming convention of a label files corresponding to an S2 tile is the following: 
    
    label_name_downstream = <region>_<tile_nr>_label_<label_name> 

There is no timeseries_id since the label is valid at all times.

**ROADS**: 1 channel, float <br>
The labels are expressed of number of squared meters of roads on a given pixel. Values are between 0-100 $m^2$ and for a resolution of 10m this reflect the percentage of coverage. This data is a combination of the Google Open Buildings dataset, OSM buildings and manual labeling.


**BUILDINGS**: 1 channel, float <br>
The labels are expressed of number of squared meters of building on a given pixel. Values are between 0-100 $m^2$ and for a resolution of 10m this reflect the percentage of coverage. This data is a combination of the Google Open Buildings dataset, OSM buildings and manual labeling.


**LANDCOVER (lc)**: 1 channel, int <br>
Landcover labels taken from ESA worldcover. More info about classes at https://worldcover2020.esa.int/data/docs/WorldCover_PUM_V1.1.pdf

**CLIMATE ZONES (kg)**: 1 channel, int <br>
Koppen-Geiger climate zone labels. Hierarchical labels on climate zones around the world at a resolution of 1 km. Based on  Beck et al. but with additional water masks to classify inland rivers and lakes as water instead of the surrounding climate zone. More info on the labels at https://koeppen-geiger.vu-wien.ac.at/present.htm 

*Beck, H.E., N.E. Zimmermann, T.R. McVicar, N. Vergopolan, A. Berg, E.F. Wood: Present and future Köppen-Geiger climate classification maps at 1-km resolution, Nature Scientific Data, 2018.*

**COORDINATES (coords)**: 3 channels, float <br>
 Encoded latitute and longitude coordinates so all values lie between 0-1 and longitude is cyclical. Latitude and longitude (in degrees) are encoded as follows: 

    lat,lng --> (90-lat)/180, lng_sin, lng_cos
    NOTE: (90-lat)/180 is actually an accident, I wanted to do (lat+90)/180

where

    lng_sin = (sin(2 * pi * lng/180) + 1 ) / 2
    lng_cos = (cos(2 * pi * lng/180) + 1) / 2

**TERRAIN**: 4 channels, float <br>
 Encoded terrain data based on the Copernicus Global Digital Elevation Model (GLO-30) which offers global coverage at a resolution of 30 metres. More information about the Copernicus DEM at https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model. The Copernicus DEM gives elevation in meters above sea-level.  This information is encoded (values between 0-1) in 4 channels: 
    
    elevation --> aspect, slope, elevation --> aspect_sin, aspect_cos, slope / 90, elevation / 8849.0

 where 8849.0m is the height of Mt. Everest and,
 
    aspect_sin = (sin(2 * pi * aspect/360) + 1) / 2
    aspect_cos = (cos(2 * pi * aspect/360) + 1) / 2

NOTE: road labels are not availble for all regions, only for east-africa, eq-guinea, europe, japan, nigeria, north-america, senegal and south-america.




# Mini foundation


## S2 images
The S2 images in the downstream dataset are sampled from different biomes over the world. All images were taken during the course of 1 year. Each location has been sampled 4 times during the year at different months, in 01-2022, 04-2022, 07-2022 and 10-2022. Naming convention for the S2 tiles is the standard naming convention:
    
    s2_name_foundation = <Mission ID>_MSIL2A_<datatake time>_<Processing Number>_<Relative Orbit>_<Tile Number field>_<Product Discriminator>

The S2 images contain 11 bands in the following order: 0-SCL, 1-B02, 2-B03, 3-B04, 4-B08, 5-B05, 6-B06, 7-B07, 8-B8A, 9-B11, 10-B12
where SCL is the Scene Classification Layer. More info on S2 bands see https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a
## Labels mini_foundation
The naming convention of a label files corresponding to an S2 tile is the following: 
    
    label_name_foundation = <s2_name>_label_<label_name>

**CLIMATE ZONES (kg)**: <br>
 see labels downstream

**COORDINATES (coords)**: <br>
 see labels downstream

**DAY OF THE YEAR (time)**: 2 channels, float <br>
 Encoded day of the year so that values lie between 0-1. Day (between 1-365) are encoded as follows: 

    day  --> day_sin, day_cos
    
where 
    
    day_sin = (sin(2 * pi * day/365) + 1) / 2
    day_cos = (cos(2 * pi * day/365) + 1) / 2

**TERRAIN**:  <br>
see labels downstream

# Formats

The data is available in 3 available formats
- **.SAFE**: Sentinel data format containing all bands seperately (https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/data-formats). Only available for mini foundation s2 data
- **.tif**: raster data for both s2 images and labels 
- **.npy**: NumPy arrays obtained for labels and images after processing tif-files. NumPy arrays have a shape of (number_of_patches, patch_size, patch_size, number_of_channels). For an s2 images this shape could be for example $(8000,128,128,10)$


# Folder structure

``` 
├── downstream
│   │
│   ├── downstream_dataset_tifs
│   │   ├── <region>_<tile_nr>_label_<label_name>.tif
│   │   ├── <region>_<tile_nr>_<timeseries_id>.tif
│   │   └── ...
│   │
│   └── downstream_dataset_patches_np
│       ├── <region>_<tile_nr>_label_<label_name>.npy
│       ├── <region>_<tile_nr>_<timeseries_id>.npy
│       ├── ...
│       └── metadata
│
├── mini_foundation
│   │  
│   ├── mini_foundation_SAFE
│   │   
│   ├── mini_foundation_tifs
│   │   ├── 10_points_filtered_22_01
│   │   │   ├── <s2_name_foundation>_label_<label_name>.tif
│   │   │   ├── <s2_name_foundation>_s2.tif
│   │   │   └── ...
│   │   ├── 10_points_filtered_22_04
│   │   ├── 10_points_filtered_22_07
│   │   └── 10_points_filtered_22_10
│   │   
│   └── mini_foundation_patches_np
│       ├── 10_points_filtered_22_01
│       │   ├── <s2_name_foundation>_label_<label_name>.npy
│       │   ├── <s2_name_foundation>_s2.npy
│       │   └── ...
│       ├── 10_points_filtered_22_04
│       ├── 10_points_filtered_22_07
│       └── 10_points_filtered_22_10
│       
└── aux_data
        ├── beck_kg_map_masked.tif
        ├── encoded_coordinates_global.tif
        ├── terrain
        └── train_test_location_all.json
```

metadata:  holds information on each tile during processing from tifs to npy. Info on the labels associated to a s2 image, label sizes, train/test split, max cloud cover for processing etc.

aux_data: auxiliary data used to create tif labels for each tile.

train_test_location_all.json: stores which tiles belong to the train set and which to the test set.
