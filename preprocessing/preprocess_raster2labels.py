import os
import sys; sys.path.append("./")
import numpy as np

from glob import glob
from tqdm import tqdm
import glob 
import buteo as bo
from datetime import datetime
from functools import partial
import concurrent.futures
import re

from utils import encode_date, raster_clip_to_reference

DATASETS = ['mini_foundation', 'downstream']

def create_label_terrain(tile_path:str, 
                    dem_map:str,
                    folder_dst:str,
                    folder_src:str,
                    overwrite:bool=False,
                    dataset:str='downstream'):

    if dataset=='mini_foundation':
        tile = tile_path.split('_s2.tif')[0].split('/')[-1]
        reference_raster = tile_path
    
    elif dataset=='downstream':
        file_name = tile_path.split('/')[-1]
        tile = re.split('_[0-9]+.tif',file_name)[0]
        reference_raster = tile_path 

    try:
        if os.path.exists(reference_raster):
            output_path = f'{folder_dst}/{tile}_label_terrain.tif'
            
            if not os.path.exists(output_path) or overwrite:
                terrain_clipped = raster_clip_to_reference(dem_map, reference_raster=reference_raster)
                terrain_encoded = bo.raster_dem_to_orientation(terrain_clipped, output_raster=output_path,  height_normalisation=True, include_height=True,)
                bo.raster_align(terrain_encoded, out_path=output_path ,reference=reference_raster,method='reference', resample_alg='bilinear', creation_options=["COMPRESS=DEFLATE", "PREDICTOR=3"])
            
            else:
                print(f"{output_path} already exists")
        else:
            print(f"WARNING: {reference_raster} not found")
    except Exception as e:
        print(f'WARNING: tile {tile} failed with error: \n',e)


def create_label_from_global_map(tile_path:str, 
                    global_map:str,
                    folder_dst:str,
                    resample:str = 'nearest',
                    overwrite:bool=False,
                    dataset:str='downstream'):

    if dataset=='mini_foundation':
        tile = tile_path.split('_s2.tif')[0].split('/')[-1]
        reference_raster = tile_path 
    
    elif dataset=='downstream':
        file_name = tile_path.split('/')[-1]
        tile = re.split('_[0-9]+.tif',file_name)[0]
        reference_raster = tile_path 

    try:
        if os.path.exists(reference_raster):
            output_path = f'{folder_dst}/{tile}_label_kg.tif'
            
            if not os.path.exists(output_path) or overwrite:
                global_clipped = raster_clip_to_reference(global_map, reference_raster=reference_raster)
                bo.raster_align(global_clipped, out_path=output_path ,reference=reference_raster,method='reference', resample_alg=resample)
            
            else:
                print(f"{output_path} already exists")
        else:
            print(f"WARNING: {reference_raster} not found")
    
    except Exception as e:
        print(f'WARNING: tile {tile} failed with error: \n',e)


def create_label_encoded_coordinates(tile_path:str, 
                    encoded_coords_map:str,
                    folder_dst:str,
                    folder_src:str,
                    overwrite:bool=True,
                    dataset:str='downstream'):
    
    assert dataset in ['downstream','mini_foundation'], f"dataset must be either 'downstream' or 'mini_foundation'"
    
    if dataset=='mini_foundation':
        tile = tile_path.split('_s2.tif')[0].split('/')[-1]
        reference_raster = tile_path 
        (_, _, datatake_time, _, _, tile_number, _) = tile.split('_')
        datetime_tile = datetime.strptime(datatake_time, "%Y%m%dT%H%M%S")
        first_day = datetime(datetime_tile.year,month=1,day=1)
        day_of_year = (datetime_tile-first_day).days
        day_encoded = encode_date(day_of_year) #((1+np.sin(2*np.pi*day_of_year/365))/2, (1+np.cos(2*np.pi*day_of_year/365))/2)

    elif dataset=='downstream':
        file_name = tile_path.split('/')[-1]
        tile = re.split('_[0-9]+.tif',file_name)[0]
        reference_raster = tile_path
    
    output_path = f'{folder_dst}/{tile}_label_coords.tif'


    try:
        if os.path.exists(reference_raster):
            if not os.path.exists(output_path) or overwrite:
                raster_clipped = raster_clip_to_reference(encoded_coords_map, reference_raster=reference_raster)
                
                if dataset=='mini_foundation':
                    coord_ar = bo.raster_to_array(raster_clipped)
                    day_encoded_ar = np.resize(day_encoded,(coord_ar.shape[0],coord_ar.shape[1],2))
                    time_raster = bo.array_to_raster(day_encoded_ar,reference=raster_clipped)
                    bo.raster_align(time_raster, out_path=output_path.replace('_label_coords','_label_time') ,reference=reference_raster,method='reference', creation_options=["COMPRESS=DEFLATE", "PREDICTOR=3"])

                bo.raster_align(raster_clipped, out_path=output_path ,reference=reference_raster,method='reference', resample_alg='bilinear', creation_options=["COMPRESS=DEFLATE", "PREDICTOR=3"])
            
            else:
                print(f"{output_path} already exists")
        else:
            print(f"WARNING: {reference_raster} not found")

    except Exception as e:
        print(f'WARNING: tile {tile} failed with error: \n',e)




def create_raster_labels(folder_dst:str,
                  folder_src:str,
                  tile_paths:list,
                  labels: list = ['label_encoded_coordinates','label_kg', 'label_terrain'],
                  kg_map = None,
                  encoded_coords_map = None,
                  dem_map = None,
                  num_workers = None,
                  dataset:str='downstream'
    ):
    '''
    For each s2-tile create a tif raster with label.
    label_encoded_coordinates: scaled latitute and encoded longitude (sine and cosine). For each tile the corresponding raster is created from the global encoded_coords_map
    label_kg: Koppen-Geiger climate zones. For each tile the corresponding raster is created from the global kg_map
    label_terrain: Copdem30 

    '''
    assert dataset in DATASETS, f"dataset name must be in {DATASETS}"

    for label in labels:
        if label=='label_encoded_coordinates':
            print(f'creating {label}')
            proc = partial(create_label_encoded_coordinates, encoded_coords_map = encoded_coords_map, folder_dst = folder_dst, folder_src = folder_src, dataset=dataset)


        elif label=='label_kg':
            print(f'creating {label}')
            proc = partial(create_label_from_global_map, folder_dst = folder_dst, folder_src = folder_src, resample='nearest', global_map=kg_map, dataset=dataset)
        

        elif label=='label_terrain':
            print(f'creating {label}')
            proc = partial(create_label_terrain, folder_dst = folder_dst, folder_src = folder_src, global_map=dem_map, dataset=dataset)
        
        
        else:
            raise ValueError("Only possible labels to be created are [label_encoded_coordinates, label_kg, label_terrain]")
        
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(proc, tile_paths), total=len(tile_paths)))


def main():
    src_folder = '/phileo_data/mini_foundation/mini_foundation_tifs/10_points_filtered_22_10'
    dst_folder = src_folder
    
    koppen_geiger_map = '/phileo_data/aux_data/beck_kg_map_masked.tif'
    encoded_coords_map =  '/phileo_data/aux_data/encoded_coordinates_global.tif'
    dem_map = '/phileo_data/aux_data/copdem30.vrt'
        

    labels = ['label_encoded_coordinates'] # 'label_kg', 'label_terrain'
    dataset = 'mini_foundation' # either "downstream" or "mini_foundation"
     
    tile_paths = glob.glob(f"{src_folder}/*_0.tif") if dataset=='downstream' else glob.glob(f"{src_folder}/*_s2.tif")

    create_raster_labels(
        folder_src=src_folder,
        folder_dst=dst_folder,
        tile_paths=tile_paths,
        labels=labels,
        kg_map=koppen_geiger_map,
        encoded_coords_map = encoded_coords_map,
        dem_map = dem_map,
        dataset=dataset,
        num_workers=4
        )
        
if __name__ == '__main__':
    main()
