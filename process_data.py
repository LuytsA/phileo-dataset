import glob
import os 
import json 
from tqdm import tqdm
from functools import partial
import concurrent.futures

from preprocessing.preprocess_raster2labels import create_raster_labels
from preprocessing.preprocess_SAFE2TIF import preprocess_SAFE
from preprocessing.preprocess_TIF2NPY import preprocess_tifs

def main(task, dataset, dst_folder, max_workers=1):

    assert task in ['SAFE2TIF','TIF2NPY', 'raster2labels']
    assert dataset in ['downstream','mini_foundation/22_01','mini_foundation/22_04', 'mini_foundation/22_07', 'mini_foundation/22_10']

    dataset, month = dataset.split('/') if dataset.startswith('mini_foundation') else [dataset, None]



    if task == 'SAFE2TIF':
        assert dataset =='mini_foundation', f"SAFE files only available for dataset mini_foundation"

        folder_src='/phileo_data/mini_foundation/mini_foundation_SAFE'
        folder = f'10_points_filtered_{month}'
        folder_dst=dst_folder #'/phileo_data/foundation_data/tif_files'

        # select .SAFE files to process to .tif
        paths = glob.glob(os.path.join(folder_src, f'{folder}/*.SAFE'))

        # process paths to .tif files where bands have been merged
        print('converting S2 Tile Granule to Image')
        proc = partial(preprocess_SAFE, folder_dst = folder_dst)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(proc, paths), total=len(paths)))



    elif task == 'TIF2NPY':
        src_folder = '/phileo_data/downstream/downstream_dataset_tifs' 
        dst_folder = dst_folder # /phileo_data/downstream/downstream_dataset_patches_np
            
        N_OFFSETS = 0
        PATCH_SIZE = 128
        VAL_SPLIT_RATIO = 0.1

        with open(f'utils/train_test_location_downstream.json', 'r') as f:
            train_test_locations = json.load(f)


        train_locations =  train_test_locations['train_locations'] 
        test_locations = train_test_locations['test_locations']

        if dataset == 'mini_foundation':
            folder = f'10_points_filtered_{month}'
            src_folder = f'/phileo_data/mini_foundation/mini_foundation_tifs/{folder}'
            files = glob.glob(f"/phileo_data/mini_foundation/mini_foundation_SAFE/{folder}/**.SAFE")
            train_locations = [f.split('/')[-1].split('.SAFE')[0] for f in files]
            test_locations = []


        preprocess_tifs(
            folder_src=src_folder,
            folder_dst=dst_folder,
            dataset=dataset,
            overlaps=N_OFFSETS,
            patch_size=PATCH_SIZE,
            val_split_ratio=VAL_SPLIT_RATIO,
            test_locations=test_locations,
            train_locations=train_locations,
            create_geo_label=False,
            num_workers=max_workers
        )



    elif task == 'raster2labels':
    
        # create_raster_labels
        src_folder = '/phileo_data/downstream/downstream_dataset_tifs/'
        dst_folder = dst_folder

        koppen_geiger_map = '/phileo_data/aux_data/beck_kg_map_masked.tif'
        encoded_coords_map =  '/phileo_data/aux_data/encoded_coordinates_global.tif'
        dem_map = '/phileo_data/aux_data/copdem30.vrt'
            

        labels = ['label_encoded_coordinates', 'label_kg', 'label_terrain']


            
        tile_paths = glob.glob(f"{src_folder}/*_0.tif")
        if dataset=='mini_foundation':
            folder = f'10_points_filtered_{month}'
            src_folder = f'/phileo_data/mini_foundation/mini_foundation_tifs/{folder}'
            tile_paths = glob.glob(f"{src_folder}/*_s2.tif")


        create_raster_labels(
            folder_src=src_folder,
            folder_dst=dst_folder,
            tile_paths=tile_paths,
            labels=labels,
            kg_map=koppen_geiger_map,
            encoded_coords_map = encoded_coords_map,
            dem_map = dem_map,
            dataset=dataset,
            num_workers=max_workers
            )





if __name__ == '__main__':

    '''
    There are 3 main preprocessing scripts
    1)  preprocess_SAFE2TIF: takes the level 2A S2 files (.SAFE), merges the bands and saves the results as a .tif file.
        In the datafolder this is the step mini_foundation_SAFE --> mini_foundation_tifs

    2)  preprocess TIF2NPY: takes the raster files (.tif) and creates numpy arrays of shape (n_patches, tile_size, tile_size, channels)
        In the process cloudy patches will be discarded. If raster labels are available, numpy arrays of the corresponding labels are created as well.
        if dataset==mini_foundation, this is the step mini_foundation_tifs --> mini_foundation_patches_np
        if dataset==downstream, this is the step downstream_dataset_tifs --> downstream_dataset_patches_np

    3)  preprocess_raster2labels: create raster labels for terrain, encoded coordinates or Koppen-Geiger climate zones.
        labels for images are created by cropping from a global raster. Global rasters are found under aux_data
        Terrain: aux_data/copdem30.vrt
        Koppen-Geiger: aux_data/beck_kg_map_masked.tif
        encoded coordinates: encoded_coordinates_global.tif
        In the datafolder this is the step aux_data --> **label_kg.tif, **label_coords.tif, **label_terrain.tif

    '''
    
    task = 'raster2labels'       # task in ['SAFE2TIF','TIF2NPY', 'raster2labels']
    dataset = 'mini_foundation'  # dataset in ['downstream','mini_foundation/22_01',mini_foundation/22_04, 'mini_foundation/22_07', mini_foundation/22_10]



    dst_folder = 'sample'
    max_workers = 2

    main(task=task, dataset=dataset, dst_folder=dst_folder, max_workers=max_workers)