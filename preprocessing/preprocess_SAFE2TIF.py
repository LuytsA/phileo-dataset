import os
from datetime import date
import glob

import buteo as beo
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from numba import jit, prange
from zipfile import ZipFile
import gc

from functools import partial
import concurrent.futures


def preprocess_SAFE(tile_path: str,
                             folder_dst: str) -> None:
    try:

        folder = tile_path.split('/')[-2]
        name = tile_path.split('/')[-1].split('.')[0]
        out_path=os.path.join(folder_dst, f"{folder}/{name}_s2.tif")

        if not os.path.exists(out_path):
            bands = beo.s2_l2a_get_bands(tile_path, zipfile=False)

            # order S2 bands: 0-SCL, 1-B02, 2-B03, 3-B04, 4-B08, 5-B05, 6-B06, 7-B07, 8-B8A, 9-B11, 10-B12
            scl = beo.raster_align(rasters=[
                bands['20m']['SCL'],
            ], resample_alg='bilinear', reference=bands['10m']['B02'])

            s2_resampled = beo.raster_align(rasters=[
                bands['10m']['B02'],
                bands['10m']['B03'],
                bands['10m']['B04'],
                bands['10m']['B08'],
                bands['20m']['B05'],
                bands['20m']['B06'],
                bands['20m']['B07'],
                bands['20m']['B8A'],
                bands['20m']['B11'],
                bands['20m']['B12'],
            ], resample_alg='bilinear', reference=bands['10m']['B02'])
            
            os.makedirs(os.path.join(folder_dst, f"{folder}"), exist_ok=True, mode=7777)

            output = beo.raster_stack_list(scl + s2_resampled, out_path=out_path,
                                            dtype='uint16'
                                            )
            
            beo.delete_dataset_if_in_memory_list(s2_resampled)
            beo.delete_dataset_if_in_memory_list(scl)
            beo.delete_dataset_if_in_memory(output)

            gc.collect()
        else:
            print(f"{out_path} already exists")
    except Exception as e:
        print(f"Tile {tile_path} failes with error")
        print(e)


def image_to_patch(path: str, folder_dst: str, overlaps: int = 1,
                   patch_size: int = 64):

    folder = path.split('/')[-2]
    name = path.split('/')[-1].split('.')[0]

    # throw away SCL layer
    # order S2 bands: 0-SCL, 1-B02, 2-B03, 3-B04, 4-B08, 5-B05, 6-B06, 7-B07, 8-B8A, 9-B11, 10-B12
    arr = beo.raster_to_array(path)[:, :, 1:]
    patches = beo.array_to_patches(
        arr,
        tile_size=patch_size,
        n_offsets=overlaps,
        border_check=True,
    )
    os.makedirs(os.path.join(folder_dst, f"{folder}"), exist_ok=True)
    np.save(os.path.join(folder_dst, f"{folder}/{name}.npy"), np.array(patches))



def preprocess_image_to_patches(
        folder_src: str,
        folder_dst: str,
        overlaps: int = 1,
        patch_size: int = 64,
        max_workers: int = 2,
) -> None:
    
    print('Generating Patches from .tif images')
    paths = glob.glob(os.path.join(folder_src, '*/*.tif'))
    proc = partial(image_to_patch, folder_dst=folder_dst, overlaps=overlaps, patch_size=patch_size)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(proc, paths), total=len(paths)))



def extract_zip(folder_src: str) -> None:
    zips = glob.glob(os.path.join(folder_src, '*/*.zip'))
    print('Unzipping S2 .SAFE files')
    for zip in tqdm(zips):
        folder = zip.split('/')[-2]
        # loading the temp.zip and creating a zip object
        with ZipFile(zip, 'r') as zObject:
            zObject.extractall(path=os.path.join(folder_src, f"{folder}"))
        os.remove(zip)





def main():
    folder_src='/phileo_data/foundation_data'
    folder_dst='/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/data_testing' #'/phileo_data/foundation_data/tif_files'
    max_workers = 1

    # unzip all files in folder
    extract_zip(folder_src=folder_src)
    
    # select .SAFE files to process to .tif
    paths = glob.glob(os.path.join(folder_src, '10_points_filtered_22_07/*.SAFE'))
    
    # process paths to .tif files where bands have been merged
    print('converting S2 Tile Granule to Image')
    proc = partial(preprocess_SAFE, folder_dst = folder_dst)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(proc, paths), total=len(paths)))
    
    # # process tif to npy patches without any checks
    # preprocess_image_to_patches(folder_src=folder_dst, folder_dst=folder_dst)
    


if __name__ == '__main__':
    main()