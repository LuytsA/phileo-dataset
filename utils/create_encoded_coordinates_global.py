import buteo as beo
import numpy as np

def get_coordinate_array(coord_bbox, shape):
    '''
    coord_bbox in format [lat_min,lat_max,lng_min,lng_max]
    shape =(n_pixels in latitude, n_pixels in longitude)
    '''
    stepsize_lat=(coord_bbox[1]-coord_bbox[0])/shape[0]
    stepsize_lng=(coord_bbox[3]-coord_bbox[2])/shape[1]
    start_lat = coord_bbox[0]+stepsize_lat/2
    stop_lat = coord_bbox[1]+stepsize_lat/2
    start_lng = coord_bbox[2]+stepsize_lng/2
    stop_lng = coord_bbox[3]+stepsize_lng/2

    xy = np.mgrid[start_lat:stop_lat:stepsize_lat,start_lng:stop_lng:stepsize_lng]

    return xy



if __name__ == '__main__':
    
    '''
    Create global raster of encoded coordinates at same resolution as Koppen-Geiger map
    '''

    out_path='encoded_coords.tif'

    bbox =  [-90.0, 90.0, -180.0, 180.0]
    koppen_geiger_map = '/phileo_data/aux_data/beck_kg_map_masked.tif'

    kg_arr = beo.raster_to_array(koppen_geiger_map)
    md = beo.raster_to_metadata(koppen_geiger_map)
    co_ar = get_coordinate_array(bbox, kg_arr.shape)

    encoded_ar = np.zeros((co_ar.shape[1],co_ar.shape[2],3),dtype=np.float32)
    encoded_ar[:,:,0] = (co_ar[0,:,:]+90)/180
    encoded_ar[:,:,1] = (np.sin(co_ar[1,0,:]*2*np.pi/360)+1)/2
    encoded_ar[:,:,2] = (np.cos(co_ar[1,0,:]*2*np.pi/360)+1)/2


    beo.array_to_raster(encoded_ar,reference=koppen_geiger_map,out_path=out_path, creation_options=["COMPRESS=DEFLATE", "PREDICTOR=3"])