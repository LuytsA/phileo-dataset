import numpy as np
import buteo as beo

def raster_clip_to_reference(
        global_raster:str,
        reference_raster:str,
    ):

    bbox_ltlng = beo.raster_to_metadata(reference_raster)['bbox_latlng']
    bbox_vector = beo.vector_from_bbox(bbox_ltlng, projection=global_raster)
    bbox_vector_buffered = beo.vector_buffer(bbox_vector, distance=0.1)
    global_clipped = beo.raster_clip(global_raster, bbox_vector_buffered, to_extent=True, adjust_bbox=False)

    return global_clipped



def encode_date(day_of_year):
    day_encoded = ((1+np.sin(2*np.pi*day_of_year/365))/2, (1+np.cos(2*np.pi*day_of_year/365))/2)
    return day_encoded

def decode_date(encoded_date):
    doy_sin,doy_cos = encoded_date
    doy = np.arctan2((2*doy_sin-1),(2*doy_cos-1))*365/(2*np.pi)
    if doy<1:
        doy+=365
    return np.array([np.round(doy)])


def decode_coordinates(encoded_coords):
    lat_enc,long_sin,long_cos = encoded_coords
    lat = -lat_enc*180+90
    long = np.arctan2((2*long_sin-1),(2*long_cos-1))*360/(2*np.pi)
    return np.array([lat,long])

def encode_coordinates(coords):
    lat,long = coords
    lat = (-lat + 90)/180
    long_sin = (np.sin(long*2*np.pi/360)+1)/2
    long_cos = (np.cos(long*2*np.pi/360)+1)/2

    return np.array([lat,long_sin,long_cos], dtype=np.float32)
