import numpy as np
import rasterio
import os
import cv2
from rasterio.mask import mask
from rasterio.plot import show, reshape_as_image
import geopandas as gpd

def generate_images(urls, to_folder, test_data=False):
    fpath_tiff = urls[0]
    fpath_geojson = urls[1]
    
    print("Reading in: {} tif image".format(fpath_tiff.split("/")[-2]))
    with rasterio.open(fpath_tiff) as tiff:
            df_roof_geometries = gpd.read_file(fpath_geojson)

            tiff_crs = tiff.crs.data
            df_roof_geometries['projected_geometry'] = (
                df_roof_geometries['geometry'].to_crs(tiff_crs)
            )

            roof_geometries = ""
            if not test_data:
                roof_geometries = (
                    df_roof_geometries[['id', 'roof_material', 'projected_geometry']].values
                )
            else:
                roof_geometries = (
                    df_roof_geometries[['id', 'projected_geometry']].values
                )
            numof_data = len(roof_geometries)
            for i, roof in enumerate(roof_geometries):
                print("Generating {} -> {} / {}".format(fpath_tiff.split("/")[-2], i+1, numof_data))
                
                if not test_data:
                    roof_id, roof_material, projected_geometry = roof
                    target_path = os.path.join(to_folder, roof_material, str(roof_id)+'-'+roof_material+'.png')
                else:
                    roof_id, projected_geometry = roof
                    target_path = os.path.join(to_folder, str(roof_id)+'.png')

                img = mask(tiff, [projected_geometry], crop=True)[0]
                cv2.imwrite(target_path, reshape_as_image(img))