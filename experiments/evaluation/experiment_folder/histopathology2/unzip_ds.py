import zipfile
from glob import glob
import os
from tqdm import tqdm

# zip_paths = glob(os.path.join('/mnt/lustre-grete/usr/u12649/scratch/data/melanoma_dataset/extracted/complete_data', '*.zip'))

# for zip_path in tqdm(zip_paths):
#     dir_name, ext = os.path.splitext(os.path.basename(zip_path))
#     extract_to_path = os.path.join('/mnt/lustre-grete/usr/u12649/scratch/data/melanoma_dataset/extracted/', dir_name)
#     os.makedirs(extract_to_path, exist_ok=False)
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_to_path)
# print('Files extracted')

# path = '/mnt/lustre-grete/usr/u12649/scratch/data/nuinsseg/extracted/complete_dataset'
# for dir_name in tqdm(os.listdir(path)):
#     if dir_name in ['mouse', 'human']:
#         continue
#     dir_path = os.path.join(path, dir_name)
#     new_name = dir_name.strip('"').replace(" ", "_")
#     species = new_name.split("_")[0]
#     new_path = os.path.join(path, species, new_name)
#     os.rename(dir_path, new_path)
# human_path = '/mnt/lustre-grete/usr/u12649/scratch/data/nuinsseg/extracted/complete_dataset/human'
# mouse_path = '/mnt/lustre-grete/usr/u12649/scratch/data/nuinsseg/extracted/complete_dataset/mouse'
# for name in os.listdir(path):
#     if 'human' in name:
#         dir_path = os.path.join(path, name)
#         os.rename(dir_path, human_path)
#     else:
#         dir_path = os.path.join(path, name)
#         os.rename(dir_path, mouse_path)
# path = '/mnt/lustre-grete/usr/u12649/scratch/data/nuinsseg/extracted/complete_dataset'


# def clean_directory_names(base_directory):
#     # Loop through the directory structure recursively
#     for dirpath, dirnames, filenames in os.walk(base_directory, topdown=False):
#         for dirname in dirnames:
#             # Construct the full path to the directory
#             dir_path = os.path.join(dirpath, dirname)
            
#             # Remove quotation marks and replace spaces with underscores
#             new_name = dirname.strip('"').replace(" ", "_")
            
#             # Construct the new full path with the cleaned directory name
#             new_path = os.path.join(dirpath, new_name)
            
#             # Rename the directory if the name has changed
#             if dir_path != new_path:
#                 os.rename(dir_path, new_path)
#                 print(f"Renamed: {dir_path} -> {new_path}")



# clean_directory_names(path)
import geopandas as gpd

# Load the GeoJSON file into a GeoDataFrame
# geojson_files = '/mnt/lustre-grete/usr/u12649/scratch/data/melanoma_dataset/extracted/01_training_dataset_geojson_nuclei'
# for geojson_file in glob(os.path.join(geojson_files, '*.geojson')):
#     gdf = gpd.read_file(geojson_file)

# # Print the first few rows to inspect the data
#     #print(gdf.head())

#     # Extract the polygons (geometry) for annotation
#     # Assuming the geometry column contains the polygons for instance segmentation
#     polygons = gdf['geometry'].apply(lambda geom: geom.__geo_interface__['coordinates'])

#     # Show the polygons (which should be lists of coordinates)
#     print(polygons.head())
#     break

import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
from shapely.geometry import mapping
import torch
# Step 1: Load the GeoJSON file
folder1 = '/mnt/lustre-grete/usr/u12649/scratch/data/melanoma_dataset/extracted/01_training_dataset_geojson_nuclei/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_path = os.path.join('/mnt/lustre-grete/usr/u12649/scratch/data/melanoma_dataset/extracted/', 'labels_tiff')
os.makedirs(output_path, exist_ok=True)
for geojson_file in tqdm(glob(os.path.join(folder1, '*.geojson'))):
    gdf = gpd.read_file(geojson_file)

    # Step 2: Set up the rasterization parameters (size, resolution, etc.)
    # Define the resolution and size of the output raster (in pixels)
        # 0.23 μm per pixel (as per your microscopy data)

    # Calculate the bounds of the entire GeoJSON
    minx, miny, maxx, maxy = gdf.total_bounds
    pixel_size = (maxx - minx) / 1024
    width = 1024
    height = 1024

    # Step 3: Create a transformation for the raster (affine transform)
    transform = rasterio.transform.from_origin(minx, maxy, pixel_size, pixel_size)

    # Step 4: Rasterize the vector data (GeoJSON polygons) to raster format
    # Create an empty array to store the rasterized data
    raster = torch.zeros((height, width), dtype=torch.int32, device=device)

    # Rasterize the geometries (this will assign a value to each pixel inside the polygons)
    for idx, row in gdf.iterrows():
        # Geometry to be rasterized (polygon or multipolygon)
        geometry = row['geometry']
        # Rasterize the polygon geometry and add the result to the raster array
        mask = geometry_mask([mapping(geometry)], transform=transform, invert=True, out_shape=(height, width))
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
        instance_id = idx + 1
        raster[mask_tensor] = instance_id  # You can use different values for different features or classes
    raster_cpu = raster.cpu().numpy()
    raster_cpu_final = np.flipud(raster_cpu)
    # Step 5: Save the rasterized data as a TIFF file using rasterio
    tiff_name, _ = os.path.splitext(os.path.basename(geojson_file))
    tiff_file = os.path.join(output_path, f'{tiff_name}.tiff')
    with rasterio.open(
        tiff_file, 'w', driver='GTiff', height=height, width=width,
        count=1, dtype=rasterio.uint16, crs=gdf.crs,
        transform=transform
    ) as dst:
        dst.write(raster_cpu_final, 1)  # Write the raster data to the first band
    print(f"GeoJSON successfully converted to TIFF and saved to {tiff_file}")


# from osgeo import gdal, ogr

# # Input and output files
# input_geojson = "/mnt/lustre-grete/usr/u12649/scratch/data/melanoma_dataset/extracted/01_training_dataset_geojson_nuclei/training_set_metastatic_roi_063_nuclei.geojson"
# output_tiff = "output.tiff"

# # Open the GeoJSON
# source = ogr.Open(input_geojson)
# layer = source.GetLayer()

# # Define raster properties
# x_res = 1024  # pixels in x direction
# y_res = 1024  # pixels in y direction
# xmin, xmax, ymin, ymax = layer.GetExtent()
# pixel_size = (xmax - xmin) / x_res

# # Create raster dataset
# target_ds = gdal.GetDriverByName("GTiff").Create(output_tiff, x_res, y_res, 1, gdal.GDT_Byte)
# target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))

# # Set a no-data value
# band = target_ds.GetRasterBand(1)
# band.SetNoDataValue(0)

# # Rasterize using a specific attribute (e.g., 'id')
# attribute = "id"  # Change this to the attribute you want to use
# gdal.RasterizeLayer(target_ds, [1], layer, options=[f"ATTRIBUTE={attribute}"])

# # Close datasets
# target_ds = None
# source = None

# print(f"Rasterization completed! Output saved to {output_tiff}")
