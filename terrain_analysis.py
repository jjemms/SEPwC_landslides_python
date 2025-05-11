"""
terrain_analysis.py

This script uses a machine learning model to predict landslide risk based on topography, geology, land cover, and fault data.
The model trains a RandomForestClassifier on a dataset of known landslide locations.
Then the script uses the trained model on each pixel to create a landslide risk map (values between 0.0-1.0).

Plan:
1. Read inputs (DEM, geology, land cover, fault locations, landslide locations)
2. Calculate slope (from DEM) and distance to faults
3. Get training samples and create a DataFrame of positive and negative samples
   (positive = landslide, negative = no landslide)
4. Train a RandomForestClassifier on the training samples
5. Create a probability for each pixel using the classifier
6. Save results as a GeoTIFF file
"""
import argparse
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.io import MemoryFile, DatasetReader
from typing import List, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point

def convert_to_rasterio(
    raster_data: np.ndarray,
    template_raster: DatasetReader
) -> DatasetReader:
    """
    Wrap a 2D NumPy array as a rasterio DatasetReader
    using the template raster's spatial metadata
    """
    # Copy the geospatial metadata and update it to match the new array
    profile = template_raster.profile.copy()
    profile.update({
        'height': raster_data.shape[0],
        'width': raster_data.shape[1],
        'count': 1,                       # single band
        'dtype': raster_data.dtype,
        'transform': template_raster.transform,
    })

    # create an in-memory GeoTIFF file and write array into band 1
    memfile = MemoryFile()
    with memfile.open(**profile) as dst:
        dst.write(raster_data, 1)

    # re-open the in-memory file in read mode and return the DatasetReader
    return memfile.open()

def build_dataframe(
     dem: DatasetReader,
     geology: DatasetReader,
     landcover: DatasetReader,
     faults: DatasetReader,
     landslides_gdf: gpd.GeoDataFrame,
     n_samples: int = 1000,  # number of negative and positive samples
) -> gpd.GeoDataFrame:
    """
    Build a geodata frame of positive (landslide) and negative (no landslide) samples
    for training the landslide classifier
    """
    # Positive samples, randomly pick n_samples landslide points
    if len(landslides_gdf) > n_samples:
        pos_gdf = landslides_gdf.sample(n=n_samples, random_state=42)
    else:
        pos_gdf = landslides_gdf.sample(n=n_samples, replace=True, random_state=42)
    pos_pts = list(pos_gdf.geometry)

    #Create mask of 1 where landslides are, 0 elsewhere
    transform = dem.transform
    landslide_shapes = [(geom, 1) for geom in pos_pts]
    mask_arr = rasterio.features.rasterize(
        landslide_shapes,
        out_shape=(dem.height, dem.width),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    # Find indices of negative samples where mask == 0
    no_landslide_indices = np.column_stack(np.where(mask_arr ==0))
    # Randomly select n_samples negative samples
    if len(no_landslide_indices) > n_samples:
        chosen = no_landslide_indices[np.random.choice(len(no_landslide_indices), n_samples, replace=False)]
    else:
        chosen = no_landslide_indices[np.random.choice(len(no_landslide_indices), n_samples, replace=True)]
    # Convert to shapely points
    neg_pts = []
    for row, col in chosen:
        x, y = transform * (col + 0.5, row + 0.5)  # The center of pixel
        neg_pts.append(Point(x, y))

    # Sample all four rasters at a list of points
    def sample_all(pts: List[Point]):
        return {
            'elevation': extract_values_from_raster(dem, pts),
            'geology': extract_values_from_raster(geology, pts),
            'landcover': extract_values_from_raster(landcover, pts),
            'faults': extract_values_from_raster(faults, pts),
        }
    pos_data = sample_all(pos_pts)
    neg_data = sample_all(neg_pts)

    # Create DataFrame labelled
    df_pos = pd.GeoDataFrame(
        {**pos_data, 'landslide': [1]*n_samples}
        geometry=pos_pts,
        crs=landslides_gdf.crs
    )
    df_neg = pd.GeoDataFrame(
        {**neg_data, 'landslide': [0]*n_samples},
        geometry=neg_pts,
        crs=landslides_gdf.crs
    )
    # Combine positive and negative samples and return
    return pd.concat([df_pos, df_neg], ignore_index=True)


def extract_values_from_raster(
    raster: DatasetReader,
    points: List[Point]
) -> List[float]:
    """
    given an open rasterio DatasetReader and a list of shapely point
    geometries (in same CRS as raster), sample the raster at each point 
    and return a list of python floats
    """
    coordinates = [(pt.x, pt.y) for pt in points]
    samples = raster.sample(coordinates)

    # loop over each sample array, convert to float, collect in a list
    return [float(arr[0]) for arr in samples] 

def make_classifier(x, y, verbose=False):

    return

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):

    return

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):

    return


def main():


    parser = argparse.ArgumentParser(
                     prog="Landslide hazard using ML",
                     description="Calculate landslide hazards using simple ML",
                     epilog="Copyright 2024, Jon Hill"
                     )
    parser.add_argument('--topography',
                    required=True,
                    help="topographic raster file")
    parser.add_argument('--geology',
                    required=True,
                    help="geology raster file")
    parser.add_argument('--landcover',
                    required=True,
                    help="landcover raster file")
    parser.add_argument('--faults',
                    required=True,
                    help="fault location shapefile")
    parser.add_argument("landslides",
                    help="the landslide location shapefile")
    parser.add_argument("output",
                    help="the output raster file")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()


if __name__ == '__main__':
    main()
