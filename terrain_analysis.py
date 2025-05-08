"""
terrain_analysis.py

This script uses a machine learning model to predict landslide risk based on topography, geology, land cover, and fault data.
The model trains a RandomForestClassifier on a dataset of known landslide locations.
Then the script uses the trained model on each pixel to create a landslide risk map (values between 0.0-1.0).

Plan:
1. Read inputs (DEM, geology, land cover, fault locations, landslide locations)
2. Calculate slope (from DEM) and distance to faults
3. Gat training samples
4. Train a RandomForestClassifier on the training samples
5. Create a probability for each pixel using the classifier
6. Save results as a GeoTIFF file
"""
import argparse
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.io import MemoryFile, DatasetReader
from typing import List
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

def extract_values_from_raster(
    raster: DatasetReader,
    points: List[Point]
) -> List[float]:
    """
    given an open rasterio DatasetReader and a list of shapely point
    geometries, sample the raster at each point and return a list of floats
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
