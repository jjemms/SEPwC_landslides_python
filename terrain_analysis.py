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
from rasterio.features import rasterize
from typing import List, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
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
    n_samples: int = 1000
) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame of positive (landslide) and negative (no landslide) samples
    for training the landslide classifier.
    """
    np.random.seed(42)  # reproducible sampling

    # make sure we have point geometries, if not use centroids
    if not all(landslides_gdf.geometry.geom_type == 'Point'):
        landslides_gdf = landslides_gdf.copy()
        landslides_gdf.geometry = landslides_gdf.geometry.centroid

    # Positive samples, randomly pick n_samples landslide points
    if len(landslides_gdf) >= n_samples:
        pos_pts = list(landslides_gdf.sample(n=n_samples, random_state=42).geometry)
    else:
        pos_pts = list(landslides_gdf.sample(n=n_samples, replace=True, random_state=42).geometry)

    #Create mask of 1 where landslides are, 0 elsewhere
    landslide_shapes = [(pt, 1) for pt in pos_pts]
    mask = rasterize(
        landslide_shapes,
        out_shape=(dem.height, dem.width),
        transform=dem.transform,
        fill=0,
        dtype='uint8'
    )

    # Find background pixels for negative samples
    rows, cols = np.where(mask == 0)
    if rows.size == 0:
        raise ValueError("No negative samples found in the raster mask.")
    indices = list(zip(rows, cols))
    replace = len(indices) < n_samples
    choice = np.random.choice(len(indices), size=n_samples, replace=replace)
    neg_pts = []
    for idx in choice:
        r, c = indices[idx]
        x, y = dem.transform * (c + 0.5, r + 0.5)  # pixel center
        neg_pts.append(Point(x, y))

    # Extract values from rasters
    def sample_all(pts):
        return {
            'elevation': extract_values_from_raster(dem, pts),
            'geology':   extract_values_from_raster(geology, pts),
            'landcover': extract_values_from_raster(landcover, pts),
            'faults':    extract_values_from_raster(faults, pts)
        }

    pos_data = sample_all(pos_pts)
    neg_data = sample_all(neg_pts)

    # create the GeoDataFrames with unified CRS
    crs = landslides_gdf.crs
    df_pos = gpd.GeoDataFrame(
        {**pos_data, 'landslide': [1] * n_samples},
        geometry=pos_pts,
        crs=crs
    )
    df_neg = gpd.GeoDataFrame(
        {**neg_data, 'landslide': [0] * n_samples},
        geometry=neg_pts,
        crs=crs
    )

    # combine and return
    combined = pd.concat([df_pos, df_neg], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry='geometry', crs=crs)

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
    """
    Train a RandomForestClassifier on x, y
    Splits of 30% for testing, scales features, fits 100 trees

    Test on 30% to see how model will perform in the real world
    
    if verbose=True, print the accuracy of train and test
    """
    # Split into train & test 
    # Train on 70% of data so model can learn patterns
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Fit forest of 100 trees
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_s, y_train)

    if verbose:
        train_acc = clf.score(X_train_s, y_train)
        test_acc  = clf.score(X_test_s,  y_test)
        print(f"Train accuracy: {train_acc:.2f}, Test accuracy: {test_acc:.2f}")

    clf.scaler = scaler
    return clf

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    return

def create_dataframe(
    topo: DatasetReader,
    geo: DatasetReader,
    lc: DatasetReader,
    dist_fault: DatasetReader,
    slope: DatasetReader,
    shapes: List[Point],
    label: int
) -> gpd.GeoDataFrame:
    """
    Sample the raster data at the given shapes and create a GeoDataFrame
    with columns and no geometry column.
    """
    # Sample each raster band at points
    elevs = extract_values_from_raster(topo, shapes)
    faults = extract_values_from_raster(dist_fault, shapes)
    slopes = extract_values_from_raster(slope, shapes)
    lcs = extract_values_from_raster(lc, shapes)
    geos = extract_values_from_raster(geo, shapes)

    n = len(shapes)
    data = {
        'elev': elevs,
        'fault': faults,
        'slope': slopes,
        'LC': lcs,
        'Geol': geos,
        'ls': [label] * n
    }    

    # Build a GeoDataFrame and return it
    return gpd.GeoDataFrame(data)

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
