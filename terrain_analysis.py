# pylint: disable=R0913,R0914,R0917
"""terrain_analysis.py

This script uses a machine learning model to predict landslide risk based on topography,
geology, land cover, and fault data. The model trains a RandomForestClassifier on a dataset
of known landslide locations. Then the script uses the trained model on each pixel to create
a landslide risk map (values between 0.0-1.0).
"""
from typing import List
import argparse
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.io import MemoryFile, DatasetReader
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from proximity \
    import proximity


def convert_to_rasterio(raster_data: np.ndarray, template_raster: DatasetReader) -> DatasetReader:
    """Convert a numpy array to a rasterio in-memory dataset."""
    profile = template_raster.profile.copy()
    profile.update(dtype=raster_data.dtype, count=1)
    memfile = MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(raster_data, 1)
    return memfile.open()


def extract_values_from_raster(raster: DatasetReader, points: List[Point]) -> List[float]:
    """Extract values from a raster at given point locations, with bounds checking."""
    values = []
    arr = raster.read(1)
    nrows, ncols = arr.shape
    for point in points:
        row, col = raster.index(point.x, point.y)
        if 0 <= row < nrows and 0 <= col < ncols:
            values.append(float(arr[row, col]))
        else:
            values.append(float('nan'))  # or 0, or raise, depending on desired behavior
    return values


def build_dataframe(
    dem: DatasetReader,
    geology: DatasetReader,
    landcover: DatasetReader,
    faults: DatasetReader,
    landslides_gdf: gpd.GeoDataFrame,
    n_samples: int = 1000
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of positive and negative samples for training."""
    np.random.seed(42)
    if not all(landslides_gdf.geometry.geom_type == 'Point'):
        landslides_gdf = landslides_gdf.copy()
        landslides_gdf.geometry = landslides_gdf.geometry.centroid
    if len(landslides_gdf) >= n_samples:
        pos_pts = list(landslides_gdf.sample(n=n_samples, random_state=42).geometry)
    else:
        pos_pts = list(landslides_gdf.geometry)
        while len(pos_pts) < n_samples:
            pos_pts.append(np.random.choice(pos_pts))
    landslide_shapes = [(pt, 1) for pt in pos_pts]
    mask = rasterize(
        landslide_shapes,
        out_shape=(dem.height, dem.width),
        transform=dem.transform,
        fill=0,
        dtype='uint8'
    )
    rows, cols = np.where(mask == 0)
    if rows.size == 0:
        raise ValueError("No negative samples found in the raster mask.")
    indices = list(zip(rows, cols))
    replace = len(indices) < n_samples
    choice = np.random.choice(len(indices), size=n_samples, replace=replace)
    neg_pts = []
    for idx in choice:
        r, c = indices[idx]
        x, y = dem.transform * (c + 0.5, r + 0.5)
        neg_pts.append(Point(x, y))
    def sample_all(pts):
        return {
            'elevation': extract_values_from_raster(dem, pts),
            'geology': extract_values_from_raster(geology, pts),
            'landcover': extract_values_from_raster(landcover, pts),
            'faults': extract_values_from_raster(faults, pts),
            'slope': extract_values_from_raster(dem, pts)
        }
    pos_data = sample_all(pos_pts)
    neg_data = sample_all(neg_pts)
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
    combined = pd.concat([df_pos, df_neg], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry='geometry', crs=crs)


def make_classifier(x, y, verbose=False):
    """Train a RandomForestClassifier and return the trained model."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)
    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    if verbose:
        print(f"Train accuracy: {train_acc:.2f}, Test accuracy: {test_acc:.2f}")
    clf.scaler = scaler
    return clf


def make_prob_raster_data(
    topo: DatasetReader,
    geo: DatasetReader,
    lc: DatasetReader,
    dist_fault: DatasetReader,
    slope: DatasetReader,
    classifier
) -> np.ndarray:
    """Generate a probability raster using the trained classifier."""
    h, w = topo.shape
    features_stack = np.stack(
        [
            topo.read(1).ravel(),
            geo.read(1).ravel(),
            lc.read(1).ravel(),
            dist_fault.read(1).ravel(),
            slope.read(1).ravel()
        ],
        axis=1
    )
    if hasattr(classifier, 'scaler'):
        features_stack = classifier.scaler.transform(features_stack)
    probs = classifier.predict_proba(features_stack)[:, 1]
    return probs.reshape(h, w)


def create_dataframe(
    topo: DatasetReader,
    geo: DatasetReader,
    lc: DatasetReader,
    dist_fault: DatasetReader,
    slope: DatasetReader,
    shapes: List[Point],
    label: int
) -> gpd.GeoDataFrame:
    """Sample raster data at given shapes and create a GeoDataFrame (no geometry column)."""
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
    return gpd.GeoDataFrame(data)


def main():
    """Main entry point for landslide risk analysis."""
    parser = argparse.ArgumentParser(
        prog="Landslide hazard using ML",
        description="Calculate landslide hazards using simple ML",
        epilog="Copyright 2024, Jon Hill"
    )
    parser.add_argument(
        '--topography',
        required=True,
        help="topographic raster file"
    )
    parser.add_argument(
        '--geology',
        required=True,
        help="geology raster file"
    )
    parser.add_argument(
        '--landcover',
        required=True,
        help="landcover raster file"
    )
    parser.add_argument(
        '--faults',
        required=True,
        help="fault location shapefile"
    )
    parser.add_argument("landslides", help="the landslide location shapefile")
    parser.add_argument("output", help="the output raster file")
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help="Print progress"
    )
    args = parser.parse_args()
    with rasterio.open(args.topography) as dem, \
         rasterio.open(args.geology) as geo, \
         rasterio.open(args.landcover) as lc:
        dem_arr = dem.read(1).astype('float32')
        xres, yres = dem.transform.a, -dem.transform.e
        dz_dy, dz_dx = np.gradient(dem_arr, yres, xres)
        slope_arr = np.hypot(dz_dx, dz_dy)
        slope = convert_to_rasterio(slope_arr.astype('float32'), dem)
        faults_gdf = gpd.read_file(args.faults).to_crs(dem.crs)
        fault_mask = rasterize(
            [(geom, 1) for geom in faults_gdf.geometry],
            out_shape=(dem.height, dem.width),
            transform=dem.transform,
            fill=0,
            dtype='uint8'
        )
        dist_arr = proximity(dem, fault_mask, 1)
        dist_fault = convert_to_rasterio(dist_arr.astype('float32'), dem)
        slides = gpd.read_file(args.landslides).to_crs(dem.crs)
        samples = build_dataframe(
            dem, geo, lc, dist_fault, slides,
            n_samples=len(slides)
        )
        x = samples[['elevation', 'geology', 'landcover', 'faults', 'slope']]
        y = samples['landslide']
        clf = make_classifier(x, y, verbose=args.verbose)
        probability = make_prob_raster_data(
            dem, geo, lc, dist_fault, slope, clf
        )
        out_source = convert_to_rasterio(
            probability.astype('float32'), dem
        )
        profile = out_source.profile
        with rasterio.open(args.output, 'w', **profile) as dst:
            dst.write(out_source.read(1), 1)
    if args.verbose:
        print(
            f"Landslide probability risk map saved to {args.output}"
        )


if __name__ == '__main__':
    main()
