import pytest
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from shapely.geometry import Point
from terrain_analysis import build_dataframe, extract_values_from_raster


def make_mem_raster(data, dtype):
    """
    create an in-memory single-band raster from a 2D NumPy array.
    """
    transform = from_origin(0, 2, 1, 1)
    profile = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': dtype,
        'crs': 'EPSG:4326',
        'transform': transform
    }
    memfile = MemoryFile()
    with memfile.open(**profile) as dst:
        dst.write(data, 1)
    return memfile.open()

class TestBuildDataFrame:
    def test_build_dataframe_small(self):
        # Make 2×2 rasters
        dem_data       = np.array([[1, 2], [3, 4]], dtype='float32')
        geology_data   = np.array([[5, 6], [7, 8]], dtype='int32')
        landcover_data = np.array([[9,10],[11,12]], dtype='int32')
        faults_data    = np.array([[0, 1], [1, 0]], dtype='int32')

        dem       = make_mem_raster(dem_data,       'float32')
        geology   = make_mem_raster(geology_data,   'int32')
        landcover = make_mem_raster(landcover_data, 'int32')
        faults    = make_mem_raster(faults_data,    'int32')

        # build GeoDataFrame of exactly 2 landslide points
        pts = [Point(0.5, 1.5), Point(1.5, 0.5)]
        landslides_gdf = gpd.GeoDataFrame(geometry=pts, crs=dem.crs)

        # call with n_samples=2
        df = build_dataframe(
            dem, geology, landcover, faults,
            landslides_gdf,
            n_samples=2
        )

        assert isinstance(df, gpd.GeoDataFrame)
        # 2 positives + 2 negatives = 4 rows
        assert len(df) == 4

        # they must have these columns
        for col in ['elevation','geology','landcover','faults','landslide','geometry']:
            assert col in df.columns

        # exactly two ones and two zeros
        labels = df['landslide'].tolist()
        assert labels.count(1) == 2
        assert labels.count(0) == 2

        # elevation values should match the mock dem_data
        assert all(isinstance(v, float) for v in df['elevation'])
        assert min(df['elevation']) >= 1.0
        assert max(df['elevation']) <= 4.0