import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from sklearn.base import BaseEstimator
from terrain_analysis import make_prob_raster_data, convert_to_rasterio

class DummyClf(BaseEstimator):
    def fit(self, X, y): return self
    def predict_proba(self, X):
        #  return sum of features as probability
        s = X.sum(axis=1)
        p = (s - s.min()) / (s.max() - s.min())
        # return [[1-p, p]]
        return np.vstack([1-p, p]).T

def make_raster(arr):
    # helper to turn a small array into a rasterio DatasetReader
    transform = rasterio.transform.from_origin(0, arr.shape[0], 1, 1)
    profile = dict(driver='GTiff', height=arr.shape[0],
                   width=arr.shape[1], count=1,
                   dtype=arr.dtype, crs='EPSG:4326',
                   transform=transform)
    m = MemoryFile()
    with m.open(**profile) as dst:
        dst.write(arr, 1)
    return m.open()

def test_make_prob_raster_data_simple():
    # build four identical 2×2 rasters
    arr = np.array([[1,2],[3,4]], dtype='float32')
    rasters = [make_raster(arr) for _ in range(5)]
    clf = DummyClf()
    # call the function
    out = make_prob_raster_data(*rasters, classifier=clf)
    assert out.shape == (2,2)
    assert np.all(out >= 0) and np.all(out <=1)
    # the pixel at (1,1) with sum=20 (from all rasters) should have the highest probability
    assert out[1,1] == pytest.approx(out.max())