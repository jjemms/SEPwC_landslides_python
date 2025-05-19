"""This contins a single function to caluate the Euclidian
   distance from a value in a numpy ratser, using a rasterio
   object as a template. It matches GDAL's 'proximity' function."""

import rasterio
from scipy.ndimage import distance_transform_edt
import numpy as np

def proximity(raster, rasterised, value):
    """Calculate distance to source pixel value in every
    cell of the raster in pixel units

    matches GDAL's 'proximity' function by returning the
    number of pixels to nearest cell equal to 'value'
    """

    # Build a boolean mask
    mask = (rasterised != value)

    # Read affine transform to get map‐unit pixel sizes
    gt = raster.transform
    dx = gt.a       # pixel width (map units)
    dy = -gt.e      # pixel height (map units)

    # Compute distances in map units
    dist_map = distance_transform_edt(mask, sampling=(dy, dx))

    # Convert back to pixel‐distance units
    dist_pix = dist_map / dx

    return dist_pix
