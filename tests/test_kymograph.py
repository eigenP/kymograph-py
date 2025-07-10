import numpy as np
import pytest
from kymograph_py import make_kymograph

def test_make_kymograph_output():
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Generate a random image stack with shape (60, 256, 256)
    image = np.random.random((60, 256, 256))
    
    # Compute centroids as in your demo:
    # y coordinate: int(image.shape[1] // 2 * 1.2)
    # x coordinate: int(image.shape[2] // 2 * 0.85)
    centroid_y = int(image.shape[1] // 2 * 1.2)
    centroid_x = int(image.shape[2] // 2 * 0.85)
    # Centroids are stored in (y, x) order
    centroids = np.tile(np.array([centroid_y, centroid_x]), (image.shape[0], 1))
    
    # Call make_kymograph with specified parameters
    kymo = make_kymograph(image, centroids, width=10, height=100, skip_step=2)
    
    # Check that kymo is a NumPy array with two dimensions
    assert isinstance(kymo, np.ndarray)
    assert kymo.ndim == 2
    
    # Explanation for expected shape:
    # Each slice extracted from an image has height:
    #   y0 = max(centroid_y - height//2, 0) = max(108 - 50, 0) = 58
    #   y1 = min(centroid_y + height//2, image height) = min(108 + 50, 256) = 158
    # So the slice height is 158 - 58 = 100.
    # Similarly, each slice has width:
    #   x0 = max(centroid_x - width//2, 0) = max(153 - 5, 0) = 148
    #   x1 = min(centroid_x + width//2, image width) = min(153 + 5, 256) = 158
    # So the slice width is 10.
    #
    # With skip_step=2, the number of slices is 60 // 2 = 30.
    # The final kymograph is formed by concatenating slices along the horizontal axis,
    # resulting in a shape of (100, 30 * 10) = (100, 300).
    
    expected_shape = (100, 300)
    assert kymo.shape == expected_shape
    
    # Check that the kymograph values are within the expected range of the original image (0, 1)
    assert kymo.min() >= 0 and kymo.max() <= 1
