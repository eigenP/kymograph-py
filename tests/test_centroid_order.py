import numpy as np
from kymograph_py import make_kymograph


def test_centroid_coordinate_order():
    # Create an image stack where pixel values depend only on the x coordinate
    image = np.arange(3, dtype=float)[None, None, :].repeat(3, axis=1)
    image = image.repeat(2, axis=0)  # Shape (2, 3, 3)

    # Centroid is given in (y, x) order
    centroids = np.tile(np.array([1, 2]), (image.shape[0], 1))

    kymo = make_kymograph(image, centroids, width=1, height=1)

    assert kymo.shape == (1, image.shape[0])
    expected = np.full((1, image.shape[0]), 2, dtype=float)
    assert np.array_equal(kymo, expected)
