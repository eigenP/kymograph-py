import numpy as np
import pytest
from kymograph_py import make_kymograph
from kymograph_py import apply_drift_correction_2D, zero_shift_multi_dimensional
import csv

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


def test_make_kymograph_invalid_dims():
    image = np.random.random((256, 256))
    centroids = np.array([[0, 0]])
    with pytest.raises(ValueError):
        make_kymograph(image, centroids)



def test_apply_drift_correction_writes_csv(tmp_path):
    video = np.zeros((3, 10, 10))
    video[0, 5, 5] = 1
    video[1, 5, 6] = 1
    video[2, 5, 7] = 1

    csv_file = tmp_path / "drift.csv"
    _, table = apply_drift_correction_2D(
        video, save_drift_table=True, csv_filename=str(csv_file)
    )

    assert csv_file.exists()
    with open(csv_file, newline="") as f:
        rows = list(csv.DictReader(f))

    # should have as many rows as time points
    assert len(rows) == video.shape[0]
    # check a known value from the drift table
    assert float(rows[1]["dx"]) == -1.0
    assert float(rows[2]["cum_dx"]) == -2.0
    # the returned table should match the csv contents
    assert float(table[2]["cum_dx"]) == float(rows[2]["cum_dx"])


def test_apply_drift_correction_aligns_shifted_video():
    video = np.zeros((3, 10, 10))
    for i in range(3):
        video[i, 5, 5 + i] = 1

    corrected, _ = apply_drift_correction_2D(video)
    for frame in corrected:
        assert np.argmax(frame) == np.ravel_multi_index((5, 5), frame.shape)


def test_zero_shift_multi_dimensional_basic():
    arr = np.arange(16).reshape(4, 4)
    shifted = zero_shift_multi_dimensional(arr, shifts=[1, -1], fill_value=-1)
    expected = np.full_like(arr, -1)
    expected[1:, :-1] = arr[:-1, 1:]
    assert np.array_equal(shifted, expected)
