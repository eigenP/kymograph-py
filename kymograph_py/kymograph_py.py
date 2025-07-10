################
# kymograph.py #
################


import numpy as np

import pandas as pd
from tqdm import tqdm


import matplotlib.pyplot as plt

from skimage.registration import phase_cross_correlation
# from skimage.registration._phase_cross_correlation import _upsampled_dft
# from scipy.ndimage import fourier_shift, gaussian_filter1d


import pandas as pd
from tqdm import tqdm
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation



from scipy.ndimage import center_of_mass
from scipy.interpolate import UnivariateSpline

def calculate_centroids(label_array):
    # This function calculates the centroids for each time point in the label array
    centroids = []
    for t in range(label_array.shape[0]):
        unique_labels = np.unique(label_array[t])
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]  # Exclude background
        if len(unique_labels) == 0:
            centroids.append(None)  # No labels at this time point
        else:
            # Calculate the centroid of the first label (assuming one label per time point for simplicity)
            centroid = center_of_mass(label_array[t] == unique_labels[0])
            centroids.append(centroid)
    return centroids

def interpolate_centroids(centroids):
    # This function interpolates the centroids list, assuming None for missing centroids
    valid_times = [i for i, c in enumerate(centroids) if c is not None]
    valid_centroids = [centroids[i] for i in valid_times]
    
    # Separate into x and y coordinates for interpolation
    x_coords, y_coords = zip(*valid_centroids)
    
    # Create splines for each coordinate
    spline_x = UnivariateSpline(valid_times, x_coords, k=3, ext=3)  # ext=3 allows extrapolation
    spline_y = UnivariateSpline(valid_times, y_coords, k=3, ext=3)
    
    # Interpolate for all times
    interpolated_x = spline_x(range(len(centroids)))
    interpolated_y = spline_y(range(len(centroids)))
    
    # Combine the coordinates back into a list of tuples
    interpolated_centroids = np.column_stack((interpolated_x, interpolated_y))
    return interpolated_centroids

def process_labels(label_array):
    centroids = calculate_centroids(label_array)
    interpolated_centroids = interpolate_centroids(centroids)
    return interpolated_centroids

# Example usage:
# Assuming labels is a numpy array with shape (T, Y, X) where T is time and Y, X are spatial dimensions
# labels = np.load('your_labels.npy')
# result = process_labels(labels)
# print(result)




def zero_shift_multi_dimensional(arr, shifts = 0, fill_value=0):
    """
    Shift the elements of a multi-dimensional NumPy array along each axis by specified amounts, filling the vacant positions with a specified fill value.

    :param arr: A multi-dimensional NumPy array
    :param shifts: A single integer or a list/tuple of integers specifying the shift amounts for each axis
    :param fill_value: An optional value to fill the vacant positions after the shift (default is 0)
    :return: A new NumPy array with the elements shifted and the specified fill value in the vacant positions
    """
    # Ensure shifts is a list or tuple of integers, or a single integer
    if isinstance(shifts, int):
        shifts = [shifts] * arr.ndim
    elif isinstance(shifts, (list, tuple)):
        if len(shifts) != arr.ndim:
            raise ValueError("Length of shifts must be equal to the number of dimensions in the array.")
        if not all(isinstance(shift, int) for shift in shifts):
            raise TypeError("All shift values must be integers.")
    else:
        raise TypeError("Shifts must be a single integer or a list/tuple of integers.")

    # Initialize the result array with the fill value
    result = np.full_like(arr, fill_value)
    # Initialize slices for input and output arrays
    slices_input = [slice(None)] * arr.ndim
    slices_output = [slice(None)] * arr.ndim

    # Apply the shifts
    for axis, shift in enumerate(shifts):
        if shift > 0:
            slices_input[axis] = slice(None, -shift)
            slices_output[axis] = slice(shift, None)
        elif shift < 0:
            slices_input[axis] = slice(-shift, None)
            slices_output[axis] = slice(None, shift)

    # Perform the shift and fill in the result array
    result[tuple(slices_output)] = arr[tuple(slices_input)]
    return result




def _2D_weighted_image(image, overlap):
    '''
    # image := image shape
    # overlap := in pixels

    # Example usage
    # _2D_window = _2D_weight(image, overlap)
    '''

    # 1D weight function based on cubic spline
    def weight_1d(x):
        return 3 * x**2 - 2 * x**3

    # Initialize weight matrix with ones
    weight_2d = np.ones_like(image, dtype = np.float16)

    # Apply weight function to the top, bottom, left, and right overlap regions
    for i in range(overlap):
        weight = weight_1d(i / (overlap - 1))
        weight_2d[i, :] *= weight
        weight_2d[-(i + 1), :] *= weight
        weight_2d[:, i] *= weight
        weight_2d[:, -(i + 1)] *= weight

    weighted_image = image * weight_2d

    return weighted_image



def estimate_drift_2D(frame1, frame2, return_ccm = False):
    """
    Estimate the xy-drift between two 2D frames using cross-correlation.

    :param frame1: 2D numpy array, first frame
    :param frame2: 2D numpy array, second frame
    :return: Tuple (dx, dy), estimated drift in x and y directions
    """
    # Calculate the cross-correlation matrix
    # shift, error, diffphase = phase_cross_correlation(frame1, frame2)

    min_size_pixels = min(frame1.shape)

    # frame1 = _2D_weighted_image(frame1, min_size_pixels // 16)
    # frame2 = _2D_weighted_image(frame2, min_size_pixels // 16)


    frame1_max_proj_x = np.max(frame1, axis = 0)
    frame2_max_proj_x = np.max(frame2, axis = 0)

    frame1_max_proj_y = np.max(frame1, axis = 1)
    frame2_max_proj_y = np.max(frame2, axis = 1)

    # Apply gaussian smoothing for robustness
    # frame1_max_proj_x = gaussian_filter1d(frame1_max_proj_x, sigma = 3, radius = 5)
    # frame2_max_proj_x = gaussian_filter1d(frame2_max_proj_x, sigma = 3, radius = 5)

    # frame1_max_proj_y = gaussian_filter1d(frame1_max_proj_y, sigma = 3, radius = 5)
    # frame2_max_proj_y = gaussian_filter1d(frame2_max_proj_y, sigma = 3, radius = 5)



    shift_x, error, diffphase = phase_cross_correlation(frame1_max_proj_x,
                                                      frame2_max_proj_x)

    shift_y, error, diffphase = phase_cross_correlation(frame1_max_proj_y,
                                                      frame2_max_proj_y)

    shift = np.array((shift_x[0], shift_y[0]))





    if return_ccm:
        # Calculate the upsampled DFT, again to show what the algorithm is doing
        # behind the scenes.  Constants correspond to calculated values in routine.
        # See source code for details.
        # image_product = np.fft.fft2(frame1) * np.fft.fft2(frame2).conj()
        # cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()

        frame1 = np.vstack((frame1_max_proj_y,frame1_max_proj_y))
        frame2 = np.vstack((frame2_max_proj_y,frame2_max_proj_y))

        image_product = np.fft.fft2(frame1) * np.fft.fft2(frame2).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))

        return shift, cc_image.real
    else:
        return shift


def apply_drift_correction_2D(
    video_data,
    reverse_time=False,
    save_drift_table=False,
    csv_filename="drift_table.csv",
    dx_dy_weights=[1, 1],
):
    """
    Apply drift correction to video data.

    This function corrects for drift in video data frame by frame. It calculates the drift between
    consecutive frames using the `estimate_drift_2D` function and applies corrections to align the frames.
    The cumulative drift is calculated and stored. Optionally, a table of drift values can be saved
    to a CSV file.

    :param video_data: A 3D numpy array representing the video data. The dimensions should be (time, x, y).
    :param reverse_time: If True, process the frames in reverse chronological order.
                         If 'both', uses a bidirectional estimation for drift correction.
                         Default is False.
    :param save_drift_table: A boolean indicating whether to save the drift values to a CSV file. Default is False.
    :param csv_filename: The name of the CSV file to save the drift table to. Default is 'drift_table.csv'.
    :param dx_dy_weights: A list of weights to apply to the cumulative dx and dy. Default is [1, 1].
    :return: A tuple containing two elements:
        - corrected_data: A 3D numpy array of the same shape as video_data, representing the drift-corrected video.
        - drift_table: A pandas DataFrame containing the drift values, cumulative drift, and time points.
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # Get dimensions
    t_shape, x_shape, y_shape = video_data.shape

    # Initialize corrected data array
    corrected_data = np.zeros_like(video_data)

    # Get minimum value
    min_value = video_data.min()

    # Initialize drift records
    drift_records = []

    # Initialize cumulative drift arrays
    cum_dx_array = np.zeros(t_shape)
    cum_dy_array = np.zeros(t_shape)

    if reverse_time == "both":
        # Process frames bidirectionally

        # Forward pass cumulative drift
        cum_dx_forward = np.zeros(t_shape)
        cum_dy_forward = np.zeros(t_shape)

        for time_point in tqdm(range(1, t_shape), desc="Forward Drift Calculation"):
            dx, dy = estimate_drift_2D(
                video_data[time_point - 1], video_data[time_point]
            )

            # Threshold drift
            if abs(dx) > x_shape // 5:
                dx = 0
            if abs(dy) > y_shape // 5:
                dy = 0

            cum_dx_forward[time_point] = (
                cum_dx_forward[time_point - 1] + dx * dx_dy_weights[0]
            )
            cum_dy_forward[time_point] = (
                cum_dy_forward[time_point - 1] + dy * dx_dy_weights[1]
            )

        # Backward pass cumulative drift
        cum_dx_backward = np.zeros(t_shape)
        cum_dy_backward = np.zeros(t_shape)

        for time_point in tqdm(
            range(t_shape - 2, -1, -1), desc="Backward Drift Calculation"
        ):
            dx, dy = estimate_drift_2D(
                video_data[time_point + 1], video_data[time_point]
            )

            # Threshold drift
            if abs(dx) > x_shape // 5:
                dx = 0
            if abs(dy) > y_shape // 5:
                dy = 0

            cum_dx_backward[time_point] = (
                cum_dx_backward[time_point + 1] + dx * dx_dy_weights[0]
            )
            cum_dy_backward[time_point] = (
                cum_dy_backward[time_point + 1] + dy * dx_dy_weights[1]
            )

        # Average the forward and backward cumulative drifts
        cum_dx_array = (cum_dx_forward + cum_dx_backward) / 2
        cum_dy_array = (cum_dy_forward + cum_dy_backward) / 2

        # Apply cumulative drift to each frame
        for time_point in tqdm(range(t_shape), desc="Applying Drift Correction"):
            corrected_frame = zero_shift_multi_dimensional(
                video_data[time_point],
                shifts=(int(cum_dy_array[time_point]), int(cum_dx_array[time_point])),
                fill_value=min_value,
            )
            corrected_data[time_point] = corrected_frame

            # Record drift
            drift_records.append(
                {
                    "Time Point": time_point,
                    "cum_dx": cum_dx_array[time_point],
                    "cum_dy": cum_dy_array[time_point],
                }
            )

    else:
        # Determine processing order
        if reverse_time:
            time_indices = range(t_shape - 1, -1, -1)
            compare_offset = 1
            direction = -1
        else:
            time_indices = range(t_shape)
            compare_offset = -1
            direction = 1

        for idx in tqdm(range(t_shape), desc="Calculating Cumulative Drift"):
            time_point = time_indices[idx]

            if (reverse_time and time_point == t_shape - 1) or (
                not reverse_time and time_point == 0
            ):
                # First frame in the sequence
                dx = dy = 0
            else:
                # Estimate drift
                dx, dy = estimate_drift_2D(
                    video_data[time_point + compare_offset],
                    video_data[time_point],
                )
                dx *= direction
                dy *= direction

                # Threshold drift
                if abs(dx) > x_shape // 4:
                    dx = 0
                if abs(dy) > y_shape // 4:
                    dy = 0

            # Update cumulative drift
            if idx == 0:
                cum_dx_array[time_point] = dx * dx_dy_weights[0]
                cum_dy_array[time_point] = dy * dx_dy_weights[1]
            else:
                prev_time_point = time_indices[idx - 1]
                cum_dx_array[time_point] = (
                    cum_dx_array[prev_time_point] + dx * dx_dy_weights[0]
                )
                cum_dy_array[time_point] = (
                    cum_dy_array[prev_time_point] + dy * dx_dy_weights[1]
                )

            # Record drift
            drift_records.append(
                {
                    "Time Point": time_point,
                    "dx": dx,
                    "dy": dy,
                    "cum_dx": cum_dx_array[time_point],
                    "cum_dy": cum_dy_array[time_point],
                }
            )

        # Apply cumulative drift to each frame
        for idx in tqdm(range(t_shape), desc="Applying Drift Correction"):
            time_point = time_indices[idx]
            corrected_frame = zero_shift_multi_dimensional(
                video_data[time_point],
                shifts=(int(cum_dy_array[time_point]), int(cum_dx_array[time_point])),
                fill_value=min_value,
            )
            corrected_data[time_point] = corrected_frame

    # Create drift table
    drift_table = pd.DataFrame(drift_records)
    drift_table.sort_values(by=["Time Point"], inplace=True)

    # Optionally save drift table
    if save_drift_table:
        drift_table.to_csv(csv_filename, index=False)

    # Return corrected data and drift table
    return corrected_data, drift_table



def make_kymograph(image: np.ndarray, centroids, stabilize = False, width=5, height=5, skip_step=1, colorbar=False) -> np.ndarray:
    '''
    Generates a kymograph from a time series of 2D images using specified centroid coordinates and 
    optionally applies a high contrast filter and a colorbar. The function expects a 3D image stack in 
    the order [T, Y, X].

    Parameters:
    -----------
    image: ndarray
        A 3-dimensional array representing a time series of 2D images. The array should be in the 
        order [T, Y, X] where T is time, Y is the Y-axis of the images, and X is the X-axis of the images.

    centroids: ndarray
        An array of shape (N, 2) where N is the number of time points. Each entry contains the (y, x)
        coordinates of the centroid at that time point.

    width: int, optional
        The width of the slice around the centroid. Default is 5 pixels.

    height: int, optional
        The height of the slice around the centroid. Default is 5 pixels.

    skip_step: int, optional
        Number of time points to skip between each slice in the final kymograph. Default is 1, which 
        means no time points are skipped.

    colorbar: bool, optional
        If True, a colorbar is added to the plot. Default is False.
    
    Returns:
    --------
    np.ndarray
        The generated kymograph as a 2D numpy array.

    Raises:
    -------
    ValueError
        If the input image array is not 3-dimensional.
    '''
    
    # Check image dimensions
    if image.ndim != 3:
        raise ValueError("Please make sure your image is 3D with dimensions [T, Y, X]!")

    # Initialize the kymograph list to hold slices from each time point
    kymograph_slices = []

    # Process each time point based on the centroid coordinates. The centroids
    # array is expected to contain coordinates in ``(y, x)`` order. This matches
    # the [T, Y, X] axis convention of the input image.
    for t, (y, x) in enumerate(centroids):
        # Calculate the bounding box around the centroid
        x0 = int(max(x - width // 2, 0))
        x1 = int(min(x + width // 2, image.shape[2]))
        y0 = int(max(y - height // 2, 0))
        y1 = int(min(y + height // 2, image.shape[1]))

        # Slice the image to get the ROI around the centroid
        slice_img = image[t, y0:y1, x0:x1]

        # Store the slice
        kymograph_slices.append(slice_img)

    if skip_step != 1:
        kymograph_slices = kymograph_slices[::skip_step] # Apply skip step


    # Determine the maximum size along the dimension of interest (0 in your case)
    max_size = max(arr.shape[0] for arr in kymograph_slices)

    # Pad each array to have the same size
    kymograph_slices_padded = [np.pad(arr, ((0, max_size - arr.shape[0]), (0, 0)), mode='constant') for arr in kymograph_slices]


    
    if stabilize:
        kymograph_slices_padded, drift_table = apply_drift_correction_2D(np.array(kymograph_slices_padded), reverse_time=False, dx_dy_weights=[0,1])
    
    # Concatenate all slices along the new axis to form the kymograph
    kymograph = np.concatenate(kymograph_slices_padded, axis=1)

    return kymograph

 
