# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-image==0.25.2",
#     "python-kymograph",
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.4",
# ]
# ///

import marimo

__generated_with = "0.11.18"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
async def _():
    import micropip
    try:
        import kymograph_py
    except ImportError:
        await micropip.install("scikit-image")
        await micropip.install("tifffile")
        await micropip.install("python-kymograph")
        await micropip.install("imageio", keep_going=True)
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    return micropip, np, plt, skimage


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        You can upload your own file!

        or use scikit-image's cells 3d:
        https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif

        or, see a synthetic example of a blob dividing into two blobs
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # You can upload your own file!
    # or use scikit-image's cells 3d:
    # https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif 
    f = mo.ui.file(kind="area")  # Upload the file using Marimo UI
    f
    return (f,)


@app.cell(hide_code=True)
def _(f, np):
    if f.name() != None: # check if file was uploaded;
        # Create a BytesIO object to read with skimage
        import io
        from skimage import io as skio
    
        image_file = io.BytesIO(f.contents())
        img_raw = skio.imread(image_file, plugin='tifffile')
    
        print("Image loaded successfully:", img_raw.shape)
    else: # make fake synthetic numpy image
        # Define dimensions (Z, Y, X)
        nz, ny, nx = 60, 256, 256
        volume = np.zeros((nz, ny, nx), dtype=np.uint8)
    
        # Create coordinate grids for Y and X; centers of the image
        y, x = np.ogrid[:ny, :nx]
        center_x, center_y = nx // 2, ny // 2
    
        # Set parameters: smaller circles and larger separation at the end
        r_min, r_max, r_end = 4.0, 40.0, 20.0   # Radii for initial, maximum, and final circles
        shift_max = 60.0                         # Maximum vertical shift for the split circles
        split_threshold = 0.3                   # normalized time for circles to split
    
        for z in range(nz):
            # Normalize z coordinate to [0, 1]
            z_norm = z / (nz - 1)
            if z_norm < split_threshold:
                # Single circle: radius grows linearly from r_min to r_max up to split_threshold
                r = r_min + (r_max - r_min) * (z_norm / split_threshold)
                mask = (y - center_y)**2 + (x - center_x)**2 <= r**2
            else:
                # Two circles: transition parameter t goes from 0 (at split_threshold) to 1 (at z_norm = 1)
                t = (z_norm - split_threshold) / (1 - split_threshold)
                # The radius shrinks linearly from r_max to r_end and vertical shift increases linearly
                r = r_max - (r_max - r_end) * t
                shift = shift_max * t
                mask = ((y - (center_y - shift))**2 + (x - center_x)**2 <= r**2) | \
                       ((y - (center_y + shift))**2 + (x - center_x)**2 <= r**2)
            volume[z][mask] = 1
            img_raw = volume
    return (
        center_x,
        center_y,
        image_file,
        img_raw,
        io,
        mask,
        nx,
        ny,
        nz,
        r,
        r_end,
        r_max,
        r_min,
        shift,
        shift_max,
        skio,
        split_threshold,
        t,
        volume,
        x,
        y,
        z,
        z_norm,
    )


@app.cell(hide_code=True)
def _(f, img_raw, np):
    if f.name() == 'cells3d.tif':
        # Separate membranes and nuclei from the raw image
        membranes = img_raw[:,0,...]
        nuclei = img_raw[:,1,...]

        # Set the image of interest to the nuclei
        image = nuclei

        ### centroids
        # simply add the center of every frame (slightly adjusted) for demonstration
        centroids_ = np.tile(np.array([int(image.shape[1] // 2 * 1.2), int(image.shape[2] // 2 * 0.85)]), (image.shape[0], 1))


    else:
        image = img_raw
        if image.ndim > 3:
            # keep the first index from all dimensions except the last three
            image = image[(0,) * (image.ndim - 3) + (...,)]
            membranes = None
            nuclei = None

        ### centroids
        # simply add the center of every frame for demonstration
        centroids_ = np.tile(np.array([int(image.shape[1] // 2), int(image.shape[2] // 2)]), (image.shape[0], 1))

    return centroids_, image, membranes, nuclei


@app.cell
def _():
    from kymograph_py import make_kymograph
    print(make_kymograph.__doc__)
    return (make_kymograph,)


@app.cell(hide_code=True)
def _(mo):
    slider_w = mo.ui.slider(start=2, stop=20, step=2, label="width", value=10)
    slider_h = mo.ui.slider(start=40, stop=300, step=20, label="height", value=100)
    slider_s = mo.ui.slider(start=1, stop=5, step=1, label="skip_step", value=2)

    mo.vstack([slider_w, slider_h, slider_s])
    return slider_h, slider_s, slider_w


@app.cell
def _(centroids_, image, make_kymograph, plt, slider_h, slider_s, slider_w):
    ### centroids ###
    # can simply add the center of every frame, as below
    # centroids_ = np.tile(np.array([int(image.shape[1] // 2), int(image.shape[2] // 2)]), (image.shape[0], 1))


    width_ = slider_w.value
    height_ = slider_h.value
    skip_step_ = slider_s.value

    # Make kymograph
    kymo = make_kymograph(image, centroids_,
                          width=width_, 
                          height=height_, 
                          skip_step=skip_step_)

    plt.imshow(kymo)
    return height_, kymo, skip_step_, width_


@app.cell
def _(kymo, plt):
    # Remove axes and change colormap
    plt.imshow(kymo, cmap = 'gray')
    plt.axis('off')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
