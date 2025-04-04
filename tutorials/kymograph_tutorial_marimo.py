# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-image==0.25.2",          
#     "python-kymograph"
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
        import pooch
    except ImportError:
        await micropip.install("scikit-image")
        await micropip.install("tifffile")
        await micropip.install("python-kymograph")
        # await micropip.install("git+https://github.com/eigenP/kymograph-py.git@main#egg=kymograph-py")

    import matplotlib.pyplot as plt
    import numpy as np
    return micropip, skimage, tifffile, kymograph_py, plt, np



@app.cell(hide_code=True)
def _(mo):
    mo.md(
    '''
     You can upload your own file!
 
     or use scikit-image's cells 3d:
     https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif
    ''')
    return



@app.cell(hide_code=True)
def _():
    # You can upload your own file!
    # or use scikit-image's cells 3d:
    # https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif 
    f = mo.ui.file(kind="area")  # Upload the file using Marimo UI
    f
    return f


@app.cell(hide_code=True)
def _():
    # Create a BytesIO object to read with skimage
    import io
    from skimage import io as skio
    
    image_file = io.BytesIO(f.contents())
    img_raw = skio.imread(image_file, plugin='tifffile')
    
    print("Image loaded successfully:", img_raw.shape)
    return io, image_file, img_raw



@app.cell(hide_code=True)
def _():
    if f.name() == 'cells3d.tif':
        # Separate membranes and nuclei from the raw image
        membranes = img_raw[:,0,...]
        nuclei = img_raw[:,1,...]
    
        # Set the image of interest to the nuclei
        image = nuclei

    else:
        image = img_raw
        if image.ndim > 3:
            # keep the first index from all dimensions except the last three
            image = image[(0,) * (image.ndim - 3) + (...,)]
            membranes = None
            nuclei = None
    return image, membranes, nuclei


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
def _(image, make_kymograph, np, nuclei, plt, slider_h, slider_s, slider_w):
    ### centroids
    # simply add the center of every frame (slightly adjusted) for demonstration
    centroids_ = np.tile(np.array([int(image.shape[1] // 2 * 1.2), int(image.shape[2] // 2 * 0.85)]), (image.shape[0], 1))


    width_ = slider_w.value
    height_ = slider_h.value
    skip_step_ = slider_s.value

    # Make kymograph
    kymo = make_kymograph(image, centroids_,
                          width=width_, 
                          height=height_, 
                          skip_step=skip_step_)

    plt.imshow(kymo)
    return centroids_, height_, kymo, skip_step_, width_


@app.cell
def _(kymo, plt):
    # Remove axes and change colormap
    plt.imshow(kymo, cmap = 'gray')
    plt.axis('off')
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
