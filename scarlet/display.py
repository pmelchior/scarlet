import numpy as np
from matplotlib.colors import Normalize

class Asinh(Normalize):
    """Use arcsinh to map image intensities in matplotlib
    """
    def __init__(self, vmin=None, vmax=None, img=None, Q=10, clip=False):
        """Initialize

        Parameters
        ----------
        vmin: float, default=`None`
            Minimum pixel value.
            If `vmin` is `None`, the minimum value of `img` will be used.
        vmax: float, default=`None`
            Maximum pixel value.
            If `vmin` is `None`, the maximum value of `img` will be used.
        img: array_like, default=`None`
            This should be an array with dimensions (bands, height, width).
            Either `vmin` and `vmax` or `img` is required to create an Asinh mapping
        Q: float, default=`10`
            Stretching parameter for the arcsinh function.
            This reduces to a linear stretch when Q=0
        clip: bool, default=`False`
            Whether or not to clip values.
            This is a default from matplotlib, but clip=True is currently not supported.
        """
        self.Q = Q
        Normalize.__init__(self, vmin, vmax, clip)
        if img is not None:
            if vmin is None:
                vmin = np.ma.min(img)
            if vmax is None:
                vmax = np.ma.max(img)
        if vmin is not None and vmax is not None:
            self._set_scale(vmin, vmax)

    def _set_scale(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        self.data_range = vmax-vmin

    def inverse(self, value):
        """Invert the mapping

        This is used to undo the mapping, for example matplotlib uses this to
        generate the values in the colorbar.
        """
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        xmin, xmax = self.get_range()
        inner = self.Q*(xmax-xmin)*value+xmin
        return np.ma.array(self.data_range/self.Q*np.sinh(inner)+self.vmin)

    def scaled(self):
        """Have vmax and vmin been set?
        """
        return (self.vmin is not None and self.vmax is not None)
    
    def get_range(self):
        """Get the possible range for a given vmin and vmax
        """
        if not self.scaled():
            raise ValueError("No range until scaled")
        xmin = self.asinh(self.vmin)
        xmax = self.asinh(self.vmax)
        return xmin, xmax
    
    def asinh(self, value):
        """Calculate asinh with the appropriate stretching
        """
        return np.ma.array(np.arcsinh(self.Q*(value-self.vmin)/self.data_range)/self.Q)

    def __call__(self, value, clip=None):
        """Map the `value` onto [0,1]
        """
        if self.vmin is None:
            vmin = np.min(value)
        else:
            vmin = self.vmin
        if self.vmax is None:
            vmax = np.max(value)
        else:
            vmax = self.vmax
        self._set_scale(vmin, vmax)
        result = self.asinh(value)
        xmin, xmax = self.get_range()
        result = (result-xmin)/(xmax-xmin)
        return result

def img_to_rgb(img, vmin=None, vmax=None, Q=10, fill_value=0, filter_indices=None):
    """Convert an image array into an RGB image array

    Parameters
    ----------
    img: array_like
        This should be an array with dimensions (bands, height, width).
    vmin: float, default=`None`
        Minimum pixel value.
        If `vmin` is `None`, the minimum value of `img` will be used.
    vmax: float, default=`None`
        Maximum pixel value.
        If `vmin` is `None`, the maximum value of `img` will be used.
    Q: float, default=`10`
        Stretching parameter for the arcsinh function.
        This reduces to a linear stretch when Q=0
    fill_value: float, default=`0`
        Value to use for any masked pixels.
    filter_indices: list, default=`None`
        Order of the filters to convert to RGB.
        Since typically img is ordered from bluest to reddest colors,
        if `filter_indices` is `None` a default value of [2,1,0] is
        used, inverting the order to properly match the reddest color to R.

    Returns
    -------
    rgb: numpy array with dtype uint8
        RGB image
    """
    if filter_indices is None:
        filter_indices = [2,1,0]
    if len(filter_indices)!=3:
        raise ValueError("filter_indices must be a list with 3 entries, not {0}".format(filter_indices))

    norm = Asinh(vmin, vmax, img, Q)
    # The result is a masked array, so we have to set the masked pixels to something
    rgb = norm(img).filled(fill_value)
    # The result of calling Asinh is an array mapped to [0,1]
    # We need to map this to a uint8 in the range [0,255]
    rgb *= 255
    rgb[rgb<0] = 0
    rgb[rgb>255] = 255
    rgb = rgb.astype(np.uint8)
    # Now we move the SED to the outer axis and use only the selected colors
    rgb = np.transpose(rgb[filter_indices], axes=(1,2,0))
    return rgb
