import numpy as np
from scipy.ndimage import convolve


def make_edges3d(segmentation):
    """ Make 3d edge volume from 3d segmentation
    """
    # NOTE we add one here to make sure that we don't have zero in the segmentation
    gz = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1, 1))
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3, 1))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 1, 3))
    return ((gx ** 2 + gy ** 2 + gz ** 2) > 0)


def make_edges3d2d(segmentation):
    """ Make 3d edge volume from 3d segmentation
        but only compute the edges in xy-plane.
        This may be more appropriate for anisotropic volumes.
    """
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3, 1))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 1, 3))
    return ((gx ** 2 + gy ** 2) > 0)


def make_edges2d(segmentation):
    """ Make 2d edge volume from 3d segmentation
        but only compute the edges in xy-plane.
        This may be more appropriate for anisotropic volumes.
    """
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3))
    return ((gx ** 2 + gy ** 2) > 0)


def make_edges(seg, force_2d=False):
    ndim = seg.ndim
    if ndim == 2:
        edges = make_edges2d(seg)
    elif ndim == 3 and force_2d:
        edges = make_edges3d2d(seg)
    elif ndim == 3:
        edges = make_edges3d(seg)
    else:
        raise ValueError("Only 2d or 3d segmentations are supported")
    return edges
