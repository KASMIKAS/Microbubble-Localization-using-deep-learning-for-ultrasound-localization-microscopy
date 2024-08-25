import numpy as np
from scipy.ndimage import convolve
# zoom
def localizeRadialSymmetry(I, fwhmz, fwhmx):
    # Number of grid points
    Nz, Nx = I.shape

    # Radial symmetry algorithm
    zm_onerow = np.arange(-(Nz-1)/2.0+0.5, (Nz-1)/2.0, 1.0)
    zm = np.tile(zm_onerow[:, np.newaxis], (1, Nx-1))
    xm_onecol = np.arange(-(Nx-1)/2.0+0.5, (Nx-1)/2.0, 1.0)
    xm = np.tile(xm_onecol[np.newaxis, :], (Nz-1, 1))

    # Calculate derivatives along 45-degree shifted coordinates (u and v)
    dIdu = I[0:Nz-1, 1:Nx] - I[1:Nz, 0:Nx-1]
    dIdv = I[0:Nz-1, 0:Nx-1] - I[1:Nz, 1:Nx]

    # Smoothing the gradient of the I window
    h = np.ones((3, 3)) / 9
    fdu = convolve(dIdu, h, mode='constant')
    fdv = convolve(dIdv, h, mode='constant')
    dImag2 = fdu**2 + fdv**2

    # Slope of the gradient
    m = -(fdv + fdu) / (fdu - fdv)

    # Handle NaN values in m
    m[np.isnan(m)] = (dIdv + dIdu)[np.isnan(m)] / (dIdu - dIdv)[np.isnan(m)]
    m[np.isnan(m)] = 0  # Replace remaining NaNs with 0

    # Handle Inf values in m
    if np.isinf(m).any():
        m[np.isinf(m)] = 10 * np.max(m[~np.isinf(m)])

    # Calculate the z intercept of the line of slope m that goes through each grid midpoint
    b = zm - m * xm

    # Weight the intensity by square of gradient magnitude and inverse distance to gradient intensity centroid
    sdI2 = np.sum(dImag2)
    zcentroid = np.sum(dImag2 * zm) / sdI2
    xcentroid = np.sum(dImag2 * xm) / sdI2
    w = dImag2 / np.sqrt((zm - zcentroid)**2 + (xm - xcentroid)**2)

    # Least-squares minimization to determine the translated coordinate system origin
    zc, xc = lsradialcenterfit(m, b, w)
    return zc, xc

def lsradialcenterfit(m, b, w):
    # Least squares solution to determine the radial symmetry center
    wm2p1 = w / (m**2 + 1)
    sw = np.sum(wm2p1)
    smmw = np.sum(m**2 * wm2p1)
    smw = np.sum(m * wm2p1)
    smbw = np.sum(m * b * wm2p1)
    sbw = np.sum(b * wm2p1)
    det = smw * smw - smmw * sw
    xc = (smbw * sw - smw * sbw) / det  # relative to image center
    zc = (smbw * smw - smmw * sbw) / det  # relative to image center
    return zc, xc
