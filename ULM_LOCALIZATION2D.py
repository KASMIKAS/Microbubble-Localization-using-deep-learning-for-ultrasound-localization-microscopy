import numpy as np
from scipy.ndimage import maximum_filter
from scipy.interpolate import interp2d
from skimage.feature import peak_local_max
from localizeRadialSymmetry import localizeRadialSymmetry
# zoom mochkil

import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects, gaussian_filter
from scipy.interpolate import interp2d, interp1d
from scipy.optimize import curve_fit
import cv2

def ULM_localization2D(MatIn, ULM):
    fwhmz = ULM['fwhm'][1]
    fwhmx = ULM['fwhm'][0]

    vectfwhmz = np.arange(-round(fwhmz/2), round(fwhmz/2)+1)
    vectfwhmx = np.arange(-round(fwhmx/2), round(fwhmx/2)+1)

    height, width, numberOfFrames = MatIn.shape
    MatIn = np.abs(MatIn)

    if 'LocMethod' not in ULM:
        ULM['LocMethod'] = 'radial'

    if 'parameters' not in ULM:
        ULM['parameters'] = {}

    if ULM['LocMethod'] == 'interp':
        if 'InterpMethod' not in ULM['parameters']:
            ULM['parameters']['InterpMethod'] = 'spline'
        if ULM['parameters']['InterpMethod'] in ['bilinear', 'bicubic']:
            print('Warning: Faster but pixelated, Weighted Average will be faster and smoother.')

    if 'NLocalMax' not in ULM['parameters']:
        if fwhmz == 3:
            ULM['parameters']['NLocalMax'] = 2
        else:
            ULM['parameters']['NLocalMax'] = 3

    # Prepare intensity matrix
    MatInReduced = np.zeros((height, width, numberOfFrames), dtype=MatIn.dtype)
    MatInReduced[1+round(fwhmz/2):height-round(fwhmz/2), 1+round(fwhmx/2):width-round(fwhmx/2), :] = \
        MatIn[1+round(fwhmz/2):height-round(fwhmz/2), 1+round(fwhmx/2):width-round(fwhmx/2), :]
    height, width, numberOfFrames = MatInReduced.shape

    # Detection and selection of microbubbles
    Mat2D = np.reshape(np.transpose(MatInReduced, (0, 2, 1)), (height * numberOfFrames, width))
    mask2D = (maximum_filter(Mat2D, size=3) == Mat2D).astype(int)
    mask = np.transpose(np.reshape(mask2D, (height, numberOfFrames, width)), (0, 2, 1))

    IntensityMatrix = MatInReduced * mask

    tempMatrix = np.sort(IntensityMatrix.reshape(-1, numberOfFrames), axis=0)[::-1]
    IntensityFinal = IntensityMatrix - np.ones(IntensityMatrix.shape) * tempMatrix[ULM['numberOfParticles'], :]
    MaskFinal = (mask * IntensityFinal > 0).astype(int)
    MaskFinal[np.isnan(MaskFinal)] = 0
    MaskFinal = (MaskFinal > 0) * IntensityMatrix

    index_mask = np.flatnonzero(MaskFinal)
    index_mask_z, index_mask_x, index_numberOfFrames = np.unravel_index(index_mask, (height, width, numberOfFrames))

    # Sub-wavelength localization of microbubbles
    averageXc = np.full(index_mask_z.shape, np.nan, dtype=MatIn.dtype)
    averageZc = np.full(index_mask_z.shape, np.nan, dtype=MatIn.dtype)

    for iscat in range(index_mask_z.shape[0]):
        IntensityRoi = MatIn[index_mask_z[iscat]+vectfwhmz[:, None], index_mask_x[iscat]+vectfwhmx, index_numberOfFrames[iscat]]

        if np.sum(maximum_filter(IntensityRoi, size=3) == IntensityRoi) > ULM['parameters']['NLocalMax']:
            continue

        if ULM['LocMethod'] == 'radial':
            Zc, Xc, sigma = LocRadialSym(IntensityRoi, fwhmz, fwhmx)
        elif ULM['LocMethod'] == 'wa':
            Zc, Xc, sigma = LocWeightedAverage(IntensityRoi, vectfwhmz, vectfwhmx)
        elif ULM['LocMethod'] == 'interp':
            Zc, Xc, sigma = LocInterp(IntensityRoi, ULM['parameters']['InterpMethod'], vectfwhmz, vectfwhmx)
        elif ULM['LocMethod'] == 'curvefitting':
            Zc, Xc, sigma = curveFitting(IntensityRoi, vectfwhmz, vectfwhmx)
        elif ULM['LocMethod'] == 'nolocalization':
            Zc, Xc, sigma = NoLocalization(IntensityRoi)
        else:
            raise ValueError('Wrong LocMethod selected')

        averageZc[iscat] = Zc + index_mask_z[iscat]
        averageXc[iscat] = Xc + index_mask_x[iscat]

        if sigma < 0 or sigma > 25:
            continue

        if abs(Zc) > fwhmz / 2 or abs(Xc) > fwhmx / 2:
            averageZc[iscat] = np.nan
            averageXc[iscat] = np.nan
            continue

    keepIndex = ~np.isnan(averageXc)
    ind = np.ravel_multi_index((index_mask_z[keepIndex], index_mask_x[keepIndex], index_numberOfFrames[keepIndex]),
                               (height, width, numberOfFrames))

    # Build MatTracking
    MatTracking = np.zeros((np.sum(keepIndex), 4), dtype=MatIn.dtype)
    MatTracking[:, 0] = MatInReduced.ravel()[ind]
    MatTracking[:, 1] = averageZc[keepIndex]
    MatTracking[:, 2] = averageXc[keepIndex]
    MatTracking[:, 3] = index_numberOfFrames[keepIndex]

    return MatTracking

# Additional localization functions
def ComputeSigmaScat(Iin, Zc, Xc):
    Nx, Nz = Iin.shape
    Isub = Iin - np.mean(Iin)
    px, pz = np.meshgrid(np.arange(1, Nx+1), np.arange(1, Nz+1))
    zoffset = pz - Zc + Nz / 2.0
    xoffset = px - Xc + Nx / 2.0
    r2 = zoffset**2 + xoffset**2
    sigma = np.sqrt(np.sum(Isub * r2) / np.sum(Isub)) / 2
    return sigma

def LocRadialSym(Iin, fwhm_z, fwhm_x):
    Zc, Xc = localizeRadialSymmetry(Iin, fwhm_z, fwhm_x)
    sigma = ComputeSigmaScat(Iin, Zc, Xc)
    return Zc, Xc, sigma

def NoLocalization(Iin):
    Xc, Zc = 0, 0
    sigma = ComputeSigmaScat(Iin, Zc, Xc)
    return Zc, Xc, sigma

def LocWeightedAverage(Iin, vectfwhm_z, vectfwhm_x):
    Zc = np.sum(Iin * vectfwhm_z[:, None]) / np.sum(Iin)
    Xc = np.sum(Iin * vectfwhm_x) / np.sum(Iin)
    sigma = ComputeSigmaScat(Iin, Zc, Xc)
    return Zc, Xc, sigma

def LocInterp(Iin, InterpMode, vectfwhm_z, vectfwhm_x):
    Nz, Nx = Iin.shape
    if InterpMode == 'spline':
        x = np.arange(1, Nx+1)
        z = np.arange(1, Nz+1)
        interp_func = interp2d(x, z, Iin, kind='cubic')
        xq = np.linspace(1, Nx, Nx * 10)
        zq = np.linspace(1, Nz, Nz * 10)
        In_interp = interp_func(xq, zq)
    else:
        In_interp = cv2.resize(Iin, (Nx * 10, Nz * 10), interpolation=cv2.INTER_CUBIC if InterpMode == 'bicubic' else cv2.INTER_LANCZOS4)
    max_idx = np.unravel_index(np.argmax(In_interp), In_interp.shape)
    iz, ix = max_idx
    Zc = vectfwhm_z[0] - 0.5 + iz / 10 - 0.05
    Xc = vectfwhm_x[0] - 0.5 + ix / 10 - 0.05
    sigma = ComputeSigmaScat(Iin, Zc, Xc)
    return Zc, Xc, sigma

def curveFitting(Iin, vectfwhm_z, vectfwhm_x):
    def myGaussFunc(x_pos, mesh_pos):
        sigGauss_z = vectfwhm_z[-1] * 0 + 1
        sigGauss_x = vectfwhm_x[-1] * 0 + 1
        return np.exp(-((mesh_pos[:, :, 0] - x_pos[0]) ** 2 / (2 * sigGauss_z ** 2) +
                         (mesh_pos[:, :, 1] - x_pos[1]) ** 2 / (2 * sigGauss_x ** 2)))
    meshX, meshZ = np.meshgrid(vectfwhm_x, vectfwhm_z)
    meshIn = np.stack([meshX, meshZ], axis=-1)
    x_out, _ = curve_fit(lambda x_pos, *args: myGaussFunc(x_pos, meshIn).flatten(), 
                        np.array([0, 0]), 
                        Iin.flatten() / np.max(Iin),
                        p0=[0, 0])
    Zc, Xc = x_out
    sigma = ComputeSigmaScat(Iin, Zc, Xc)
    return Zc, Xc, sigma


