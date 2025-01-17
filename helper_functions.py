import scipy
import numpy as np
import scipy.fft as fft
import scipy.constants as constants

from sys import stderr


def fourier_shell_correlation(vol_1=None, vol_2=None, shell=1):
    """
    Fourier Shell correlation of two volumes in N-dimensions. The volumes need to have the same number of dimensions.
    The integration within shells uses the mean of all voxels within that particular shell.
    """
    F1 = fft.fftshift(fft.fftn(vol_1))
    F2 = fft.fftshift(fft.fftn(vol_2))

    F1_sq = np.abs(F1) ** 2
    F2_sq = np.abs(F2) ** 2

    F1F2 = F1 * np.conjugate(F2)
    F1F2_int, n_ri = radial(F1F2, f=np.mean, shell_thickness=shell)

    F1_sq_int, _ = radial(F1_sq, f=np.mean, shell_thickness=shell)
    F2_sq_int, _ = radial(F2_sq, f=np.mean, shell_thickness=shell)
    F1F2_sq_int = F1_sq_int * F2_sq_int

    FSC = np.real(F1F2_int / np.sqrt(F1F2_sq_int))

    return FSC, n_ri


def r_factor(vol_ideal=None, vol_real=None, sphere_thickness=1):
    """
    R-factor of two diffraction patterns. The sum within each resolution sphere is calculated including the maximum resolution of the sphere.
    """
    F_ideal = np.sqrt(vol_ideal)
    F_real = np.sqrt(vol_real)

    im_dim = np.array(F_ideal.shape)
    im_center = np.array(np.unravel_index(np.argmax(F_ideal), F_ideal.shape))
    num_spheres = int(min((im_dim - im_center) / sphere_thickness))

    r = np.arange(
        sphere_thickness, (num_spheres + 1) * sphere_thickness, sphere_thickness
    )
    x, y, z = np.meshgrid(
        np.arange(F_ideal.shape[0]),
        np.arange(F_ideal.shape[1]),
        np.arange(F_ideal.shape[2]),
        indexing="ij",
    )
    spherical_mask = (x - im_center[0]) ** 2 + (y - im_center[1]) ** 2 + (
        z - im_center[2]
    ) ** 2 <= r[:, None, None, None] ** 2

    rf_arr = []
    for q_vals in range(num_spheres):
        qs = spherical_mask[q_vals] * (~np.isnan(F_real))

        F_ideal_q = F_ideal[qs]
        sum_F_ideal_q = np.sum(F_ideal_q)

        F_real_q = F_real[qs]
        sum_F_real_q = np.sum(F_real_q)

        abs_diff = np.abs(F_real / sum_F_real_q - F_ideal / sum_F_ideal_q)
        r_val_q = np.sum(abs_diff[qs])
        rf_arr.append(r_val_q)

    return np.array(rf_arr)


def _radial(image, f=np.mean, shell_thickness=1, **kwargs):
    """
    Radial integration in N-dimensions. Assumes the input array has the same size in all dimensions.
    Default integration method is the mean.
    """
    n_dim = len(image.shape)
    im_dim = image.shape[0]
    im_center = image.shape[0] // 2
    num_shells = int((im_dim - im_center) / shell_thickness)

    c = np.array([im_center, im_center, im_center])

    r_out = np.arange(
        shell_thickness, (num_shells + 1) * shell_thickness, shell_thickness
    )
    r_in = r_out - 1

    if n_dim == 2:
        image = image[:, :, None]
        c = np.array([im_center, im_center, 0])
    elif n_dim == 1:
        image = image[:, None, None]
        c = np.array([im_center, 0, 0])

    x, y, z = np.meshgrid(
        np.arange(image.shape[0]),
        np.arange(image.shape[1]),
        np.arange(image.shape[2]),
        indexing="ij",
    )
    radial_mask = (
        (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2
        >= r_in[:, None, None, None] ** 2
    ) & (
        (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2
        < r_out[:, None, None, None] ** 2
    )

    radial_reduction = np.array([f(np.squeeze(image[shell])) for shell in radial_mask])
    return radial_reduction[np.isfinite(radial_reduction)], radial_mask


def radial(image, **kwargs):
    return _radial(image, **kwargs)


def write_text(txt):
    stderr.write(txt)


# https://stackoverflow.com/questions/42464334/find-the-intersection-of-two-curves-given-by-x-y-data-with-high-precision-in
def interpolated_intercepts(x, y1, y2):
    """Find the intercepts of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.
        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)
        Returns: the intercept, in (x,y) format
        """

        def line(p1, p2):
            A = p1[1] - p2[1]
            B = p2[0] - p1[0]
            C = p1[0] * p2[1] - p2[0] * p1[1]
            return A, B, -C

        def intersection(L1, L2):
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x, y

        L1 = line([point1[0], point1[1]], [point2[0], point2[1]])
        L2 = line([point3[0], point3[1]], [point4[0], point4[1]])

        R = intersection(L1, L2)

        return R

    idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

    xcs = []
    ycs = []

    for idx in idxs:
        xc, yc = intercept(
            (x[idx], y1[idx]),
            ((x[idx + 1], y1[idx + 1])),
            ((x[idx], y2[idx])),
            ((x[idx + 1], y2[idx + 1])),
        )
        xcs.append(xc)
        ycs.append(yc)
    return np.squeeze(np.array(xcs)), np.squeeze(np.array(ycs))


# from: https://stackoverflow.com/questions/46626267/how-to-generate-a-sphere-in-3d-numpy-array
def sphere_idx(shape, radius, position):
    """Generate an n-dimensional spherical mask."""
    assert len(position) == len(shape)
    n = len(shape)
    position = np.array(position).reshape((-1,) + (1,) * n)
    arr = np.linalg.norm(np.indices(shape) - position, axis=0)
    return arr <= radius


def electron_density_to_dn(map3d_ed, wavelength):
    classical_electron_radius = 2.81794e-15
    return wavelength**2 / (2 * np.pi) * classical_electron_radius * (map3d_ed * 1e30)


def add_water_saxs(img, px_size, distance, wavelength, pulse_energy):
    c = [img.shape[0] / 2, img.shape[1] / 2]
    px, py = np.meshgrid(
        np.linspace(-c[0], c[0], img.shape[0]), np.linspace(-c[1], c[1], img.shape[1])
    )
    r = np.sqrt(px** 2 + py**2)
    rr = np.sqrt((r * px_size) ** 2 + distance**2)
    theta = np.arctan((r * px_size) / distance) / 2

    water_xs = 0.01632  # cm^-1
    sample_thickness = 50e-7 # cm

    pixel_solid_angle = (np.cos(theta) ** 3) * ((px_size / distance) ** 2)
    pol_correction = np.cos(np.arcsin((px * px_size) / rr)) ** 2

    hc = scipy.constants.h * scipy.constants.c
    n_photons = pulse_energy / (hc / wavelength)

    water = (
        np.ones_like(img)
        * sample_thickness
        * pixel_solid_angle
        * pol_correction
        * n_photons
        * water_xs
    )
    return water
