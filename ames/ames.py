from h5py import File
from netCDF4 import Dataset
import numpy as np

from .radprop import get_particle_sizes, get_wavelengths, get_scattering_cross_section, get_extinction_cross_section


def areo2ls(areo: np.ndarray) -> np.ndarray:
    """Convert 'areo' to a solar longitude

    Parameters
    ----------
    areo
        The areo structure of the GCM.

    Returns
    -------
    The solar longitude of each bin.

    """
    return np.mod(np.squeeze(areo), 360)


def compute_pressure(surface_pressure: np.ndarray, ak: np.ndarray, bk: np.ndarray) -> np.ndarray:
    pressure = np.multiply.outer(surface_pressure, bk) + ak   # This will put the vertical axis at the end of the array
    return np.moveaxis(pressure, -1, 2)  # this will make an array with shape (sol, LT, vertical, lat, lon)


def get_closest_index(array: np.ndarray, value: float) -> int:
    return np.abs(array - value).argmin()


def scale_optical_depth(
        simulation_optical_depth: np.ndarray,
        simulation_particle_sizes: np.ndarray,
        simulation_reference_wavelength: float,
        radprop_extinction_cross_section: np.ndarray,
        radprop_particle_sizes: np.ndarray,
        radprop_wavelengths: np.ndarray,
        target_wavelength: float) -> np.ndarray:

    target_wavelength_index = get_closest_index(radprop_wavelengths, target_wavelength)
    reference_wavelength_index = get_closest_index(radprop_wavelengths, simulation_reference_wavelength)

    target_cext = radprop_extinction_cross_section[:, target_wavelength_index]
    reference_cext = radprop_extinction_cross_section[:, reference_wavelength_index]

    scaling_factor = np.interp(simulation_particle_sizes, radprop_particle_sizes, target_cext / reference_cext)
    return simulation_optical_depth * scaling_factor


def compute_optical_depth(simulation_pressure: np.ndarray, simulation_optical_depth: np.ndarray) -> np.ndarray:
    p_diff = np.diff(simulation_pressure, axis=2)
    g = 3.72076
    return np.sum(simulation_optical_depth * p_diff / g, axis=2)


def compute_dust_optical_depth(simulation: Dataset, radprop: File, target_wavelength: float) -> np.ndarray:
    simulation_extinction_optical_depth = simulation['dustref'][:]
    simulation_particle_sizes = np.ones(simulation_extinction_optical_depth.shape) * 1.2
    reference_wavelength = 0.690

    surface_pressure = simulation['ps'][:]
    ak = simulation['pk'][:]
    bk = simulation['bk'][:]

    radprop_extinction_cross_section = get_extinction_cross_section(radprop)
    radprop_particle_sizes = get_particle_sizes(radprop)
    radprop_wavelengths = get_wavelengths(radprop)

    scaled_optical_depth = scale_optical_depth(
        simulation_extinction_optical_depth, simulation_particle_sizes, reference_wavelength,
        radprop_extinction_cross_section, radprop_particle_sizes, radprop_wavelengths, target_wavelength)
    pressure = compute_pressure(surface_pressure, ak, bk)
    return compute_optical_depth(pressure, scaled_optical_depth)


def compute_ice_optical_depth(simulation: Dataset, radprop: File, target_wavelength: float) -> np.ndarray:
    simulation_absorption_optical_depth = simulation['cldref'][:]
    simulation_particle_sizes = np.ones(simulation_absorption_optical_depth.shape) * 1.5
    reference_wavelength = 12

    surface_pressure = simulation['ps'][:]
    ak = simulation['pk'][:]
    bk = simulation['bk'][:]

    radprop_extinction_cross_section = get_extinction_cross_section(radprop)
    radprop_scattering_cross_section = get_scattering_cross_section(radprop)
    radprop_particle_sizes = get_particle_sizes(radprop)
    radprop_wavelengths = get_wavelengths(radprop)

    factor = radprop_extinction_cross_section[:, reference_wavelength] / \
          (radprop_extinction_cross_section[:, reference_wavelength] - radprop_scattering_cross_section[:, reference_wavelength])
    absorption_to_extinction_factor = np.interp(simulation_particle_sizes, radprop_particle_sizes, factor)
    simulation_extinction_optical_depth = simulation_absorption_optical_depth * absorption_to_extinction_factor

    scaled_optical_depth = scale_optical_depth(
        simulation_extinction_optical_depth, simulation_particle_sizes, reference_wavelength,
        radprop_extinction_cross_section, radprop_particle_sizes, radprop_wavelengths, target_wavelength)
    pressure = compute_pressure(surface_pressure, ak, bk)
    return compute_optical_depth(pressure, scaled_optical_depth)


def time2sol(time: np.ndarray) -> np.ndarray:
    return np.mod(time, 668)
