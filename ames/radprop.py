from h5py import File
import numpy as np


def _get_file_info(file: File, field: str) -> np.ndarray:
    return file[field][:]


def get_particle_sizes(file: File) -> np.ndarray:
    return _get_file_info(file, 'particle_sizes')


def get_wavelengths(file: File) -> np.ndarray:
    return _get_file_info(file, 'wavelengths')


def get_scattering_cross_section(file: File) -> np.ndarray:
    return _get_file_info(file, 'scattering_cross_section')


def get_extinction_cross_section(file: File) -> np.ndarray:
    return _get_file_info(file, 'extinction_cross_section')
