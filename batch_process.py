from h5py import File
from netCDF4 import Dataset
import numpy as np

import ames

if __name__ == '__main__':
    # These are simulation 1
    '''grid = Dataset('/home/kyle/iuvs/ames/sim1/10000.fixed.nc')
    gcm = Dataset('/home/kyle/iuvs/ames/sim1/c48_big.atmos_diurn_plev-002.nc')
    yearly_gcm = Dataset('/home/kyle/iuvs/ames/sim1/c48_big.atmos_average_plev-001.nc')'''

    gcm = Dataset('/home/kyle/iuvs/ames/03704.atmos_diurn.nc')
    dust_radprop = File('/home/kyle/iuvs/mars_dust_v01.hdf5')
    ice_radprop = File('/home/kyle/iuvs/mars_water-ice_v01.hdf5')

    file = File('/home/kyle/iuvs/ames/simulation2.hdf5', mode='a')

    model_grid = file.create_group('grid')
    model_grid.create_dataset('latitude_centers', data=gcm['lat'][:])
    model_grid.create_dataset('latitude_edges', data=np.linspace(-90, 90, num=gcm['lat'][:].shape[0]+1))
    model_grid.create_dataset('longitude_centers', data=gcm['lon'][:])
    model_grid.create_dataset('longitude_edges', data=np.linspace(0, 360, num=gcm['lon'][:].shape[0]+1))
    model_grid.create_dataset('sol_centers', data=ames.time2sol(gcm['time'][:]))
    model_grid.create_dataset('sol_edges', data=np.unique(gcm['time_bnds'][:]))
    model_grid.create_dataset('local_time_centers', data=gcm['time_of_day_24'][:])
    model_grid.create_dataset('local_time_edges', data=gcm['time_of_day_edges_24'][:])

    pressure_comment = 'The pressure is the outer product of the surface pressure * bk + ak. This 5D array is about 500 MB' \
                       'with 2x2 resolution, 24 local times, and 4 sols. It would exceed my RAM if the resolution and range' \
                       'of these outputs get much larger, so I compute it on the fly.'

    bk = model_grid.create_dataset('bk', data=gcm['bk'][:])
    bk.attrs['unit'] = 'Unitless'
    bk.attrs['comment'] = pressure_comment
    ak = model_grid.create_dataset('ak', data=gcm['pk'][:])   # John said he was giving me ak but instead he gave me pk and said it's the same thing
    ak.attrs['unit'] = 'Pa'
    ak.attrs['comment'] = pressure_comment

    time = file.create_group('time')
    time.create_dataset('solar_longitude', data=ames.areo2ls(gcm['areo'][:, :, 0]))

    surface = file.create_group('surface')
    surface_temperature = surface.create_dataset('temperature', data=gcm['ts'], compression=ames.compression, compression_opts=ames.compression_opts)
    surface_temperature.attrs['unit'] = 'K'
    surface_pressure = surface.create_dataset('pressure', data=gcm['ps'])
    surface_pressure.attrs['unit'] = 'Pa'

    atmosphere = file.create_group('atmosphere')
    atmosphere_temperature = atmosphere.create_dataset('temperature', data=gcm['temp'], compression=ames.compression, compression_opts=ames.compression_opts)
    atmosphere_temperature.attrs['unit'] = 'K'

    ice_od = atmosphere.create_dataset('uv_ice_optical_depth', data=ames.compute_ice_optical_depth(gcm, ice_radprop, 0.25), compression=ames.compression, compression_opts=ames.compression_opts)
    ice_od.attrs['comment'] = 'Total column integrated optical depth'
    dust_od = atmosphere.create_dataset('uv_dust_optical_depth', data=ames.compute_ice_optical_depth(gcm, ice_radprop, 0.25), compression=ames.compression, compression_opts=ames.compression_opts)
    dust_od.attrs['comment'] = 'Total column integrated optical depth'

    file.close()
