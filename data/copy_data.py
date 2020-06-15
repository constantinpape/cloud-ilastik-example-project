import h5py
import z5py

path = '/home/pape/Work/data/mmwc/knott_data.h5'

with h5py.File(path, 'r') as f:
    data = f['raw'][:]


with z5py.File('./data.n5') as f:
    ds = f.create_dataset('volumes/raw', shape=data.shape, compression='gzip', chunks=(64, 64, 64),
                          dtype=data.dtype)
    ds.n_threads = 4
    ds[:] = data


data = data[:256, :256, :256]
with z5py.File('./data_small.n5') as f:
    ds = f.create_dataset('volumes/raw', shape=data.shape, compression='gzip', chunks=(64, 64, 64),
                          dtype=data.dtype)
    ds.n_threads = 4
    ds[:] = data
