import argparse
import os
import subprocess

import h5py
import z5py
from skimage.filters import gaussian
from skimage.measure import label

CHUNKS = (64,) * 3


def predict_ilastik(in_path, out_path, ilastik_bin, ilastik_project):
    in_key = 'volumes/raw'
    out_key = 'boundaries'

    with z5py.File(out_path) as f:
        if out_key in f:
            return

    input_str = f'{in_path}/{in_key}'
    tmp_path = 'tmp.h5'

    cmd = [ilastik_bin, '--headless',
           '--project=%s' % ilastik_project,
           '--output_format=hdf5',
           '--raw_data=%s' % input_str,
           '--output_filename_format=%s' % tmp_path,
           '--readonly']
    subprocess.check_call(cmd)

    with h5py.File(tmp_path, 'r') as f:
        prediction = f['exported_data'][..., 1]

    with z5py.File(out_path) as f:
        ds = f.create_dataset(out_key, shape=prediction.shape, compression='gzip',
                              dtype=prediction.dtype, chunks=CHUNKS)
        ds.n_threads = 4
        ds[:] = prediction

    os.remove(tmp_path)


def threshold(path, threshold, sigma):
    in_key = 'boundaries'
    out_key = 'thresholded'
    with z5py.File(path) as f:

        if out_key in f:
            return

        data = f[in_key][:]
        data = gaussian(data, sigma)
        data = (data > threshold).astype('uint8')

        f.create_dataset(out_key, data=data, chunks=CHUNKS, compression='gzip')


def connected_components(path):
    in_key = 'thresholded'
    out_key = 'connected_components'

    with z5py.File(path) as f:
        data = f[in_key][:]
        print("Run connceted components ...")
        seg = label(data, background=1).astype('uint64')
        f.create_dataset(out_key, data=seg, compression='gzip', chunks=CHUNKS)


def expected_results(in_path, out_path):

    ilastik_bin = '/home/pape/Work/software/src/ilastik-1.4.0b0-Linux/run_ilastik.sh'
    ilastik_project = './data/example-project.ilp'

    predict_ilastik(in_path, out_path, ilastik_bin, ilastik_project)
    threshold(out_path, threshold=0.25, sigma=1.6)
    connected_components(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='data/data_small.n5')
    parser.add_argument('--output_path', default='data/expected_small.n5')

    args = parser.parse_args()
    expected_results(args.input_path, args.output_path)
