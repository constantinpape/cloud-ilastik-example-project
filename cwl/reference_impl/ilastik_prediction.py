import os
import subprocess

import h5py
from .blocking import Blocking


# NOTE this is not a really efficient way to do things, because the ilastik project is spun
# up separately for each block. But I haven't found a better way to do this for 3d data
# where one only wants to predict sub-blocks yet
def predict_block(input_dataset, output_dataset,
                  ilastik_bin, ilastik_project,
                  blocking, block_id, channel_id):
    bounding_box = blocking[block_id]

    # specify the subregion
    start = [str(bb.start) for bb in bounding_box] + ['None']
    stop = [str(bb.stop) for bb in bounding_box] + ['None']
    start = '(%s)' % ','.join(start)
    stop = '(%s)' % ','.join(stop)
    subregion_str = '[%s, %s]' % (start, stop)

    input_str = '%s/%s' % (input_dataset.file.filename,
                           input_dataset.name)

    output_path = './block%i.h5' % block_id

    cmd = [ilastik_bin, '--headless',
           '--project=%s' % ilastik_project,
           '--output_format=hdf5',
           '--raw_data=%s' % input_str,
           '--cutout_subregion=%s' % subregion_str,
           '--output_filename_format=%s' % output_path,
           '--readonly']
    subprocess.check_call(cmd)

    with h5py.File(output_path, 'r') as f:
        if channel_id is None:
            prediction = f['exported_data'][:]
        else:
            prediction = f['exported_data'][..., channel_id]

    output_dataset[bounding_box] = prediction
    os.remove(output_path)


def predict_blocks(input_dataset, output_dataset,
                   ilastik_bin, ilastik_project,
                   block_shape, block_ids, channel_id):
    shape = input_dataset.shape
    blocking = Blocking(shape, block_shape)
    for block_id in block_ids:
        predict_block(input_dataset, output_dataset,
                      ilastik_bin, ilastik_project,
                      blocking, block_id, channel_id)
