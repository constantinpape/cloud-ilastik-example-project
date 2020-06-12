import numpy as np
from skimage.measure import label
from skimage.segmentation import relabel_sequential

from .blocking import Blocking


#
# Step 1: run connected components in the individual blocks
#


def label_block(input_dataset, output_dataset, blocking, block_id):
    bounding_box = blocking[block_id]
    data = input_dataset[bounding_box]
    # TODO is this the correct way?
    data = label(data, background=1).astype('uint64')
    # offset by the lowest pixel index in this block to gurantee unique ids
    offset = np.prod([bb.start for bb in bounding_box])
    data[data != 0] += offset
    output_dataset[bounding_box] = data


def label_blocks(input_dataset, output_dataset, block_shape, block_ids):
    shape = input_dataset.shape
    blocking = Blocking(shape, block_shape)
    for block_id in block_ids:
        label_block(input_dataset, output_dataset, blocking, block_id)


#
# Step 2: compute the labels to be merged along the block faces
#


#
# Step 3: merge labels via union find
#


#
# Step 4: write the new labels to the volume
#
