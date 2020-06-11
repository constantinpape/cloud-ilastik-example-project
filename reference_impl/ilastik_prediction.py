import subprocess
from .blocking import blocking


def predict_block(input_dataset, output_dataset,
                  ilastik_bin, ilastik_project,
                  blocking, block_id):
    pass


def predict_blocks(input_dataset, output_dataset,
                   ilastik_bin, ilastik_project,
                   block_shape, block_ids):
    shape = input_dataset.shape
    blocking = blocking(shape, block_shape)
    for block_id in block_ids:
        predict_block(input_dataset, output_dataset, blocking, block_id)
