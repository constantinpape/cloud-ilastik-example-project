import os
from random import random
import numpy as np


def simple_failure(func, error_rate=0.05):
    """ Stochastic failure with RuntimeError; output is not written
    """
    def wrapper(*args, **kwargs):
        if random() < error_rate:
            raise RuntimeError("Simple failure")
        else:
            return func(*args, **kwargs)
    return wrapper


def failure_with_incorrect_output(func, error_rate=0.05):
    """ Stochastic failure with RuntimeError; output is written but incorrect

    Only works if the following keyword arguments are passed to func:
    output_dataset, blocking, block_id
    """
    def wrapper(*args, **kwargs):
        if random() < error_rate:
            output_dataset = kwargs['output_dataset']
            blocking = kwargs['blocking']
            block_id = kwargs['block_id']

            bounding_box = blocking[block_id]
            shape = tuple(bb.stop - bb.start for bb in bounding_box)
            dtype = np.dtype(output_dataset)

            if np.issubdtype(dtype, np.integer):
                data = np.random.randint(0, 100, dtype=dtype, size=shape)
            else:
                data = np.random.rand(*shape)

            output_dataset[bounding_box] = data
            raise RuntimeError("Failure with incorrect output")
        else:
            return func(*args, **kwargs)
    return wrapper


def failure_with_corrupted_output(func, error_rate=0.5):
    def wrapper(*args, **kwargs):
        if random() < error_rate:
            output_dataset = kwargs['output_dataset']
            blocking = kwargs['blocking']
            block_id = kwargs['block_id']

            bounding_box = blocking[block_id]
            chunks = output_dataset.chunks

            # find the lowest chunk id in this bounding box and the corresponding path on the file sysyem
            chunk_id = [str(bb.start // chunk) for bb, chunk in zip(bounding_box, chunks)]
            ds_path = os.path.join(output_dataset.file.filename, output_dataset.name.lstrip('/'))
            chunk_path = os.path.join(ds_path, '/'.join(chunk_id[::-1]))
            chunk_dir = os.path.split(chunk_path)[0]
            os.makedirs(chunk_dir, exist_ok=True)

            # write some random bytes to this chunk
            with open(chunk_path, 'wb') as f:
                f.write(np.random.bytes(10))

            raise RuntimeError("Failure with corrupted output")
        else:
            return func(*args, **kwargs)
    return wrapper
