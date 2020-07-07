from skimage.filters import gaussian
from .blocking import Blocking


def threshold_block(input_dataset, output_dataset,
                    threshold, sigma,
                    blocking, block_id):
    bounding_box = blocking[block_id]
    data = input_dataset[bounding_box]
    data = gaussian(data, sigma)
    data = (data > threshold).astype('uint8')
    output_dataset[bounding_box] = data


def threshold_blocks(input_dataset, output_dataset,
                     threshold, sigma,
                     block_shape, block_ids,
                     func=threshold_block):
    shape = input_dataset.shape
    blocking = Blocking(shape, block_shape)
    for block_id in block_ids:
        func(input_dataset=input_dataset,
             output_dataset=output_dataset,
             threshold=threshold,
             sigma=sigma,
             blocking=blocking,
             block_id=block_id)
