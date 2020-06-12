from .blocking import Blocking


def threshold_block(input_dataset, output_dataset, threshold,
                    blocking, block_id):
    bounding_box = blocking[block_id]
    data = input_dataset[bounding_box]
    data = (data > threshold).astype('uint8')
    # print(block_id, data.sum())
    output_dataset[bounding_box] = data


def threshold_blocks(input_dataset, output_dataset,
                     threshold, block_shape, block_ids):
    shape = input_dataset.shape
    blocking = Blocking(shape, block_shape)
    for block_id in block_ids:
        threshold_block(input_dataset, output_dataset, threshold,
                        blocking, block_id)
