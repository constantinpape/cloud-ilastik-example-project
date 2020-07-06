import argparse
from concurrent import futures
from functools import partial

import z5py
import reference_impl as ref
import env


def prediction_workflow(input_path, input_key,
                        output_path, output_key,
                        ilastik_bin, ilastik_project,
                        block_shape, n_jobs):
    f_in = z5py.File(input_path, 'r')
    ds_in = f_in[input_key]
    shape = ds_in.shape

    f_out = z5py.File(output_path, 'a')
    # we assume that if we have the output dataset the task has passed
    if output_key in f_out:
        print("Have", output_key, "already, skipping prediction")
        return

    ds_out = f_out.create_dataset(output_key, shape=shape, chunks=tuple(block_shape),
                                  compression='gzip', dtype='float32')

    block_lists = ref.blocks_to_jobs(shape, block_shape, n_jobs)
    channel_id = 1

    with futures.ThreadPoolExecutor(n_jobs) as tp:
        tasks = [tp.submit(ref.predict_blocks,
                           ds_in, ds_out,
                           ilastik_bin, ilastik_project,
                           block_shape, blocks, channel_id=channel_id)
                 for blocks in block_lists]
        [t.result() for t in tasks]


def threshold_workflow(input_path, input_key,
                       output_path, output_key,
                       threshold, sigma, n_jobs):
    f_in = z5py.File(input_path, 'r')
    ds_in = f_in[input_key]
    shape = ds_in.shape

    f_out = z5py.File(output_path, 'a')
    # we assume that if we have the output dataset the task has passed
    if output_key in f_out:
        print("Have", output_key, "already, skipping thresholding")
        return

    ds_out = f_out.create_dataset(output_key, shape=shape, chunks=tuple(block_shape),
                                  compression='gzip', dtype='uint8')

    block_lists = ref.blocks_to_jobs(shape, block_shape, n_jobs)

    # example of running with stochastic failures
    func = ref.threshold_blocks
    # func = partial(ref.threshold_blocks, func=ref.simple_failure(ref.threshold.threshold_block))
    # func = partial(ref.threshold_blocks, func=ref.failure_with_incorrect_output(ref.threshold.threshold_block))
    # func = partial(ref.threshold_blocks, func=ref.failure_with_corrupted_output(ref.threshold.threshold_block))

    with futures.ThreadPoolExecutor(n_jobs) as tp:
        tasks = [tp.submit(func,
                           ds_in, ds_out,
                           threshold, sigma,
                           block_shape, blocks)
                 for blocks in block_lists]
        [t.result() for t in tasks]


def connected_components_workflow(input_path, input_key,
                                  output_path, output_key,
                                  n_jobs):
    ref.set_up()

    f_in = z5py.File(input_path, 'r')
    ds_in = f_in[input_key]
    shape = ds_in.shape

    f_out = z5py.File(output_path, 'a')

    ds_out = f_out.require_dataset(output_key, shape=shape, chunks=tuple(block_shape),
                                   compression='gzip', dtype='uint64')
    block_lists = ref.blocks_to_jobs(shape, block_shape, n_jobs)

    with futures.ThreadPoolExecutor(n_jobs) as tp:
        tasks = [tp.submit(ref.label_blocks, ds_in, ds_out, block_shape, blocks, job_id)
                 for job_id, blocks in enumerate(block_lists)]
        [t.result() for t in tasks]

    ref.find_uniques(n_jobs)

    ref.merge_faces(ds_out, block_shape, block_lists[0], 0)
    with futures.ThreadPoolExecutor(n_jobs) as tp:
        tasks = [tp.submit(ref.merge_faces, ds_out, block_shape, blocks, job_id)
                 for job_id, blocks in enumerate(block_lists)]
        [t.result() for t in tasks]

    ref.merge_labels(n_jobs)

    with futures.ThreadPoolExecutor(n_jobs) as tp:
        tasks = [tp.submit(ref.write_labels, ds_out, ds_out, block_shape, blocks)
                 for blocks in block_lists]
        [t.result() for t in tasks]

    ref.clean_up()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='data/data_small.n5')
    parser.add_argument('--output_path', default='data/result_small.n5')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    input_key = 'volumes/raw'

    prediction_key = 'boundaries'
    threshold_key = 'thresholded'
    segmentation_key = 'connected_components'

    ilastik_bin = env.ilastik_bin
    ilastik_project = './data/example-project.ilp'
    threshold = .25
    sigma = 1.6

    n_jobs = 4
    block_shape = [64, 64, 64]

    prediction_workflow(input_path, input_key,
                        output_path, prediction_key,
                        ilastik_bin, ilastik_project,
                        block_shape, n_jobs)
    threshold_workflow(output_path, prediction_key,
                       output_path, threshold_key,
                       threshold, sigma, n_jobs)
    connected_components_workflow(output_path, threshold_key,
                                  output_path, segmentation_key,
                                  n_jobs)
