#/usr/bin/env python3

import argparse
from concurrent import futures
from typing import Tuple
# from functools import partial

import z5py
import reference_impl as ref

def threshold_workflow(
    in_file_path: str,
    in_dataset_path: str,

    output_path: str,
    threshold: float,
    sigma: float,
    n_jobs: int,
    block_shape: Tuple[int, ...]
):
    f_in = z5py.File(in_file_path, 'r')
    ds_in = f_in[in_dataset_path]
    shape = ds_in.shape

    out_file_path = output_path.split(".n5")[0] + ".n5"
    out_dataset_path = output_path.split(".n5")[1]

    f_out = z5py.File(out_file_path, 'a')
    # we assume that if we have the output dataset the task has passed
    if out_dataset_path in f_out:
        print("Have", out_dataset_path, "already, skipping thresholding")
        return

    ds_out = f_out.create_dataset(out_dataset_path, shape=shape, chunks=tuple(block_shape),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file-path', required=True)
    parser.add_argument('--in-dataset-path', default="exported_data")

    parser.add_argument('--output-path', required=True)
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--sigma', type=float, default=1.6)
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args()

    threshold_workflow(in_file_path=args.in_file_path,
                       in_dataset_path=args.in_dataset_path,
                       output_path=args.output_path,
                       threshold=args.threshold, sigma=args.sigma, n_jobs=args.n_jobs,
                       block_shape=[64, 64, 64])