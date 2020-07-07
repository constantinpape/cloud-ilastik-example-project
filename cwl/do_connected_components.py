import argparse
from concurrent import futures
from functools import partial

import z5py
import reference_impl as ref


def connected_components_workflow(input_path, input_key,
                                  output_path, output_key,
                                  n_jobs, block_shape):
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
    parser.add_argument('--in-file-path', required=True)
    parser.add_argument('--in-dataset-path', default="exported_data")

    parser.add_argument('--out-file-path', required=True)
    parser.add_argument('--out-dataset-path', required=True)


    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--sigma', type=float, default=1.6)
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args()

    connected_components_workflow(input_path=args.in_file_path, input_key=args.in_dataset_path,
                                  output_path=args.out_file_path, output_key=args.out_dataset_path,
                                  n_jobs=args.n_jobs, block_shape=[64, 64, 64])
