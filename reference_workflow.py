from concurrent import futures
import z5py
import reference_impl as ref


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
        return

    ds_out = f_out.create_dataset(output_key, shape=shape, chunks=tuple(block_shape),
                                  compression='gzip', dtype='float32')

    block_lists = ref.blocks_to_jobs(shape, block_shape, n_jobs)
    channel_id = 1

    with futures.ThreadPoolExecutor(n_jobs) as tp:
        tasks = [tp.submit(ref.predict_blocks, ds_in, ds_out,
                           ilastik_bin, ilastik_project,
                           block_shape, blocks, channel_id=channel_id)
                 for blocks in block_lists]
        [t.result() for t in tasks]


def threshold_workflow(input_path, input_key,
                       output_path, output_key,
                       threshold, n_jobs):
    pass


def connected_components_workflow(input_path, input_key,
                                  output_path, output_key,
                                  n_jobs):
    pass


if __name__ == '__main__':
    input_path = './data/sampleA.n5'
    input_key = 'volumes/raw'

    output_path = './result.n5'
    prediction_key = 'boundaries'
    threshold_key = 'thresholded'
    segmentation_key = 'connected_components'

    ilastik_bin = '/home/pape/Work/software/src/ilastik-1.4.0b0-Linux/run_ilastik.sh'
    ilastik_project = './data/example-project.ilp'
    threshold = .5

    n_jobs = 4
    block_shape = [25, 256, 256]

    prediction_workflow(input_path, input_key,
                        output_path, prediction_key,
                        ilastik_bin, ilastik_project,
                        block_shape, n_jobs)
    threshold_workflow(output_path, prediction_key,
                       output_path, threshold_key,
                       threshold, n_jobs)
    connected_components_workflow(output_path, threshold_key,
                                  output_path, segmentation_key,
                                  n_jobs)
