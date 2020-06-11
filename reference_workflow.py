from concurrent import futures
import reference_impl as ref


def prediction_workflow(input_path, input_key,
                        output_path, output_key,
                        ilastik_bin, ilastik_project,
                        block_shape, n_jobs):
    pass


def threshold_workflow(input_path, input_key,
                       output_path, output_key,
                       threshold, n_jobs):
    pass


def connected_components_workflow(input_path, input_key,
                                  output_path, output_key,
                                  n_jobs):
    pass


if __name__ == '__main__':
    input_path = ''
    input_key = ''

    output_path = ''
    prediction_key = ''
    threshold_key = ''
    segmentation_key = ''

    ilastik_bin = ''
    ilastik_project = ''
    threshold = .5

    n_jobs = 4
    block_shape = []

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
