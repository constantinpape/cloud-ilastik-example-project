import argparse
import tempfile
from concurrent import futures
from functools import partial

import z5py
import reference_impl as ref
from reference_impl import connected_components_new as cc

from prefect import task, Flow, unmapped


@task
def blocks_to_jobs(input_path, input_key, block_shape, n_jobs):
    f_in = z5py.File(input_path, "r")
    ds_in = f_in[input_key]
    shape = ds_in.shape
    return ref.blocks_to_jobs(shape, block_shape, n_jobs)


@task
def predict_blocks(input_path, input_key, output_path, output_key, ilastik_bin, ilastik_project, block_shape, blocks):
    f_in = z5py.File(input_path, "r")
    ds_in = f_in[input_key]
    shape = ds_in.shape

    f_out = z5py.File(output_path, "a")
    ds_out = f_out.require_dataset(
        output_key, shape=shape, chunks=tuple(block_shape), compression="gzip", dtype="float32"
    )
    print("WRITING PREDICTIONS TO", output_path, output_key)

    channel_id = 1
    ref.predict_blocks(ds_in, ds_out, ilastik_bin, ilastik_project, block_shape, blocks, channel_id=channel_id)
    return output_path, output_key


@task
def threshold_blocks(input_path, input_key, output_path, output_key, threshold, sigma, blocks):
    print("INPUT", input_path, input_key)
    f_in = z5py.File(input_path, "r")
    ds_in = f_in[input_key]
    shape = ds_in.shape

    f_out = z5py.File(output_path, "a")
    # we assume that if we have the output dataset the task has passed
    ds_out = f_out.require_dataset(
        output_key, shape=shape, chunks=tuple(block_shape), compression="gzip", dtype="uint8"
    )

    # example of running with stochastic failures
    thres_func = ref.threshold_blocks
    # thres_func = partial(ref.threshold_blocks, func=ref.simple_failure(ref.threshold.threshold_block))
    # thres_func = partial(ref.threshold_blocks, func=ref.failure_with_incorrect_output(ref.threshold.threshold_block))
    # thres_func = partial(ref.threshold_blocks, func=ref.failure_with_corrupted_output(ref.threshold.threshold_block))
    thres_func(ds_in, ds_out, threshold, sigma, block_shape, blocks)
    return output_path, output_key


@task
def cc_create_temp_dir():
    return tempfile.mkdtemp(prefix="connected_components_")


@task
def cc_label_blocks(input_path, input_key, output_path, output_key, tmp_path, block_shape, blocks):
    f_in = z5py.File(input_path, "r")
    ds_in = f_in[input_key]
    shape = ds_in.shape

    f_out = z5py.File(output_path, "a")
    ds_out = f_out.require_dataset(
        output_key, shape=shape, chunks=tuple(block_shape), compression="gzip", dtype="uint64"
    )
    return cc.label_blocks(ds_in, ds_out, block_shape, blocks, tmp_path)


@task
def cc_find_uniques(label_paths, tmp_path):
    return cc.find_uniques(label_paths, tmp_path)


@task
def cc_merge_faces(output_path, output_key, block_shape, blocks, tmp_path, labeled_blocks_paths):
    f_out = z5py.File(output_path, "a")
    ds_out = f_out[output_key]

    return cc.merge_faces(ds_out, block_shape, blocks, tmp_path)


@task
def cc_merge_labels(unique_labels_path, merge_block_paths, tmp_path):
    return cc.merge_labels(unique_labels_path, merge_block_paths, tmp_path)


@task
def cc_write_labels(output_path, output_key, block_shape, blocks, unique_labels_path, merged_labels_path):
    f_out = z5py.File(output_path, "a")
    ds_out = f_out[output_key]

    return cc.write_labels(ds_out, ds_out, unique_labels_path, merged_labels_path, block_shape, blocks)


@task
def cc_remove_temp_dir(tmp_path, **kwargs):
    print("TODO REMOVE", tmp_path, kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="data/data_small.n5")
    parser.add_argument("--output_path", default="data/prefect_result_small.n5")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    input_key = "volumes/raw"

    prediction_key = "boundaries"
    threshold_key = "thresholded"
    segmentation_key = "connected_components"

    ilastik_bin = "./ilastik-1.3.3post3-Linux/run_ilastik.sh"
    ilastik_project = "./data/example-project.ilp"
    threshold = 0.25
    sigma = 1.6

    n_jobs = 4
    block_shape = [64, 64, 64]

    with Flow("Hackathon workflow") as flow:
        blocks = blocks_to_jobs(input_path, input_key, block_shape, n_jobs)
        predict_blocks_task = predict_blocks.map(
            input_path=unmapped(input_path),
            input_key=unmapped(input_key),
            output_path=unmapped(output_path),
            output_key=unmapped(prediction_key),
            ilastik_bin=unmapped(ilastik_bin),
            ilastik_project=unmapped(ilastik_project),
            block_shape=unmapped(block_shape),
            blocks=blocks,
        )

        threshold_blocks_task = threshold_blocks.map(
            input_path=unmapped(output_path),
            input_key=unmapped(prediction_key),
            output_path=unmapped(output_path),
            output_key=unmapped(threshold_key),
            threshold=unmapped(threshold),
            sigma=unmapped(sigma),
            blocks=blocks,
        )
        flow.set_dependencies(task=threshold_blocks_task, upstream_tasks=[predict_blocks_task])

        cc_temp = cc_create_temp_dir()
        flow.set_dependencies(task=cc_temp, upstream_tasks=[threshold_blocks_task])

        labeled_blocks_paths = cc_label_blocks.map(
            input_path=unmapped(output_path),
            input_key=unmapped(threshold_key),
            output_path=unmapped(output_path),
            output_key=unmapped(segmentation_key),
            tmp_path=unmapped(cc_temp),
            blocks=blocks,
            block_shape=unmapped(block_shape),
        )
        unique_path = cc_find_uniques(labeled_blocks_paths, cc_temp)
        merged_blocks_paths = cc_merge_faces.map(
            output_path=unmapped(output_path),
            output_key=unmapped(segmentation_key),
            tmp_path=unmapped(cc_temp),
            blocks=blocks,
            block_shape=unmapped(block_shape),
            labeled_blocks_paths=labeled_blocks_paths,
        )
        merged_labels_path = cc_merge_labels(unique_path, merged_blocks_paths, cc_temp)
        write_labels = cc_write_labels.map(
            output_path=unmapped(output_path),
            output_key=unmapped(segmentation_key),
            blocks=blocks,
            block_shape=unmapped(block_shape),
            unique_labels_path=unmapped(unique_path),
            merged_labels_path=unmapped(merged_labels_path),
        )
        cc_remove_temp_dir(cc_temp, res=write_labels)
        flow.run()
