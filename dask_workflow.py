import itertools
import subprocess
import tempfile
import uuid

import numpy
import dask
import dask.array
import h5py
import structlog
import z5py
from dask.distributed import Client, wait


log = structlog.get_logger()


def ilastik_predict(block_info=None, *, ilastik_bin, ilastik_project, input_path, input_key, output_path, output_key, channel_id):
    array_location = block_info[None]["array-location"]
    output_idx = tuple(itertools.starmap(slice, array_location))

    start = ",".join([*(str(i.start) for i in output_idx), "None"])
    stop = ",".join([*(str(i.stop) for i in output_idx), "None"])
    cutout_subregion = f"[({start}), ({stop})]"

    with tempfile.NamedTemporaryFile(suffix=".h5") as temp_file:
        log.msg("ilastik_predict", state="prediction", temp_file=temp_file.name)

        subprocess.check_call([
            ilastik_bin, "--headless", "--readonly",
            "--output_format", "hdf5",
            "--project", ilastik_project,
            "--raw_data", f"{input_path}/{input_key}",
            "--cutout_subregion", cutout_subregion,
            "--output_filename_format", temp_file.name,
        ])

        log.msg("ilastik_predict", state="copying")

        output_file = z5py.File(output_path)
        output_group = output_file.require_group(output_key)
        output_dataset = output_group.require_dataset(
            "predictions",
            shape=block_info[None]["shape"],
            dtype=block_info[None]["dtype"],
            chunks=block_info[None]["chunk-shape"],
            compression="gzip",
        )
        output = output_dataset[output_idx]

        with h5py.File(temp_file.name, "r") as temp_h5:
            if channel_id is None:
                output[...] = temp_h5["exported_data"][:]
            else:
                output[...] = temp_h5["exported_data"][..., channel_id]

        log.msg("ilastik_predict", state="finished")

    output_file.close()
    return output


def predictions():
    ilastik_bin = "/Users/em/Downloads/ilastik-1.4.0b7-OSX.app/Contents/ilastik-release/run_ilastik.sh"
    ilastik_project = "data/example-project.ilp"

    input_path = "data/data_small.n5"
    input_key = "volumes/raw"
    output_path = "data/result_small.n5"
    chunks = 64
    output_key = str(uuid.uuid4())

    input = dask.array.from_array(z5py.File(str(input_path), "r")[input_key], chunks=chunks)
    input = input.partitions[:, :, :2]

    return dask.array.map_blocks(
        ilastik_predict,
        ilastik_bin=ilastik_bin,
        ilastik_project=ilastik_project,
        input_path=input_path,
        input_key=input_key,
        output_path=output_path,
        output_key=output_key,
        channel_id=1,
        chunks=input.chunks,
        dtype="float32",
        meta=numpy.empty((0,), dtype="float32"),
    )


def main():
    with Client() as client:
        log.msg("client_started")
        preds = predictions()
        wait(client.persist(preds))

    log.msg("client_stopped")


if __name__ == "__main__":
    main()
