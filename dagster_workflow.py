import concurrent.futures

import z5py
from dagster import Field, String, pipeline, solid, InputDefinition, Materialization, EventMetadataEntry, Array, Int, Output

import reference_impl as ref


@solid(
    config_schema={
        "ilastik_bin": Field(String, description="Path to the ilastik executable"),
        "block_shape": Field(Array(Int), description="Shape of a single dataset block"),
        "n_jobs": Field(Int, description="Number of jobs to start"),
        "output_key_suffix": Field(String, default_value="boundaries", is_required=False, description="Output dataset suffix"),
    },
    input_defs=[
        InputDefinition("input_path", String, description="Path to the input dataset file/directory"),
        InputDefinition("input_key", String, description="Path to the array inside the input dataset file/directory"),
        InputDefinition("output_path", String, description="Path to the output dataset file/directory"),
        InputDefinition("ilastik_project", String, description="Path to the ilastik project file/directory"),
    ],
)
def prediction(context, input_path, input_key, output_path, ilastik_project):
    """Predict pixels using ilastik project file."""
    cfg = context.solid_config

    input_file = z5py.File(input_path, "r")
    input_ = input_file[input_key]

    output_file = z5py.File(output_path, "a")
    output_key = "/".join((context.run_id, cfg["output_key_suffix"]))
    output = output_file.create_dataset(
        output_key,
        shape=input_.shape,
        chunks=cfg["block_shape"],
        compression="gzip",
        dtype="float32",
    )

    with concurrent.futures.ThreadPoolExecutor(cfg["n_jobs"]) as executor:
        futures = {
            executor.submit(
                ref.predict_blocks,
                input_,
                output,
                cfg["ilastik_bin"],
                ilastik_project,
                cfg["block_shape"],
                blocks,
                channel_id=1,
            ): blocks[0]
            for blocks in ref.blocks_to_jobs(input_.shape, cfg["block_shape"], cfg["n_jobs"])
        }
        for future in concurrent.futures.as_completed(futures):
            context.log.info(f"completed prediction for block {futures[future]}")

    yield Materialization(label="prediction", description="Prediction output", metadata_entries=[
        EventMetadataEntry.path(output_path, "output_path"),
        EventMetadataEntry.text(output_key, "output_key"),
    ])
    yield Output(output_key)


@pipeline
def workflow():
    prediction()
