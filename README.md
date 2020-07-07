# Cloud Ilastik Example Project

Reference implementation for the example workflow to evaluate different orchestration frameworks for large scale data processing.
The goal is to implement the following workflow, all tasks being parallelized over sub-blocks of the input volume:
- Run ilastik pixel classification
- Smooth and threshold the pixel classification result
- Run connected components on the thresholding result

The two first tasks can be easily parallelized over blocks; the connected components require a "map-reduce-style" approach to be parallized over blocks. See below for more details on the additional tasks.

Ideally, you should use the functions from `reference_impl/` to run the individual steps; you can check out `reference_workflow.py` to see how to connect the pieces.

Some additional considerations when implementing this:
- The solution should be deployable locally and on the EMBL cluster (additional computations backends would of course be great)
- Robustness to partial (stochastic) failures: you can use the decorators in `reference_impl/job_failure.py` to simulate this.
- Logging and monitoring task execution
- Rerunning tasks if intermediate results change


## Ilastik prediction

Easily parallelisable task: run ilastik pixel classification prediction for all blocks


## Smoothing & Thresholding

Easily parallelisable task: smooth prediction and apply threshold for all blocks


## Connected components

More complicated task:
1. Run connected components for all blocks; add offset to make the resulting ids unique
2. Find all unique ids
3. Iterate over all block faces and determine the labels to be merged
4. Use union find to merge the labels determined in 4
5. Write the new labeling to the blocks


## Data & Validation

I have cloned this repository to `/g/kreshuk/software/cloud-ilastik-example-project`.
The folder `/g/kreshuk/software/cloud-ilastik-example-project/data` contains the example data:
a small (`data_small.n5`) and large (`data.n5`) 3d volume as well as the example ilastik project `example_project.ilp`.
It also contains results from the reference implementation (`result_small.n5` / `result.n5`) and results from the not parallelized implementation (`expected_small.n5` / `expected.n5`).

You can use the script `validate_output.py` to visually and quantitatively compare results. Note that the results of the reference workflow and the non-parallel implementation do not
agree fully because of block boundary artifacts in the filter computation.

## Dagster

Run the example pipeline: `dagster pipeline execute -f dagster_workflow.py -e dagster_config.yaml`
