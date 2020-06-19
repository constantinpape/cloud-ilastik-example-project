# Cloud Ilastik Example Project

Reference implementation for the example workflow to evaluate different orchestration frameworks for large scale data processing.
The goail is to implement the following workflow:
- Run ilastik pixel classification
- Threshold the pixel classification result
- Run connected components on the thresholding result

The two first tasks can just be parallelized over blocks; the connected components require a more "map-reduce" style approach to be parallized over blocks. See below for more details on the additional tasks.

Ideally, you should use the functions from `reference_impl/` to run the individual steps; you can check out `reference_workflow.py` to see how to connect the pieces.

Some additional considerations when implementing this:
- The solution should be deployable locally and on the EMBL cluster, additional options would of course be great
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
