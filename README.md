# Cloud Ilastik Example Project

Reference implementation for the example workflow to evaluate different orchestration frameworks for large scale data processing.
The goail is to implement the following workflow:
- Run ilastik pixel classification
- Threshold the pixel classification result
- Run connected components on the thresholding result

The two first tasks can just be parallelized over blocks; the connected components require a more "map-reduce" style approach to be parallized over blocks. See below for more details on the additional tasks.

Ideally, you should use the functions from `reference_impl/` to run the individual steps; you can check out `reference_workflow.py` to see how to connect the pieces.

Some additional considerations when implementing this:
- Robustness to partial (stochastic) failures: you can use the decorators in `reference_impl/job_failure.py` to simulate this.
- Logging and monitoring task execution
- Rerunning tasks if intermediate results change


## Ilastik prediction

## Thresholding

## Connected components
