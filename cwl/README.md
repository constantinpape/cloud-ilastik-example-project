# Common Workflow Language usage demo

The Common Workflow Language is a specification for a `.yml` file with the `.cwl` extension that specifies how a task (really, a process) is to be run, by enumerating its inputs and outputs, and potentially connecting the outputs of one step into the inputs of another step, creating workflows. These `.cwl` files can then be run via an implementation of `cwl-runner`, of which there are a few. One can also specify how the environment should look like for a workflow to be runnable such as available environment variables, executables, or even a docker image in which this step/workflow is supposed to run in.

Usually one runs a `.cwl` file like so:

    cwl-runner pixel_classification.cwl pixel_classification_args.yml

`cwl-runner` will then check the `pixel_classification_args.yml` file  to verify that it provides all inputs needed by the `pixel_classification.cwl` workflow.

Alternatively, the `.cwl` files can be executed directly, with its inputs specified in the command line:

    ./pixel_classification.cwl --project /home/tomaz/MyProject.ilp --raw_data /home/tomaz/SampleData/c_cells/cropped/cropped2.png

making it convenient to test and reuse individual steps of a workflow.


## Installing a cwl-runner implementation

You can follow the instructions to install the reference implementation [here](https://github.com/common-workflow-language/cwltool#install) or, if you're in Debian/Ubuntu, just `apt install cwltool`

## Files and what they do:

- `pixel_classification.cwl`

    Represents an execution of pixel classification via ilastik headless; Used as a step in `reference_workflow.cwl`


- `do_threshold.py` and `reference_impl/`

    Code out of constantine's repo. `do_threshold.py` is an executable that runs only the thresholding step of constantin's reference workflow.


- `threshold.cwl`

    Wraps `do_threshold.py` with CWL. Can be executed independently. Used as a step in `reference_workflow.cwl`

- `reference_workflow.cwl`

    Connects `pixel_classification.cwl` and `threshold.cwl` (connected components is still missing) into a single CWL workflow, by making pixel_classification and threshold into steps of the bigger workflow.

    One can run this file with `cwl-runner --cachedir /tmp/my_cache_dir (...)` so that the execution can remember the step outputs and not run everything again on the next execution. Caching is a bit limited on the reference implementation oc `cwl-runner`, though, and I think I'm missing a few more parameters on the `.cwl` files to help the runner know what it is and isn't supposed to cache.
