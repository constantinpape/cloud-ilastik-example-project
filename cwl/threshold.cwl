#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool
baseCommand: python
arguments: 
    - /home/tomaz/source/cwl_tests/do_threshold.py
    - --output-path=$(runtime.outdir)/thresholded_data.n5/data
inputs:
    in_file_path:
        type: Directory
        inputBinding:
            prefix: --in-file-path
#    in_dataset_path:
#        type: string
#        inputBinding:
#            prefix: --in-dataset-path

temporaryFailCodes: [1]

outputs:
    thresholded_data:
        type: Directory
        outputBinding:
            glob: thresholded_data.n5