#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool
baseCommand: python
arguments:
    - /home/tomaz/source/ilastik_master/ilastik.py
    - --headless
    - --output_filename_format=$(runtime.outdir)/predictions.n5
    - --output_format=n5
inputs:
    project:
        type: File
        inputBinding:
            prefix: --project=
            separate: false
    raw_data:
        type: File
        inputBinding:
            prefix: --raw-data=
            separate: false
outputs:
    predictions:
        type: Directory
        outputBinding:
            glob: predictions.n5