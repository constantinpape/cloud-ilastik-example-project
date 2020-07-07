#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: Workflow
inputs:
    project: File
    raw_data: File

outputs:
    thresholds:
        type: Directory
        outputSource: threshold/thresholded_data


steps:
    pixel_classification:
        run: pixel_classification.cwl
        in:
            project: project
            raw_data: raw_data
        out: [predictions]

    threshold:
        run: threshold.cwl
        in:
            in_file_path: pixel_classification/predictions
        out: [thresholded_data]