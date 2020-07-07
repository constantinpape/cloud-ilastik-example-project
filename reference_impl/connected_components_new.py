import os
import uuid
from shutil import rmtree

import numpy as np
from skimage.measure import label
from skimage.segmentation import relabel_sequential

from .blocking import Blocking
from .union_find import UnionFind


def set_up():
    return os.makedirs("tmp_cc", exist_ok=True)


def clean_up():
    try:
        rmtree("tmp_cc")
    except OSError:
        pass


#
# Step 1: run connected components in the individual blocks
#


def label_block(input_dataset, output_dataset, blocking, block_id):
    bounding_box = blocking[block_id]
    data = input_dataset[bounding_box]
    data = label(data, background=1).astype("uint64")

    # offset by the number of pixels in the block * the block_id
    offset = np.prod(list(blocking.block_shape)) * block_id
    data[data != 0] += offset
    output_dataset[bounding_box] = data

    return np.unique(data)


def label_blocks(input_dataset, output_dataset, block_shape, block_ids, tmp_path):
    shape = input_dataset.shape
    blocking = Blocking(shape, block_shape)

    uniques = []
    for block_id in block_ids:
        uniques_block = label_block(input_dataset, output_dataset, blocking, block_id)
        uniques.append(uniques_block)
    uniques = np.concatenate(uniques)
    uniques = np.unique(uniques)

    out_path = os.path.join(tmp_path, f"uniques_patrial_{uuid.uuid4().hex}.npy")
    np.save(out_path, uniques)
    return out_path


def find_uniques(label_paths, tmp_path):
    uniques = []
    for label_path in label_paths:
        uniques_job = np.load(label_path)
        uniques.append(uniques_job)

    uniques = np.concatenate(uniques)
    uniques = np.unique(uniques)

    out_path = os.path.join(tmp_path, f"uniques.npy")
    np.save(out_path, uniques)
    return out_path


#
# Step 2: compute the labels to be merged along the block faces
#


def merge_face(face_a, face_b, dataset):
    labels_a = dataset[face_a]
    labels_b = dataset[face_b]
    assert labels_a.shape == labels_b.shape

    mask = np.logical_and(labels_a != 0, labels_b != 0)

    labels_a = labels_a[mask]
    labels_b = labels_b[mask]

    merge_labels = np.concatenate([labels_a[:, None], labels_b[:, None]], axis=1)
    merge_labels = np.unique(merge_labels, axis=0)
    return merge_labels


def merge_faces_for_block(dataset, blocking, block_id):
    bounding_box = blocking[block_id]

    ndim = 3
    merge_labels = []
    for axis in range(ndim):
        pos = bounding_box[axis].start
        if pos == 0:
            continue

        this_face = tuple(slice(pos, pos + 1) if ii == axis else slice(None) for ii in range(ndim))
        ngb_face = tuple(slice(pos - 1, pos) if ii == axis else slice(None) for ii in range(ndim))

        merge_for_face = merge_face(this_face, ngb_face, dataset)
        merge_labels.append(merge_for_face)

    if len(merge_labels) > 0:
        return np.concatenate(merge_labels)
    else:
        return None


def merge_faces(dataset, block_shape, block_ids, tmp_path):
    shape = dataset.shape
    blocking = Blocking(shape, block_shape)

    merge_labels = []
    for block_id in block_ids:
        merge_for_block = merge_faces_for_block(dataset, blocking, block_id)
        if merge_for_block is not None:
            merge_labels.append(merge_for_block)

    merge_labels = np.concatenate(merge_labels)
    merge_labels = np.unique(merge_labels, axis=0)
    out_path = os.path.join(tmp_path, f"merge_block_part_{uuid.uuid4().hex}.npy")
    np.save(out_path, merge_labels)
    return out_path


#
# Step 3: merge labels via union find
#


def merge_labels(unique_labels_path, merge_block_paths, tmp_path):
    labels = np.load(unique_labels_path)

    to_merge = []
    for path in merge_block_paths:
        merge_job = np.load(path)
        to_merge.append(merge_job)

    to_merge = np.concatenate(to_merge, axis=0)
    to_merge = np.unique(to_merge, axis=0)

    ufd = UnionFind()
    for label_id in labels:
        ufd.make_set(label_id)

    for label_a, label_b in to_merge:
        ufd.union(label_a, label_b)

    label_assignment = np.array([ufd.find(label) for label in labels], dtype="uint64")
    label_assignment, _, _ = relabel_sequential(label_assignment)
    out_path = os.path.join(tmp_path, f"label_assignment.npy")
    np.save(out_path, label_assignment)
    return out_path

#
# Step 4: write the new labels to the volume
#


def apply_dict_to_array(array, dict_):
    elements, inv = np.unique(array, return_inverse=True)
    return np.array([dict_[elem] for elem in elements])[inv].reshape(array.shape)


def write_labels(input_dataset, output_dataset, unique_labels_path, merged_labels_path, block_shape, block_ids):
    shape = input_dataset.shape
    blocking = Blocking(shape, block_shape)

    labels = np.load(unique_labels_path)
    assignments = np.load(merged_labels_path)
    assert len(labels) == len(assignments)
    assignments = {label_id: assignmnet for label_id, assignmnet in zip(labels, assignments)}

    for block_id in block_ids:
        bounding_box = blocking[block_id]
        data = input_dataset[bounding_box]
        data = apply_dict_to_array(data, assignments)
        output_dataset[bounding_box] = data
