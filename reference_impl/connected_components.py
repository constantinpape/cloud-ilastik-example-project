import numpy as np
from skimage.measure import label
from skimage.segmentation import relabel_sequential

from .blocking import Blocking
from .union_find import UnionFind

# TODO use varlen dataset for temporary files!


#
# Step 1: run connected components in the individual blocks
#

def label_block(input_dataset, output_dataset, blocking, block_id):
    bounding_box = blocking[block_id]
    data = input_dataset[bounding_box]
    # TODO is this the correct way?
    data = label(data, background=1).astype('uint64')

    # offset by the lowest pixel index in this block to gurantee unique ids
    offset = np.prod([bb.start for bb in bounding_box])
    data[data != 0] += offset
    output_dataset[bounding_box] = data

    return np.unique(data)


def label_blocks(input_dataset, output_dataset, block_shape, block_ids, job_id):
    shape = input_dataset.shape
    blocking = Blocking(shape, block_shape)

    uniques = []
    for block_id in block_ids:
        uniques_block = label_block(input_dataset, output_dataset, blocking, block_id)
        uniques.append(uniques_block)
    uniques = np.concatenate(uniques)
    uniques = np.unique(uniques)

    np.save(f'uniques_{job_id}.npy', uniques)


def find_uniques(n_jobs):
    uniques = []
    for job_id in range(n_jobs):
        path = f'uniques_{job_id}.npy'
        uniques_job = np.load(path)
        uniques.append(uniques_job)

    uniques = np.concatenate(uniques)
    uniques = np.unique(uniques)
    np.save('uniques.npy', uniques)


#
# Step 2: compute the labels to be merged along the block faces
#

def merge_face(face_a, face_b, dataset):
    labels_a = dataset[face_a]
    labels_b = dataset[face_b]
    assert labels_a.shape == labels_b.shape

    mask = np.logical_and(labels_a != 0, labels_b != 0)

    labels_a = labels_a[mask]
    labels_b = labels_a[mask]

    merge_labels = np.concatenate([labels_a[:, None], labels_b[:, None]],
                                  axis=1)
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

        this_face = tuple(slice(pos, pos + 1) if ii == axis else slice(None)
                          for ii in range(ndim))
        ngb_face = tuple(slice(pos - 1, pos) if ii == axis else slice(None)
                         for ii in range(ndim))

        merge_for_face = merge_face(this_face, ngb_face, dataset)
        merge_labels.append(merge_for_face)

    if len(merge_labels) > 0:
        return np.concatenate(merge_labels)
    else:
        return None


def merge_faces(dataset, block_shape, block_ids, job_id):
    shape = dataset.shape
    blocking = Blocking(shape, block_shape)

    merge_labels = []
    for block_id in block_ids:
        merge_for_block = merge_faces_for_block(dataset, blocking, block_id)
        if merge_for_block is not None:
            merge_labels.append(merge_for_block)

    merge_labels = np.concatenate(merge_labels)
    merge_labels = np.unique(merge_labels, axis=0)
    out_path = f'./merge_block{job_id}.npy'
    np.save(out_path, merge_labels)


#
# Step 3: merge labels via union find
#

def merge_labels(n_jobs):
    labels = np.load('uniques.npy')

    to_merge = []
    for job_id in range(n_jobs):
        path = f'./merge_block{job_id}.npy'
        merge_job = np.load(path)
        to_merge.append(merge_job)
        # os.remove(path)

    to_merge = np.unique(to_merge, axis=0)
    labels = np.unique(to_merge)

    ufd = UnionFind()
    for label_id in labels:
        ufd.make_set(label_id)

    for label_a, label_b in to_merge:
        ufd.union(label_a, label_b)

    label_assignment = np.array([ufd.find(label) for label in labels], dtype='uint64')
    label_assignment, _, _ = relabel_sequential(label_assignment)
    np.save('label_assignment.npy', label_assignment)


#
# Step 4: write the new labels to the volume
#

def write_labels(dataset, block_shape, block_ids):
    shape = dataset.shape
    blocking = Blocking(shape, block_shape)

    labels = np.load('uniques.npy')
    assignments = np.load('label_assignments.npy')
    assert len(labels) == len(assignments)
    assignments = {label_id: assignmnet for label_id, assignmnet
                   in zip(labels, assignments)}

    for block_id in block_ids:
        bounding_box = blocking(block_ids)
        data = dataset[bounding_box]
        # TODO
        data = ''
        dataset[bounding_box] = data


# TODO
def clean_up():
    pass
    # os.remove('uniques.npy')
    # os.remove('label_assignments.npy')
