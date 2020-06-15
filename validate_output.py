import argparse
import z5py
from skimage.metrics import adapted_rand_error


def show(in_path, res_path):
    import napari

    with z5py.File(in_path, 'r') as f:
        ds = f['volumes/raw']
        ds.n_threads = 4
        raw = ds[:]

    layers = {'raw': (raw, 'image', {})}

    prediction_key = 'boundaries'
    threshold_key = 'thresholded'
    segmentation_key = 'connected_components'
    with z5py.File(res_path, 'r') as f:
        if prediction_key in f:
            ds = f[prediction_key]
            ds.n_threads = 4
            data = ds[:]
            layers.update({'prediction': (data, 'image', {})})

        if threshold_key in f:
            ds = f[threshold_key]
            ds.n_threads = 4
            data = ds[:]
            layers.update({'threshold': (data, 'image', {'contrast_limits': [0, 1]})})

        if segmentation_key in f:
            ds = f[segmentation_key]
            ds.n_threads = 4
            data = ds[:]
            layers.update({'segmentation': (data, 'labels', {})})

    with napari.gui_qt():
        viewer = napari.Viewer()
        for name, (data, layer_type, kwargs) in layers.items():
            func = getattr(viewer, f'add_{layer_type}')
            func(data, name=name, **kwargs)


def validate(res_path, exp_path):
    seg_key = 'connected_components'
    with z5py.File(res_path, 'r') as f:
        ds = f[seg_key]
        res = ds[:]
    with z5py.File(exp_path, 'r') as f:
        ds = f[seg_key]
        exp = ds[:]
    are, _, _ = adapted_rand_error(exp, res)
    print("Adapted rand error:", are)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', type=int, default=0)
    parser.add_argument('--validate', type=int, default=0)
    parser.add_argument('--input_path', type=str, default='data/data_small.n5')
    parser.add_argument('--result_path', type=str, default='./data/result_small.n5')
    parser.add_argument('--expected_path', type=str, default='./data/expected_small.n5')

    args = parser.parse_args()
    in_path = args.input_path
    res_path = args.result_path
    exp_path = args.expected_path

    if bool(args.show):
        show(in_path, res_path)

    if bool(args.validate):
        validate(res_path, exp_path)
