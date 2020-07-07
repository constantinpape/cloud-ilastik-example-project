from itertools import product


class Blocking:

    def make_blocking(self, shape, block_shape):
        ranges = [range(sha // bsha if sha % bsha == 0 else sha // bsha + 1)
                  for sha, bsha in zip(shape, block_shape)]
        start_points = product(*ranges)

        min_coords = [0] * len(shape)
        max_coords = shape

        blocks = [tuple(slice(max(sp * bsha, minc),
                              min((sp + 1) * bsha, maxc))
                        for sp, bsha, minc, maxc in zip(start_point,
                                                        block_shape,
                                                        min_coords,
                                                        max_coords))
                  for start_point in start_points]
        return blocks

    def __init__(self, shape, block_shape):
        self.shape = shape
        self.block_shape = block_shape
        self._blocks = self.make_blocking(shape, block_shape)
        self.n_blocks = len(self._blocks)

    def __getitem__(self, block_id):
        return self._blocks[block_id]


def blocks_to_jobs(shape, block_shape, n_jobs):
    n_blocks = Blocking(shape, block_shape).n_blocks
    block_list = list(range(n_blocks))
    block_lists = []
    for job_id in range(n_jobs):
        block_lists.append(block_list[job_id::n_jobs])
    return block_lists
