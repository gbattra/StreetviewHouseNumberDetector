import numpy as np
from models.pool_filter_model import PoolFilterModel


class PoolLayerModel:

    def __init__(self,
                 pool_filter: PoolFilterModel,
                 stride: int,
                 name: str,
                 mode='max'):
        self.pool_filter = pool_filter
        self.stride = stride
        self.mode = mode
        self.name = name
        self.forward_cache = {}
        self.backward_cache = {}

    def forward_propogate(self, A_prev):
        # get dims of previous input
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape

        # get hparameters for this layer
        f = self.pool_filter.filter_size
        stride = self.stride

        # get dimensions of output volume
        n_H, n_W = self.compute_output_dimensions(n_H_prev, n_W_prev, f, stride)
        n_C = n_C_prev  # because with pool, channels don't expand or condense

        # initialize output matrix
        A = np.zeros((m, n_H, n_W, n_C))

        # loop over training examples
        for i in range(m):
            # select training example
            a_prev = A_prev[i]
            # loop over vertical axis of example
            for h in range(n_H):
                # loop over horizontal axis of example
                for w in range(n_W):
                    # loop over all channels
                    for c in range(n_C):
                        # find the corners of the slice
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # use corners to get slice of example to pool over
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                        # compute pool operation based on model's mode attr
                        if self.mode == 'max':
                            A[i, h, w, c] = a_prev_slice.max()
                        elif self.mode == 'average':
                            A[i, h, w, c] = a_prev_slice.mean()

        self.forward_cache = {
            'A_prev': A_prev,
            'A': A,
            'hparameters': {
                'filter_size': f,
                'stride': stride
            }
        }

        return A

    def backward_propogate(self, grads):
        dZ = grads['dZ']
        # get info from cache
        A_prev = self.forward_cache['A_prev']
        A = self.forward_cache['A']
        stride = self.forward_cache['hparameters']['stride']
        f = self.forward_cache['hparameters']['filter_size']

        # get dims from A_prev and dA
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = A.shape

        # resize dZ if needed
        dZ = dZ.T.reshape(m, n_H, n_W, n_C)

        # define placeholder for grad for inputs
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))

        # loop over training examples
        for i in range(m):
            # select ith training example
            a_prev = A_prev[i]

            # loop over vert axis of output
            for h in range(n_H):
                # loop over horiz axis of output
                for w in range(n_W):
                    # loop over channels of outputs
                    for c in range(n_C):
                        # get the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        if self.mode == 'max':
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                            mask = self.create_mask_from_window(x=a_prev_slice)
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dZ[i, h, w, c])

                        elif self.mode == 'average':
                            da = dZ[i, h, w, c]
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += self.create_mask_from_window(dz=da)

        self.backward_cache = {
            'dA_prev': dA_prev
        }

        return {
            'dZ': dA_prev
        }

    def update_weights(self):
        return self  # pool layer has no weights to update

    def store_weights(self, directory=''):
        return self  # no weights to save

    def create_mask_from_window(self, dz=None, x=None):
        if self.mode == 'max':
            mask = (x == np.max(x))
        elif self.mode == 'average':
            f = self.pool_filter.filter_size
            avg = 1 / dz
            mask = np.ones((f, f)) * avg

        return mask

    def compute_output_dimensions(self, n_H_prev: int, n_W_prev: int, filter_size: int, stride_size: int):
        n_H = int(1 + (n_H_prev - filter_size) / stride_size)
        n_W = int(1 + (n_W_prev - filter_size) / stride_size)

        return n_H, n_W
