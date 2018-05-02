# the model object for a CONV Layer
import numpy as np
from models.conv_filter_model import CONVFilterModel
from services.weight_initializer_service import CNNWeightInitializerService
import helpers.image_transform as it
from pathlib import Path


class CONVLayerModel:

    weights: {}

    def __init__(self,
                 conv_filter: CONVFilterModel,
                 stride: list,
                 padding: str,
                 name: str,
                 alpha: float = 1.0):
        self.conv_filter = conv_filter
        self.stride = stride
        self.padding = padding
        self.alpha = alpha
        self.forward_cache = {}
        self.backward_cache = {}
        self.update_params = {
            'sdW': 0,
            'sdb': 0,
            'vdW': 0,
            'vdb': 0,
            'beta_one': 0.9,
            'beta_two': 0.999,
            'epsilon': 10e-8
        }
        self.name = name
        self.__load_weights()

    def __load_weights(self):
        W, b = CNNWeightInitializerService.random_initialize_filters([self.conv_filter.filter_size, self.conv_filter.channels_in, self.conv_filter.channels_out])

        self.W, self.b = W, b

    def forward_propogate(self, A_prev):
        # get dims from a and weight shapes
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        f, f, n_C_prev, n_C = self.W.shape

        # get height and width of output
        n_H, n_W = self.compute_output_dimensions(n_H_prev, n_W_prev)

        # initialize output volume as zeros
        Z = np.zeros((m, n_H, n_W, n_C))

        # pad a_prev with zeros
        pad = self.get_pad_size()
        stride = self.get_stride_size()
        A_prev_pad = it.zero_pad(A_prev, pad_size=pad, is_batch=True)

        # loop over the batch of training examples
        for i in range(m):
            # select ith training example
            a_prev_pad = A_prev_pad[i]
            # loop over the vertical axis of the training example
            for h in range(n_H):
                # loop over horizontal axis of training example
                for w in range(n_W):
                    # loop over classes/channels of training example
                    for c in range(n_C):
                        # find corneres of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # use these corners to get the slice to pass to forward_single_step()
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # convolve the 3D slice with the corresponding filter W and b (bias)
                        Z[i, h, w, c] = self.forward_single_step(a_slice_prev, self.W[:, :, :, c], self.b)

        # store info in "cache" to use in backprop
        self.forward_cache = {
            'A_prev': A_prev,
            'W': self.W,
            'b': self.b,
            'hparameters': {
                'pad': pad,
                'stride': stride
            }
        }

        # return convolved output
        return Z

    def forward_single_step(self, a_prev_slice, W, b):
        s = np.multiply(a_prev_slice, W)
        Z = np.sum(s)
        Z += float(np.sum(b))

        return Z

    def backward_propogate(self, grads):
        dZ = grads['dZ']
        # get info from cache
        A_prev = self.forward_cache['A_prev']
        W = self.forward_cache['W']
        b = self.forward_cache['b']
        stride = self.forward_cache['hparameters']['stride']
        pad = self.forward_cache['hparameters']['pad']

        # get A_prev and W dimensions
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        f, f, n_C_prev, n_C = W.shape

        # retrieve dZ's dims
        m, n_H, n_W, n_C = dZ.shape

        # initialize placeholder matrices
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        # pad A_prev and dA_prev
        A_prev_pad = it.zero_pad(A_prev, pad)
        dA_prev_pad = it.zero_pad(dA_prev, pad)

        for i in range(m):
            # select ith training example from padded A_prev and dA_prev
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            # loop over vert axis of output volume
            for h in range(n_H):
                # loop over horiz axis of output volume
                for w in range(n_W):
                    # loop over the channels of the output volume
                    for c in range(n_C):
                        # find the corners of the slice to select
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # select the slice of the input volume (A_prev_pad)
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # update grads for the slice and the filter's params
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]

            dA_prev[i, :, :, :] = da_prev_pad

        self.backward_cache = {
            'dA_prev': dA_prev,
            'dW': dW,
            'db': db
        }

        return {
            'dZ': dA_prev
        }

    def update_weights(self, iteration: int):
        # updat_param_W, update_param_b = self.backward_cache['dW'], self.backward_cache['db']
        updat_param_W, update_param_b = self.compute_momentum_params(iteration)

        self.W -= self.alpha * updat_param_W
        self.b -= self.alpha * update_param_b

        # self.store_weights()
        return self

    def store_weights(self):
        fW = self.W.flatten()
        fb = self.b.flatten()
        np.savetxt('stored/' + self.name + '_W', fW)
        np.savetxt('stored/' + self.name + '_b', fb)

        return self

    def compute_output_dimensions(self, n_H_prev: int, n_W_prev: int):
        pad_size = self.get_pad_size()
        stride_size = self.get_stride_size()
        filter_size = self.get_filter_size()
        n_H = int(((n_H_prev - filter_size + 2 * pad_size) / stride_size) + 1)
        n_W = int(((n_W_prev - filter_size + 2 * pad_size) / stride_size) + 1)

        return n_H, n_W

    def get_pad_size(self):
        pad = self.conv_filter.filter_size if self.padding == 'SAME' else 0
        return pad

    def get_stride_size(self):
        return self.stride[0]

    def get_filter_size(self):
        return self.conv_filter.filter_size

    def compute_momentum_params(self, iteration: int):
        # compute momentum gradients
        vdW = self.update_params['beta_one'] * self.update_params['vdW'] + (1 - self.update_params['beta_one']) * self.backward_cache['dW']
        vdb = self.update_params['beta_one'] * self.update_params['vdb'] + (1 - self.update_params['beta_one']) * self.backward_cache['db']

        # set as corrected grads
        self.update_params['vdW'] = vdW / (1 - np.power(self.update_params['beta_one'], iteration))
        self.update_params['vdb'] = vdb / (1 - np.power(self.update_params['beta_one'], iteration))

        # use adam to update weights
        update_param_W = self.update_params['vdW']
        update_param_b = self.update_params['vdb']

        return update_param_W, update_param_b
