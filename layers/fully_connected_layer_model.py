import numpy as np
from services.weight_initializer_service import DenseNNWeightInitializerService
from pathlib import Path


class FullyConnectedLayerModel:

    def __init__(self,
                 units_in: int,
                 units_out: int,
                 m: int,
                 name: str,
                 alpha: float = 1.0):
        self.units_in = units_in
        self.units_out = units_out
        self.m = m
        self.name = name
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
        self.__load_weights()

    def __load_weights(self):
        W, b = DenseNNWeightInitializerService.random_initialize_weights([self.units_in, self.units_out])
        W = W.reshape(self.units_out, self.units_in)
        b = b.reshape(1, self.units_out)

        self.W, self.b = W, b

    def forward_propogate(self, A_prev):
        # get dims and use them to flatten A_prev
        if len(A_prev.shape) > 2:
            m, n_H, n_W, n_C = A_prev.shape
            A_prev = A_prev.reshape(m, n_H * n_W * n_C)

        a = A_prev.dot(self.W.T)
        a += self.b

        self.forward_cache = {
            'A_prev': A_prev,
            'A': a,
            'W': self.W,
            'b': self.b
        }

        return a

    def backward_propogate(self, grads):
        dZ = grads['dZ']
        A_prev = self.forward_cache['A_prev']
        if len(A_prev.shape) > 2:
            m, n_H, n_W, n_C = self.forward_cache['A_prev'].shape
            A_prev = self.forward_cache['A_prev'].reshape(m, n_H * n_W * n_C)
        dW = (A_prev.T.dot(dZ)).T / self.m
        db = np.sum(dZ) / self.m

        # update dZ for previous layer output
        dZ = self.W.T.dot(dZ.T)

        self.backward_cache = {
            'dZ': dZ,
            'dW': dW,
            'db': db
        }

        return {
            'dZ': dZ
        }

    def update_weights(self, iteration: int):
        # update_param_W, update_param_b = self.backward_cache['dW'], self.backward_cache['db']
        update_param_W, update_param_b = self.compute_momentum_params(iteration)

        self.W -= self.alpha * update_param_W
        self.b -= self.alpha * update_param_b

        return self

    def store_weights(self):
        fW = self.W.flatten()
        fb = self.b.flatten()
        np.savetxt('stored/' + self.name + '_W', fW)
        np.savetxt('stored/' + self.name + '_b', fb)

        return self

    def compute_momentum_params(self, iteration: int):
        # compute momentum gradients
        vdW = (self.update_params['beta_one'] * self.update_params['vdW']) + (1 - self.update_params['beta_one']) * self.backward_cache['dW']
        vdb = (self.update_params['beta_one'] * self.update_params['vdb']) + (1 - self.update_params['beta_one']) * self.backward_cache['db']
        # set as corrected grads
        self.update_params['vdW'] = vdW / (1 - np.power(self.update_params['beta_one'], iteration))
        self.update_params['vdb'] = vdb / (1 - np.power(self.update_params['beta_one'], iteration))

        update_param_W = self.update_params['vdW']
        update_param_b = self.update_params['vdb']

        return update_param_W, update_param_b
