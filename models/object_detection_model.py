# the CNN model to detect objects in an image
import numpy as np
import matplotlib.pyplot as plt


class ObjectDetectionModel:

    x_train = []

    def __init__(self,
                 data: dict,
                 epochs: int,
                 layers: list
                 ):
        self.data = data
        self.epochs = epochs
        self.layers = layers
        self.prediction = None
        self.cost_history = []
        self.y_pred = []

    # train model using this CNN architecture: X -> CONV -> POOL -> FC -> SOFTMAX
    def train(self):
        # loop over epochs and perform gradient descent
        for epoch in range(self.epochs):
            print('Epoch: ' + str(epoch))
            # forward propogate and get predictions
            self.y_pred = self.forward_propogate(self.data.x)

            # compute the cost and use it to track J_history
            cost = self.compute_cost(self.data.y, self.y_pred)
            print('Cost: ' + str(cost))
            self.cost_history.append(cost)

            # use cost to perform backpropogations across the layers
            self.backward_propogate()

            # self.gradient_check(self.layers[0])

            # update the weights
            self.update_weights(epoch + 1)  # plus 1 to avoid divide by zero

        # save the weights
        self.store_weights()

    def forward_propogate(self, A_prev):
        for layer in self.layers:
            A_prev = layer.forward_propogate(A_prev)

        return A_prev

    def compute_cost(self, y, y_prediction):
        m = y.shape[0]
        cost = -(np.sum(y * np.log(y_prediction + 0.001) + (1 - y) * np.log(1 - y_prediction + 0.001))) / m  # added + 0.001 to avoid log of zeros
        return cost

    def backward_propogate(self):
        # get starting grad for y prediction
        dZ = np.subtract(self.y_pred, self.data.y)

        grads = {
            'dZ': dZ
        }

        # add grads to skipped layer
        self.layers[len(self.layers) - 1].backward_cache = grads

        for layer in reversed(self.layers[:-1]):  # skip output layer as it is computed above
            grads = layer.backward_propogate(grads)

        return grads

    def update_weights(self, iteration: int):
        for layer in self.layers:
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                layer.update_weights(iteration)

        return True

    def store_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                layer.store_weights()

    def gradient_check(self, layer):
        # get grads from layer
        grads = layer.backward_cache['dW']
        # flatten layer W
        shape = layer.W.shape
        W_flat = layer.W.flatten()

        epsilon = 0.001

        print('Numerical Grad', 'Computed Grad')
        # loop through first few W's
        for i in range(0, 10):
            W_initial = W_flat[i]
            W_plus = W_initial + epsilon
            W_minus = W_initial - epsilon

            W_flat[i] = W_plus
            layer.W = W_flat.reshape(shape)
            pred = self.forward_propogate(self.data.x)
            cost_plus = self.compute_cost(self.data.y, pred)

            W_flat[i] = W_minus
            layer.W = W_flat.reshape(shape)
            pred = self.forward_propogate(self.data.x)
            cost_minus = self.compute_cost(self.data.y, pred)

            computed_grad = (cost_plus - cost_minus) / (2 * epsilon)

            print(grads.flatten()[i], computed_grad)

            # reset layers W's
            W_flat[i] = W_initial
            layer.W = W_flat.reshape(shape)

        return layer

    def display_data(self):
        for i, image in enumerate(self.data.x):
            plt.imshow(image)
            plt.title(np.argmax(self.data.y[i]))
            plt.show()
