import numpy as np
from sklearn.utils import shuffle


def y_to_vector(y, shape):
    result = np.zeros(shape)
    for i in range(result.shape[0]):
        result[i][y[i][0] - 1] = 1
    return result.T


class MultiLayerPerceptron:
    def __init__(self, eta, functions, layers, weights_mean, weights_sigma, bias_mean, bias_sigma, stop_at):
        self.eta = eta
        self.functions = functions
        self.W = [np.random.normal(weights_mean, weights_sigma, size=(layers[i], layers[i + 1]))
                  for i in range(len(layers) - 1)]
        self.biases = [np.random.normal(bias_mean, bias_sigma, size=(1, layers[i + 1])) for i in range(len(layers) - 1)]
        self.A = []
        self.Z = []
        self.package_size = 0
        self.stop_at = stop_at
        self.eras = 0

    def forward(self, X):
        a = X
        self.A.clear()
        self.Z.clear()
        self.A.append(a.T)
        for w_i, act, bias in zip(self.W, self.functions, self.biases):
            z = a.dot(w_i) + bias
            a = act[0](z)
            self.Z.append(z.T)
            self.A.append(a.T)
        return self.A[-1]

    def backward(self, y_pred, y):
        delta = y - y_pred
        deltas = [delta]
        for i in range(len(self.W) - 2, -1, -1):
            deltas.append(self.W[i + 1].dot(deltas[-1]) * self.functions[i][1](self.Z[i]))

        deltas.reverse()
        for i in range(len(self.W)):
            self.W[i] += self.eta / self.package_size * deltas[i].dot(self.A[i].T).T
            self.biases[i] += self.eta / self.package_size * deltas[i].sum(axis=1)

    def fit(self, X, y, X_val, y_val, package_size, eras):
        self.package_size = package_size
        biggest_mean_val = 0
        counter = 0
        self.eras = 0

        for e in range(eras):
            fit = []
            fit_val = []
            X, y = shuffle(X, y, random_state=0)
            X_val, y_val = shuffle(X_val, y_val, random_state=0)

            for i in range(int((len(X)) / package_size) - 1):
                x = X[i * package_size:(i + 1) * package_size]
                y_ = y_to_vector(y[i * package_size:(i + 1) * package_size], (package_size, self.W[-1].shape[1]))

                forward_result = self.forward(x)
                fit.append(np.where(forward_result.argmax(axis=0) == y_.argmax(axis=0), 1, 0).mean())
                self.backward(forward_result, y_)

            for i in range(int((len(X_val)) / package_size) - 1):
                x = X_val[i * package_size:(i + 1) * package_size]
                y_ = y_to_vector(y_val[i * package_size:(i + 1) * package_size], (package_size, self.W[-1].shape[1]))

                forward_result = self.forward(x)
                fit_val.append(np.where(forward_result.argmax(axis=0) == y_.argmax(axis=0), 1, 0).mean())
            mean = np.array(fit).mean()
            mean_val = np.array(fit_val).mean()
            if mean_val < biggest_mean_val:
                counter += 1
            else:
                biggest_mean_val = mean_val
                counter = 0
            if counter > 5 or (self.stop_at and mean_val > self.stop_at):
                self.eras = e
                break
            # print(f"Era {e + 1}  accuracy: {round(mean, 4)} val accuracy: {round(mean_val, 4)}")
