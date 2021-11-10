from datetime import date, datetime
import numpy as np
from keras.datasets import mnist
from MLP import MultiLayerPerceptron
import matplotlib.pyplot as plt


def relu(x):
    return np.where(x > 0, x, 0)


def relu_p(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_p(x):
    return 1 - tanh(x)**2


def softmax(X):
    nominator = np.exp(X)
    return nominator / np.sum(nominator, axis=1).reshape(len(nominator), 1)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = x_train.reshape(x_train.shape[0], 784) / 255
X_test = x_test.reshape(x_test.shape[0], 784) / 255

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)


# Experiments
def exp1():
    list_ = []
    hidden_layer_sizes = [15, 30, 60, 100, 200]
    for hidden_layer_size in hidden_layer_sizes:
        total = 0
        for i in range(repeats):
            model = MultiLayerPerceptron(functions=((sigmoid, sigmoid_p), (softmax, None)), eta=0.1, weights_mean=0,
                                         weights_sigma=0.1, layers=(784, hidden_layer_size, 10), bias_mean=0, bias_sigma=0.1,
                                         stop_at=0.95)
            model.fit(X_train, y_train, X_test, y_test, package_size=100, eras=200)
            total += model.eras
        list_.append(total / repeats)
    return hidden_layer_sizes, list_


def exp2():
    list_ = []
    etas = [0.001, 0.01, 0.1]
    for eta in etas:
        total = 0
        for i in range(repeats):
            model = MultiLayerPerceptron(functions=((tanh, tanh_p), (softmax, None)), eta=eta, weights_mean=0,
                                         weights_sigma=0.1, layers=(784, 100, 10), bias_mean=0, bias_sigma=0.1,
                                         stop_at=0.95)
            model.fit(X_train, y_train, X_test, y_test, package_size=50, eras=200)
            total += model.eras
        list_.append(total / repeats)
    return etas, list_


def exp3():
    list_ = []
    batches = [20, 50, 100, 200]
    for batch in batches:
        total = 0
        for i in range(repeats):
            model = MultiLayerPerceptron(functions=((sigmoid, sigmoid_p), (softmax, None)), eta=0.1, weights_mean=0,
                                         weights_sigma=0.1, layers=(784, 100, 10), bias_mean=0, bias_sigma=0.1,
                                         stop_at=0.95)
            model.fit(X_train, y_train, X_test, y_test, package_size=batch, eras=200)
            total += model.eras
        list_.append(total / repeats)
    return batches, list_


def exp4():
    list_ = []
    sigmas = [0.01, 0.1, 0.5, 1]
    for sigma in sigmas:
        total = 0
        for i in range(repeats):
            model = MultiLayerPerceptron(functions=((sigmoid, sigmoid_p), (softmax, None)), eta=0.1, weights_mean=0,
                                         weights_sigma=sigma, layers=(784, 100, 10), bias_mean=0, bias_sigma=sigma,
                                         stop_at=0.95)
            model.fit(X_train, y_train, X_test, y_test, package_size=100, eras=200)
            total += model.eras
        list_.append(total / repeats)
    return sigmas, list_


def exp5():
    list_ = []
    funcs = [(sigmoid, sigmoid_p), (tanh, tanh_p), (relu, relu_p)]
    for fun in funcs:
        total = 0
        for i in range(repeats):
            model = MultiLayerPerceptron(functions=(fun, (softmax, None)), eta=0.1, weights_mean=0,
                                         weights_sigma=0.1, layers=(784, 100, 10), bias_mean=0, bias_sigma=0.1,
                                         stop_at=0.95)
            model.fit(X_train, y_train, X_test, y_test, package_size=100, eras=200)
            total += model.eras
        list_.append(total / repeats)
    return ["sigmoid", "tanh", "relu"], list_


def exp6():
    list_ = []
    stops = [0.9, 0.93, 0.96, None]
    for stop in stops:
        total = 0
        for i in range(repeats):
            model = MultiLayerPerceptron(functions=((sigmoid, sigmoid_p), (softmax, None)), eta=0.1, weights_mean=0,
                                         weights_sigma=0.1, layers=(784, 100, 10), bias_mean=0, bias_sigma=0.1,
                                         stop_at=stop)
            model.fit(X_train, y_train, X_test, y_test, package_size=100, eras=200)
            total += model.eras
        list_.append(total / repeats)
    return ["0.9", "0.93", "0.96", "early stop"], list_


def print_result(result_x, result_y, divider):
    print(f" {divider} ".join(map(str, result_x)))
    print(f" {divider} ".join(map(str, result_y)))


def make_exp(exp, xscale, title, x, y, dest, show, bar):
    print(f"Rozpoczeto {title.lower()}")
    exp_result = exp()
    print_result(*exp_result, "&")

    plt.bar(*exp_result) if bar else plt.plot(*exp_result)
    if xscale:
        plt.xscale(xscale)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show() if show else plt.savefig(dest)
    plt.clf()
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Zakonczono {title.lower()}\n{dt_string}")


show_tests = False
repeats = 10
dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print(f"Start {dt_string}")
# make_exp(exp=exp1, xscale="linear", title="Wpływ wielkości warstwy ukrytej na szybkość uczenia", x="wielkość warstwy ukrytej", y="epoki", dest="plots/exp1.png", show=show_tests, bar=False)

make_exp(exp=exp2, xscale="log", title="Wpływ współczynnika uczenia na szybkość uczenia", x="eta", y="epoki", dest="plots/exp2.png", show=show_tests, bar=False)

# make_exp(exp=exp3, xscale="linear", title="Wpływ wielkości batcha na szybkość uczenia", x="batch", y="epoki", dest="plots/exp3.png", show=show_tests, bar=False)

# make_exp(exp=exp4, xscale="linear", title="Wpływ wielkości odchylenia standardowego rozkladu normalnego macierzy W oraz biasu na szybkość uczenia", x="sigma", y="epoki", dest="plots/exp4.png", show=show_tests, bar=False)

# make_exp(exp=exp5, xscale="linear", title="Wpływ funkcji aktywacji na szybkość uczenia", x="funkcja", y="epoki", dest="plots/exp5.png", show=show_tests, bar=True)

# make_exp(exp=exp6, xscale="linear", title="Badanie warunku zatrzymania W na szybkość uczenia", x="warunek zatrzymania (próg dokładności)", y="epoki", dest="plots/exp6.png", show=show_tests, bar=True)