import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

# Layer class
#moving all generalized (what's done in 1f1 to here)
class Layer:
    def __init__(self, input_dim, output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        ## nn-dums == [2,3,4,2]
        ## w[0] shape == 2x3, b[0] = 1x3
        ##w[1] shape == 3x4, b[1] = 1x4
        ## w[2] shape == 4x2, b[2] = 1x2
        ##b_n 
        # initialize the weights and biases in the network``
        np.random.seed(seed)
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda


    def actFun(self, z):
        if self.actFun_type == "tanh":
            return np.tanh(z)
        elif self.actFun_type == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.actFun_type == "relu":
            return np.maximum(0, z)
        else:
            return z

    def diff_actFun(self, z):
        if self.actFun_type == "tanh":
            return 1 - np.tanh(z) ** 2
        elif self.actFun_type == "sigmoid":
            a = 1 / (1 + np.exp(-z))
            return a * (1 - a)
        elif self.actFun_type == "relu":
            return np.where(z > 0, 1, 0)
        else:
            return np.ones_like(z)

    def feedforward(self, X):
 
        # YOU IMPLEMENT YOUR feedforward HERE
        #z(1) = a(1-1)*w(1) +b(1), 
        #a(1) = f'(z_1) till second to last layer (== len(self.W))
        #with a(0) = X
        self.input = X
        self.z = np.dot(X, self.W) + self.b
        self.a = self.actFun(self.z)
        return self.a

  
    def backprop(self, delta_next, W_next=None):
        #1. compute delta 
        #2. compute weight gradient (L2 regularization) a_T*detla + reg_lambda *W
        #3. compute bias gradient 
        if W_next is not None:
            
            delta = np.dot(delta_next, W_next.T) * self.diff_actFun(self.z)
        else:
            delta = delta_next * self.diff_actFun(self.z)
        dW = np.dot(self.input.T, delta) + self.reg_lambda * self.W
        db = np.sum(delta, axis=0, keepdims=True)
        return delta, dW, db


class DeepNeuralNetwork:
    def __init__(self, nn_dims, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_dims: dimension of each layers
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.layers = []
        self.reg_lambda = reg_lambda
        self.actFun_type = actFun_type

        for i in range(len(nn_dims) - 1):
            self.layers.append(Layer(nn_dims[i], nn_dims[i + 1],
                                     actFun_type=actFun_type,
                                     reg_lambda=reg_lambda,
                                     seed=seed))


    def feedforward(self, X):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''
        #let each layer handles its own 
        a = X
        for layer in self.layers:
            a = layer.feedforward(a)
        self.z_out = self.layers[-1].z  # last layer’s pre-activation
        shifted_logits = self.z_out - np.max(self.z_out, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2, ... dL/dn, dL/bn in two lists
        '''
        num_examples = len(X)
        delta = self.probs.copy()
        delta[range(num_examples), y] -= 1

        dW, db = [], []
        next_delta = delta
        next_W = None

        for layer in reversed(self.layers):
            next_delta, dW_i, db_i = layer.backprop(next_delta, W_next=next_W)
            dW.insert(0, dW_i)
            db.insert(0, db_i)
            next_W = layer.W

        return dW, db


    def calculate_loss(self, X, y):
        num_examples = len(X)
        probs = self.feedforward(X)
        correct_logprobs = -np.log(probs[range(num_examples), y] + 1e-9)
        data_loss = np.sum(correct_logprobs)
        W_sum = sum([np.sum(np.square(layer.W)) for layer in self.layers])
        data_loss += self.reg_lambda / 2 * W_sum
        return (1. / num_examples) * data_loss

    def predict(self, X):
        probs = self.feedforward(X)
        return np.argmax(probs, axis=1)


    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(num_passes):
            self.feedforward(X)
            dW, db = self.backprop(X, y)
            for j, layer in enumerate(self.layers):
                layer.W -= epsilon * dW[j]
                layer.b -= epsilon * db[j]
            if print_loss and i % 1000 == 0:
                print(f"Loss after iteration {i}: {self.calculate_loss(X, y):.6f}")

    def visualize_decision_boundary(self, X, y):
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    X, y = generate_data()
    model = DeepNeuralNetwork(nn_dims=[2, 3, 2], actFun_type='tanh', reg_lambda=0.01)
    model.fit_model(X, y, epsilon=0.01, num_passes=10000, print_loss=True)
    model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()


from sklearn.datasets import make_circles

X, y = make_circles(n_samples=300, noise=0.20, random_state=42)

#configuration for comparison
configs = [
    {"nn_dims": [2, 3, 2], "actFun": "tanh"},
    {"nn_dims": [2, 6, 2], "actFun": "tanh"},
    {"nn_dims": [2, 4, 3, 2], "actFun": "tanh"},
    {"nn_dims": [2, 3, 2], "actFun": "relu"},
    {"nn_dims": [2, 6, 2], "actFun": "relu"},
    {"nn_dims": [2, 4, 3, 2], "actFun": "relu"},
    {"nn_dims": [2, 3, 2], "actFun": "sigmoid"},
    {"nn_dims": [2, 6, 2], "actFun": "sigmoid"},
    {"nn_dims": [2, 4, 3, 2], "actFun": "sigmoid"}
]

fig, axes = plt.subplots(3, len(configs)//3, figsize=(15, 12))
fig.suptitle("Decision Boundaries for Different Architectures and Activations", fontsize=14)

for i, cfg in enumerate(configs):
    row, col = divmod(i, len(configs)//3)
    ax = axes[row, col]
    model = DeepNeuralNetwork(
        nn_dims=cfg["nn_dims"],
        actFun_type=cfg["actFun"],
        reg_lambda=0.01
    )
    model.fit_model(X, y, epsilon=0.01, num_passes=8000, print_loss=False)
    plt.sca(ax)
    model.visualize_decision_boundary(X, y)
    ax.set_title(f"{cfg['nn_dims']} | {cfg['actFun']}")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()





### run this for cross-activation comapison 
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.20, random_state=42)


architectures = [
    [2, 3, 2],       # shallow, small hidden layer
    [2, 6, 2],       # shallow, wider
    [2, 4, 3, 2],    # 2 hidden layers
    [2, 4, 3, 2, 2]  # 3 hidden layers
]

activations = ['tanh', 'relu', 'sigmoid']

for act in activations:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Decision Boundaries for Activation: {act.upper()}", fontsize=14, y=0.95)
    axes = axes.flatten()

    for i, nn_dims in enumerate(architectures):
        print(f"\nTraining Deep Neural Network with nn_dims={nn_dims}, activation={act}")
        model = DeepNeuralNetwork(
            nn_dims=nn_dims,
            actFun_type=act,
            reg_lambda=0.01
        )


        model.fit_model(X, y, epsilon=0.01, num_passes=10000, print_loss=False)


        plt.sca(axes[i])
        model.visualize_decision_boundary(X, y)
        axes[i].set_title(f"nn_dims: {nn_dims}")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
