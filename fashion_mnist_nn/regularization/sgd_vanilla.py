
from typing import List

import numpy as np
from scipy.special import softmax

rng = np.random.default_rng()


def ReLU_derivative(x: np.array):
    r = np.ones(x.shape)
    r[r == 0] = 0
    return r


class NeuralNetwork:
    def __init__(self,
                 layers_size: List[int],
                 learning_rate: float,
                 mini_batch_size: int,
                 training_epochs: int
                 ):
        # Save training hyper-parameters
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.training_epochs = training_epochs
        # Create layers
        self.num_layers = len(layers_size) - 1
        self.weights = []
        self.biases = []
        for l in range(self.num_layers):
            k = np.sqrt(1 / layers_size[l])
            self.weights.append( rng.uniform(-k, k, size=(layers_size[l+1], layers_size[l])) )
            self.biases.append( rng.uniform(-k, k, size=layers_size[l+1]) )
        # Stores for logits and activations
        self.z = []
        self.a = []
        for size in layers_size[1:]:
            self.z.append( np.empty(size) )
            self.a.append( np.empty(size) )
        # Stores for gradients
        self.nabla_z = []
        self.sum_nabla_z = []
        self.nabla_w = []
        self.sum_nabla_w = []
        for l in range(self.num_layers):
            self.nabla_z.append( np.empty(self.biases[l].shape) )
            self.sum_nabla_z.append( np.empty(self.biases[l].shape) )
            self.nabla_w.append( np.empty(self.weights[l].shape) )
            self.sum_nabla_w.append( np.empty(self.weights[l].shape) )

    def forward(self, x: np.array):
        # First layer
        l = 0
        self.z[l] = self.weights[l] @ x.flatten() + self.biases[l]
        self.a[l] = np.maximum(0, self.z[l])
        # Hidden layers
        for l in range(1, self.num_layers-1):
            self.z[l] = self.weights[l] @ self.a[l-1] + self.biases[l]
            self.a[l] = np.maximum(0, self.z[l])
        # Last layer
        l = self.num_layers-1
        self.z[l] = self.weights[l] @ self.a[l-1] + self.biases[l]
        self.a[l] = softmax(self.z[l])
        return self.a[-1]
    
    def loss(self, y: np.array):
        return - (y * np.log(self.a[-1])).sum()
    
    def test(self, data: List[np.array]):
        n_samples = len(data)
        correct = 0
        sum_loss = 0
        for x, y in data:
            y_hat = self.forward(x)
            correct += y.argmax() == y_hat.argmax()
            sum_loss += self.loss(y)
        return correct/n_samples, sum_loss/n_samples
    
    def calc_gradients(self, x, y):
        self.forward(x)
        # Last layer
        l = self.num_layers - 1
        self.nabla_z[l] = self.a[l] - y
        self.nabla_w[l] = np.outer(self.nabla_z[l], self.a[l-1])
        # Hidden layers
        for i in range(1, self.num_layers-1):
            l = (self.num_layers-1) - i
            self.nabla_z[l] = ReLU_derivative(self.z[l]) * (self.weights[l+1].T @ self.nabla_z[l+1])
            self.nabla_w[l] = np.outer(self.nabla_z[l], self.a[l-1])
        # First layer
        l = 0
        self.nabla_z[l] = ReLU_derivative(self.z[l]) * (self.weights[l+1].T @ self.nabla_z[l+1])
        self.nabla_w[l] = np.outer(self.nabla_z[l], x.flatten())
        return self.loss(y)
    
    def train(self, train_data, test_data):
        n_mini_batches = len(train_data) // self.mini_batch_size
        accuracy_by_epoch = []
        loss_by_epoch = []
        # Performance baseline
        print("Pre-train stats")
        accuracy, avg_loss = self.test(test_data)
        accuracy_by_epoch.append(accuracy)
        loss_by_epoch.append(avg_loss)
        print(f"==> Accuracy: {100*accuracy:0.1f}%, Avg loss: {avg_loss:>8f}\n")
        # Training loop
        for epoch in range(self.training_epochs):
            print(f"Epoch {epoch}\n" + 20*'-')
            rng.shuffle(train_data)
            for mini_batch_number in range(n_mini_batches):
                sum_loss = 0
                for l in range(self.num_layers):
                    self.sum_nabla_z[l] = 0
                    self.sum_nabla_w[l] = 0

                i = mini_batch_number * self.mini_batch_size
                j = (mini_batch_number+1) * self.mini_batch_size
                mini_batch = train_data[i:j]
                n = len(mini_batch)

                for x, y in mini_batch:
                    sum_loss += self.calc_gradients(x, y)
                    for l in range(self.num_layers):
                        self.sum_nabla_z[l] += self.nabla_z[l]
                        self.sum_nabla_w[l] += self.nabla_w[l]

                for l in range(self.num_layers):
                    self.biases[l] -= self.learning_rate * self.sum_nabla_z[l] / n
                    self.weights[l] -= self.learning_rate * self.sum_nabla_w[l] / n

                if mini_batch_number % 100 == 0:
                    print(f"loss: {sum_loss/n:>8f} [mini-batch {mini_batch_number} / {n_mini_batches}]")
            # Test at the end of each epoch
            accuracy, avg_loss = self.test(test_data)
            accuracy_by_epoch.append(accuracy)
            loss_by_epoch.append(avg_loss)
            print(f"==> Accuracy: {100*accuracy:0.1f}%, Avg loss: {avg_loss:>8f}\n")
        return accuracy_by_epoch, loss_by_epoch