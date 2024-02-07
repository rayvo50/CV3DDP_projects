import numpy as np


# base class for a layer as a template for diferent layers
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Compute the output Y of a layer for a given input X
    def forward(self, input):
        raise NotImplementedError

    # Compute the gradient dE/dX for a given output gradient dE/dY and learning rate.
    def backward(self, output_error, learning_rate):
        raise NotImplementedError


# fully connected layer
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # weights and biases randomly initialized
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    # forward propagation
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # backward propagation
    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_error, axis=0)
        return input_error


# activation layer that supports many diferent activation functions
class Activation(Layer):
    def __init__(self, activation, activation_derivative=None):
        # check if a default actvation is used
        if isinstance(activation, str):
            activations = {
                'relu': (self.relu, self.relu_derivative),
                'leaky_relu': (self.leaky_relu, self.leaky_relu_derivative),
                'sigmoid': (self.sigmoid, self.sigmoid_derivative),
                'softmax': (self.softmax, self.softmax_derivative)
            }

            if activation not in activations:
                raise ValueError(f"Invalid activation function: '{activation}'")

            self.activation, self.activation_derivative = activations[activation]
        else:
            # if custom activation functions are provided
            if not callable(activation) or (activation_derivative is not None and not callable(activation_derivative)):
                print("Invalid activation function")
            
            self.activation = activation
            self.activation_derivative = activation_derivative

    ##### Activation functions and their derivatives #####
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    
    def softmax_derivative(self, x):
        return np.ones_like(x)

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return output_error * self.activation_derivative(self.input)


class Loss:
    def __init__(self, loss_function, loss_derivative=None):
        # check if a default loss is used
        default_losses = {
            'cross_entropy': (self.cross_entropy, self.cross_entropy_derivative)
        }

        # If loss_function is a string, use a predefined loss function
        if isinstance(loss_function, str):
            if loss_function not in default_losses:
                raise ValueError(f"Invalid loss function: '{loss_function}'")
            self.loss, self.loss_derivative = default_losses[loss_function]
        else:
            # if custom loss function is provided
            if not callable(loss_function) or (loss_derivative is not None and not callable(loss_derivative)):
                raise ValueError("Custom loss function and its derivative must be callable.")
            self.loss = loss_function
            self.loss_derivative = loss_derivative


    ##### Loss functions and their derivatives #####
    def cross_entropy(self, y, t):
        m = t.shape[0]
        return -np.sum(t * np.log(y + 1e-15)) / m

    def cross_entropy_derivative(self, y, t):
        return y - t

    # Forward and backward methods
    def forward(self, y, t):
        return self.loss(y, t)

    def backward(self, y, t):
        return self.loss_derivative(y, t)



# main class for building the model
class Model:
    def __init__(self):
        self.layers = []  
        self.loss = None 
        self.loss_derivative = None

    # add a layer to he network in order
    def add(self, layer):
        self.layers.append(layer)

    # set the loss for training
    def set_loss(self, loss):
        self.loss = loss

    # run forward propagation based on input
    def predict(self, input_data):    
        result = []
        for i in range(len(input_data)):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result


    # train the model
    def fit(self, x_train, y_train, x_val, y_val, epochs, learning_rate, batch_size):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            # training
            train_err, train_correct, train_samples = 0, 0, 0
            for j in range(0, len(x_train), batch_size):
                x_batch = x_train[j:j + batch_size]
                y_batch = y_train[j:j + batch_size]

                # forward propagation
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # backward propagation
                error = self.loss.backward(output, y_batch)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

                # compute loss and accuracy
                train_err += self.loss.forward(output, y_batch)
                train_correct += np.sum(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1))  # argmax because output is onehot encoded
                train_samples += len(x_batch)

            # validation
            val_err, val_correct, val_samples = 0, 0, 0
            for j in range(0, len(x_val), batch_size):
                x_batch = x_val[j:j + batch_size]
                y_batch = y_val[j:j + batch_size]

                # forward propagation
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # compute loss and accuracy
                val_err += self.loss.forward(output, y_batch)
                val_correct += np.sum(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1)) 
                val_samples += len(x_batch)

            # print results per epoch
            train_loss = train_err / train_samples
            train_accuracy = train_correct / train_samples
            val_loss = val_err / val_samples
            val_accuracy = val_correct / val_samples
            print(f"Epoch: {epoch+1}/{epochs}  |  Loss: {train_loss:.4f}  |  Accuracy: {train_accuracy:.4f}  |  Val Loss: {val_loss:.4f}  |  Val Accuracy: {val_accuracy:.4f}")

            # metrics for plotting
            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

        return history

    # count total number of parameters of the model
    def count_params(self):
        total_params = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                total_params += np.prod(layer.weights.shape)
            if hasattr(layer, 'bias'):
                total_params += np.prod(layer.bias.shape)
        return total_params