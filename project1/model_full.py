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


# Fully connected layer
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Weights are initialized with small random values and biases are initialized with zeros
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    # Forward propagation for a fully connected layer is the affine transformation: output = input * weights + bias
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # Backward propagation for a fully connected layer computes the gradient of the loss function
    # with respect to the input, which is needed for the previous layer, as well as the gradient
    # with respect to the weights and biases for updating these parameters.
    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # Update parameters: Gradient descent
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_error, axis=0)
        return input_error


# ActivationLayer applies an activation function to the inputs. It's separate from the fully connected layer
# to give us flexibility in assembling different network architectures.
class Activation(Layer):
    def __init__(self, activation, activation_prime=None):
        # Check if activation is a string and refers to a predefined function
        if isinstance(activation, str):
            activations = {
                'relu': (self.relu, self.relu_prime),
                'leaky_relu': (self.leaky_relu, self.leaky_relu_prime),
                'sigmoid': (self.sigmoid, self.sigmoid_prime),
                'softmax': (self.softmax, self.softmax_prime)
            }

            if activation not in activations:
                raise ValueError(f"Invalid activation function: '{activation}'. Valid options are: {list(activations.keys())}")

            self.activation, self.activation_prime = activations[activation]
        else:
            # If custom activation functions are provided
            if not callable(activation) or (activation_prime is not None and not callable(activation_prime)):
                raise ValueError("Custom activation function and its derivative must be callable.")
            
            self.activation = activation
            self.activation_prime = activation_prime

    # Activation functions and their derivatives
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_prime(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_prime(x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)

    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    @staticmethod
    def softmax_prime(x):
        # Softmax derivative is handled in the loss function due to simplification
        # when combined with cross-entropy. Therefore, this is a placeholder.
        return np.ones_like(x)

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return output_error * self.activation_prime(self.input)



class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, input_data, training=True):
        if training:
            # Create a mask that will "drop" some of the neuron outputs
            self.mask = np.random.binomial(1, 1.0 - self.rate, size=input_data.shape) / (1.0 - self.rate)
            return input_data * self.mask
        else:
            # During prediction we don't drop any units, so just return the input data
            return input_data

    def backward(self, output_error, learning_rate):
        # During backpropagation, we only backpropagate through the neurons that we kept in the forward pass
        return output_error * self.mask



class Loss:
    def __init__(self, loss_function, loss_prime=None):
        # Dictionary mapping predefined loss function names to their corresponding functions and derivatives.
        default_losses = {
            'cross_entropy': (self.cross_entropy, self.cross_entropy_prime),
            'mse': (self.mean_squared_error, self.mean_squared_error_prime),
            'dice_loss': (self.dice_loss, self.dice_loss_prime),
            'focal_loss': (self.focal_loss, self.focal_loss_prime)
        }

        # If loss_function is a string, use a predefined loss function
        if isinstance(loss_function, str):
            if loss_function not in default_losses:
                raise ValueError(f"Invalid loss function: '{loss_function}'. Valid options are: {list(default_losses.keys())}")
            self.loss, self.loss_prime = default_losses[loss_function]
        else:
            # For custom loss functions, loss_function and loss_prime should be callable
            if not callable(loss_function) or (loss_prime is not None and not callable(loss_prime)):
                raise ValueError("Custom loss function and its derivative must be callable.")
            self.loss = loss_function
            self.loss_prime = loss_prime


    # Static methods for predefined loss functions and their derivatives
    @staticmethod
    def cross_entropy(predicted, actual):
        m = actual.shape[0]
        return -np.sum(actual * np.log(predicted + 1e-15)) / m

    @staticmethod
    def cross_entropy_prime(predicted, actual):
        return predicted - actual

    @staticmethod
    def mean_squared_error(predicted, actual):
        return np.mean(np.power(predicted - actual, 2))

    @staticmethod
    def mean_squared_error_prime(predicted, actual):
        return 2 * (predicted - actual) / actual.size

    # Placeholder methods for Dice Loss and Focal Loss
    @staticmethod
    def dice_loss(predicted, actual):
        # Implement Dice Loss calculation
        pass

    @staticmethod
    def dice_loss_prime(predicted, actual):
        # Implement derivative of Dice Loss
        pass

    @staticmethod
    def focal_loss(predicted, actual):
        # Implement Focal Loss calculation
        pass

    @staticmethod
    def focal_loss_prime(predicted, actual):
        # Implement derivative of Focal Loss
        pass

    # Forward and backward methods
    def forward(self, predicted, actual):
        return self.loss(predicted, actual)

    def backward(self, predicted, actual):
        return self.loss_prime(predicted, actual)



# Network class represents the neural network itself. It holds the layers and provides functions to
# perform training (fitting to the training data) and prediction.
class Model:
    def __init__(self):
        self.layers = []  # This list will hold all layers: both fully connected and activation layers.
        self.loss = None  # This is a placeholder for the loss function.
        self.loss_prime = None  # This is a placeholder for the derivative of the loss function.
        self.training_mode = True  # By default, the network is in training mode

    # Add a layer to the network. Layers are added sequentially, so the order matters.
    def add(self, layer):
        self.layers.append(layer)

    # Set the loss and its derivative, which will be used during backpropagation.
    def set_loss(self, loss):
        self.loss = loss

    # Predict the output for a given set of inputs.
    def predict(self, input_data):
        # Set training_mode to False to disable dropout
        self.training_mode = False
        
        result = []
        for i in range(len(input_data)):
            output = input_data[i]
            for layer in self.layers:
                # Check if layer is DropoutLayer and pass the training_mode flag
                if isinstance(layer, Dropout):
                    output = layer.forward(output, training=self.training_mode)
                else:
                    output = layer.forward(output)
            result.append(output)

        # Re-enable training mode
        self.training_mode = True
        return result


    # Train the network on a given dataset.
    def fit(self, x_train, y_train, epochs, learning_rate, batch_size):
        history = {'loss': [], 'accuracy': []}

        for epoch in range(epochs):
            err = 0
            correct = 0
            for j in range(0, len(x_train), batch_size):
                x_batch = x_train[j:j + batch_size]
                y_batch = y_train[j:j + batch_size]

                # Forward propagation
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # Calculate loss and accuracy
                err += self.loss.forward(output, y_batch)
                correct += np.sum(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1))

                # Backward propagation
                error = self.loss.backward(output, y_batch)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # Store average loss and accuracy for this epoch
            history['loss'].append(err / (len(x_train) / batch_size))
            history['accuracy'].append(correct / len(x_train))

        return history


    def count_params(self):
        total_params = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                total_params += np.prod(layer.weights.shape)
            if hasattr(layer, 'bias'):
                total_params += np.prod(layer.bias.shape)
        return total_params