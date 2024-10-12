
import numpy as np

def sigmoid(n):
    return 1/(1 + np.exp(-n))

def sigmoid_d(n): 
    a = sigmoid(n)
    return (1 - a) * a

def bipolar_sigmoid(n):
    return 2 * sigmoid(n) - 1

def bipolar_sigmoid_d(n):
    a = bipolar_sigmoid(n)
    return 1/2 * (1 - a**2)

def tanh(n):
    return np.tanh(n)

def tanh_d(n):
    a = tanh(n)
    return 1 - a**2 

def relu(n):
    result = np.copy(n)
    result[result <= 0] = 0
    return result

def relu_d(n):
    result = np.copy(n)
    result[result > 0] = 1
    result[result <= 0] = 0
    return result

def linear(n):
    return n

def linear_d(n):
    return np.ones(shape = n.shape)


#define a neural network class for the mean squared error loss function
class NeuralNetwork:
    def __init__(self, input_size, layers, layer_activations, learning_rate = 0.1, initialization = 'random', hidden_layer_activations = 'sigmoid'):
        self.W = [] #weights
        self.b = [] #biases
        self.__a = [] #activations of each layer
        self.__n = [] #net input into each layer
        self.__s = [] #sensitivities of each layer
        self.__layer_sizes = [input_size] + layers #define the sizes of all layers, including the input layer
        self.__layer_activations = layer_activations #a list of the activation functions for each layer
        
        self.__activations = { #a dict of all possible activation functions and their derivatives
            'sigmoid': sigmoid,
            'sigmoid_d' : sigmoid_d,
            'linear' : linear,
            'linear_d' : linear_d,
            'relu' : relu,
            'relu_d' : relu_d,
            'bipolar_sigmoid' : bipolar_sigmoid,
            'bipolar_sigmoid_d' : bipolar_sigmoid_d,
            'tanh' : tanh,
            'tanh_d' : tanh_d
        }
        
        self.num_layers = len(self.__layer_sizes)
        self.learning_rate = learning_rate
        
        #the activation functions of the layers
        self.F = [self.__activations[layer_activation] for layer_activation in self.__layer_activations]
        #the derivative of the activation function for the layers
        self.F_d = [self.__activations[layer_activation + '_d'] for layer_activation in self.__layer_activations] 
        
        #initialize the parameters
        for i in range(self.num_layers - 1):
            if initialization == 'random':
                self.W.append(np.random.uniform(-1, 1, size = (self.__layer_sizes[i+1], self.__layer_sizes[i])))
            else:
                self.W.append(np.zeros(shape = (self.__layer_sizes[i + 1], self.__layer_sizes[i])))    

            self.b.append(np.zeros(shape = (self.__layer_sizes[i + 1], 1), dtype = np.float64))
            self.__a.append(np.zeros(shape = (self.__layer_sizes[i + 1], 1), dtype = np.float64))
            self.__n.append(np.zeros(shape = (self.__layer_sizes[i + 1], 1), dtype = np.float64))
            self.__s.append(np.zeros(shape = (self.__layer_sizes[i + 1], 1), dtype = np.float64))
        
        
    def predict(self, X):
        return self.__forward_pass(X)
        
        
    def fit(self, X, y):
        self.__forward_pass(X)
        self.__backward_pass(y)
        self.__update_weights(X)
    
    
    def __forward_pass(self, X):
        
        #first layer
        self.__n[0] = self.W[0] @ X + self.b[0]
        self.__a[0] = self.F[0](self.__n[0])
        
        #hidden layers
        for i in range(1, self.num_layers - 1):
            self.__n[i] = self.W[i] @ self.__a[i - 1] + self.b[i]
            self.__a[i] = self.F[i](self.__n[i])
        
        #output layer
        self.__n[-1] = self.W[-1] @ self.__a[-2] + self.b[-1]
        self.__a[-1] = self.F[-1](self.__n[-1])
        
        return self.__a[-1]
    
    
    def __backward_pass(self, y):
        self.__s[-1] = -2 * self.F_d[-1](self.__n[-1]) * (y - self.__a[-1]) #final layer sensitivity
        
        #remaining layers
        for i in range(self.num_layers - 3, -1, -1):
            self.__s[i] = self.F_d[i](self.__n[i]) * self.W[i + 1].T @ self.__s[i+1]

        return self.__s
    
    def __update_weights(self, X):
        
        #first layer weight update
        self.W[0] = self.W[0] - self.learning_rate * self.__s[0] @ X.T
        self.b[0] = self.b[0] - self.learning_rate * self.__s[0]    
        
        #remaining layers weight update
        for i in range(1, self.num_layers - 1):
            self.W[i] = self.W[i] - self.learning_rate * self.__s[i] @ self.__a[i - 1].T
            self.b[i] = self.b[i] - self.learning_rate * self.__s[i]    
