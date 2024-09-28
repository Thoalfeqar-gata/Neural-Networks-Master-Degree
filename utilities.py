
import numpy as np

def sigmoid(n):
    return 1/(1 + np.exp(-n))

def sigmoid_d(n): 
    a = sigmoid(n)
    return (1 - a) * a

#define a neural network class for the mean squared error loss function
class NeuralNetwork:
    def __init__(self, layers, learning_rate = 0.1, initialization = 'random', hidden_layer_activations = 'sigmoid'):
        self.W = [] #weights
        self.b = [] #biases
        self.__a = [] #activations of each layer
        self.__n = [] #net input into each layer
        self.__s = [] #sensitivities of each layer
        self.activations = { #a dict of all possible activation functions for the hidden layers
            'sigmoid': sigmoid,
            'sigmoid_d' : sigmoid_d
        }
        self.num_layers = len(layers)
        self.learning_rate = learning_rate
        
        #the activation function of the hidden layers
        self.F = self.activations[hidden_layer_activations] 
        #the derivative of the activation function for the hidden layers
        self.F_d = self.activations[hidden_layer_activations + '_d'] 
        
        #initialize the parameters
        for i in range(self.num_layers - 1):
            if initialization == 'random':
                self.W.append(np.random.uniform(-1, 1, size = (layers[i+1], layers[i])))
            else:
                self.W.append(np.zeros(shape = (layers[i + 1], layers[i])))    

            self.b.append(np.zeros(shape = (layers[i + 1], 1)))
            self.__a.append(np.zeros(shape = (layers[i + 1], 1)))
            self.__n.append(np.zeros(shape = (layers[i + 1], 1)))
            self.__s.append(np.zeros(shape = (layers[i + 1], 1)))
        
        
    def predict(self, X):
        return self.__forward_pass(X)
        
        
    def fit(self, X, y):
        self.__forward_pass(X)
        self.__backward_pass(y)
        self.__update_weights(X)
    
    
    def __forward_pass(self, X):
        
        #first layer
        self.__n[0] = self.W[0] @ X + self.b[0]
        self.__a[0] = self.F(self.__n[0])
        
        #hidden layers
        for i in range(1, self.num_layers - 1):
            self.__n[i] = self.W[i] @ self.__a[i - 1] + self.b[0]
            self.__a[i] = self.F(self.__n[i])
        
        #output layer
        self.__a[-1] = self.W[-1] @ self.__a[-2] + self.b[-1]
        
        return self.__a[-1]
    
    
    def __backward_pass(self, y):
        
        self.__s[-1] = -2 * 1 * (y - self.__a[-1]) #final layer sensitivity
        
        #remaining layers
        for i in range(self.num_layers - 3, -1, -1):
            self.__s[i] = self.F_d(self.__n[i]) * self.W[i + 1].T @ self.__s[i+1]

        return self.__s
    
    def __update_weights(self, X):
        
        #first layer weight update
        self.W[0] = self.W[0] - self.learning_rate * self.__s[0] @ X.T
        self.b[0] = self.b[0] - self.learning_rate * self.__s[0]    
        
        #remaining layers weight update
        for i in range(1, self.num_layers - 1):
            self.W[i] = self.W[i] - self.learning_rate * self.__s[i] @ self.__a[i - 1].T
            self.b[i] = self.b[i] - self.learning_rate * self.__s[i]    
