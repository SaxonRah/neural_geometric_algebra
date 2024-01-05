import numpy as np
import clifford as cf

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NEURAL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NODE
class NeuralNode:
    def __init__(self, layer_type='feedforward', **kwargs):
        self.layer_type = layer_type
        self.input = None
        self.output = None

        if self.layer_type == 'feedforward':
            self.layer = FeedforwardLayer(**kwargs)
        elif self.layer_type == 'recurrent':
            self.layer = RecurrentLayer(**kwargs)
        elif self.layer_type == 'convolutional':
            self.layer = ConvolutionalLayer(**kwargs)
        elif self.layer_type == 'geometric_algebra':
            self.layer = GeometricAlgebraNodeLayer(**kwargs)
        else:
            raise ValueError("Unsupported layer type")

    def feedforward(self, input):
        self.input = input
        self.output = self.layer.feedforward(input)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FEEDFORWARD
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LAYER
# Feedforward layer
class FeedforwardLayer:
    def __init__(self, weight=0.0):
        self.weight = weight

    def feedforward(self, input):
        return input * self.weight

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RECURRENT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LAYER
# Recurrent layer
class RecurrentLayer:
    def __init__(self, weight=0.0, recurrent_weight=0.0):
        self.weight = weight
        self.recurrent_weight = recurrent_weight
        self.hidden_state = 0.0

    def feedforward(self, input):
        self.hidden_state = self.hidden_state * self.recurrent_weight + input * self.weight
        return self.hidden_state

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CONVOLUTIONAL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LAYER
# Convolutional layer
class ConvolutionalLayer:
    def __init__(self, filter_size=3, weight_initializer=None):
        self.filter_size = filter_size
        self.weight_initializer = weight_initializer
        self.filters = self.initialize_filters()
        self.input = None
        self.output = None

    def initialize_filters(self):
        if self.weight_initializer is None:
            # Use an identity filter for simplicity
            return np.eye(self.filter_size)
        else:
            return self.weight_initializer((self.filter_size, self.filter_size))

    def feedforward(self, input):
        self.input = input
        height, width = input.shape
        output = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1))

        for i in range(height - self.filter_size + 1):
            for j in range(width - self.filter_size + 1):
                receptive_field = input[i:i+self.filter_size, j:j+self.filter_size]
                output[i, j] = np.sum(receptive_field * self.filters)

        self.output = output
        return output

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GEOMETRIC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LAYER
# Geometric algebra layer
class GeometricAlgebraNodeLayer:
    def __init__(self, weight=0.0):
        # Create a 2D geometric algebra layout
        layout, blades = cf.Cl(2)
        self.layout = layout
        # Extend the weight to a 4-dimensional sequence for the 2D layout
        self.weight = cf.MultiVector(value=[weight, 0, 0, 0], layout=layout)
        self.input = 0.0
        self.output = 0.0

    def feedforward(self, input):
        self.input = cf.MultiVector(value=[input, 0, 0, 0], layout=self.layout)
        # Geometric algebra multiplication
        self.output = self.weight * self.input
        return self.output.value[0]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FEEDFORWARD
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TEST
# Create and initialize a single NeuralNode with a feedforward layer
neuron_feedforward = NeuralNode(layer_type='feedforward', weight=1.0)
x = 3.0
neuron_feedforward.feedforward(x)
print("The output of the feedforward neural node is:", neuron_feedforward.output)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RECURRENT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TEST
# Create and initialize a single NeuralNode with a recurrent layer
neuron_recurrent = NeuralNode(layer_type='recurrent', weight=0.5, recurrent_weight=0.1)
sequence = [1.0, 2.0, 3.0]
for x in sequence:
    neuron_recurrent.feedforward(x)
    print("Output after processing input", x, "is:", neuron_recurrent.output)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CONVOLUTIONAL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TEST
# Create and initialize a single NeuralNode with a convolutional layer
neuron_convolutional = NeuralNode(layer_type='convolutional', filter_size=3)
example_input = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])
neuron_convolutional.feedforward(example_input)
print("The output of the convolutional neural node is:\n", neuron_convolutional.output)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GEOMETRIC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TEST
# Create and initialize a single NeuralNode with a geometric algebra layer
ga_neuron = NeuralNode(layer_type='geometric_algebra', weight=1.0)
x = 3.0
ga_neuron.feedforward(x)
print("The output of the geometric algebra neural node is:", ga_neuron.output)
