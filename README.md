# neural_geometric_algebra
A basic Python example of a neural network node with feedforward, recurrent, convolutional, and geometric layers.

```python
neuron_feedforward = NeuralNode(layer_type='feedforward', weight=1.0)
neuron_recurrent = NeuralNode(layer_type='recurrent', weight=0.5, recurrent_weight=0.1)
neuron_convolutional = NeuralNode(layer_type='convolutional', filter_size=3)
ga_neuron = NeuralNode(layer_type='geometric_algebra', weight=1.0)
```

In the NeuralNode class, the feedforward method is a generic method that sets the input, calls the layer-specific feedforward method, and stores the output.
Each layer class overrides the feedforward method from the base NeuralNode class, providing a specific implementation tailored to the characteristics of the respective layer type.
