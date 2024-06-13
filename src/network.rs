use std::{
    collections::VecDeque,
    fmt::{self, Debug},
};

use crate::math::{Matrix, Vector};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSpec {
    pub neurons: usize,
    pub activation: ActivationFunction,
}

impl From<(usize, ActivationFunction)> for LayerSpec {
    fn from((neurons, activation): (usize, ActivationFunction)) -> Self {
        LayerSpec {
            neurons,
            activation,
        }
    }
}

pub struct DeltaCost {
    pub weights: Matrix,
    pub biases: Vector,
    last_layer: Vector,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingData {
    pub input: Vector,
    pub target: Vector,
}

/// NOTE: Softmax is currently brokenn, your network ain't gonna learn, chief
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    Sigmoid,
    Softmax,
    ReLU,
}

impl ActivationFunction {
    pub fn activation(&self, mut x: Vector) -> Vector {
        match self {
            ActivationFunction::Sigmoid => x.map(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationFunction::ReLU => x.map(|x| x.max(0.0)),
            ActivationFunction::Softmax => {
                let d = -x[x.argmax()];
                for i in 0..x.0.len() {
                    x[i] = (x[i] + d).exp();
                }
                let sum = x.sum_values();
                x / sum
            }
        }
    }

    pub fn derivative(&self, x: Vector) -> Vector {
        match self {
            ActivationFunction::Sigmoid => {
                let s = self.activation(x);
                s.map(|x| x * (1.0 - x))
            }
            ActivationFunction::ReLU => x.map(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::Softmax => {
                let s = self.activation(x);
                let n = s.0.len();
                let mut result = Vector::new(n);
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0..n {
                        sum += if i == j {
                            s[i] * (1.0 - s[i])
                        } else {
                            -s[i] * s[j]
                        };
                    }
                    result[i] = sum;
                }
                result
            }
        }
    }
}

/// info: Matrix layout: neurons - rows, last_layer - cols
#[derive(Clone, PartialEq)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Vector,
    pub activation: ActivationFunction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub input_size: usize,
}

impl Layer {
    pub fn new(weights: Matrix, biases: Vector, activation: ActivationFunction) -> Layer {
        Layer {
            weights,
            biases,
            activation,
        }
    }

    /// Create a new layer with random weights and biases.
    /// * `neurons` Number of neurons in this layer.
    /// * `last_layer` Number of neurons in the last layer.
    pub fn random_with_size(
        neurons: usize,
        last_layer: usize,
        activation: ActivationFunction,
    ) -> Layer {
        let weights = Matrix::new(neurons, last_layer).randomize();
        let biases = Vector::new(neurons).randomize();

        Layer::new(weights, biases, activation)
    }

    /// Returns the number of neurons in this layer.
    pub fn size(&self) -> usize {
        self.biases.0.len()
    }

    pub fn forward(&self, input: &Vector) -> Vector {
        self.activation
            .activation(&self.weights * input + &self.biases)
    }

    pub fn back_propagate(&self, input: &Vector, cost1: &Vector) -> DeltaCost {
        let output = self.forward(input);
        // note: dC/dA = cost1 -> C' for the last layer,
        // sum(dZ/dX * dA/dZ * dC/dA) for the rest.

        let d_z = self.activation.derivative(output.clone()) * cost1;

        // compute cost function's derivative with respect to each bias.
        // dC/dB = dZ/dB * dA/dZ * dC/dA
        // dZ/dB = 1, less work for us :)
        let biases = d_z.clone();

        // compute cost function's derivative with respect to each weight.
        let mut weights = self.weights.clone();

        // col represents the input neuron, row represents the output neuron.
        for col in 0..weights.cols() {
            // dC/dW = dZ/dW * dA/dZ * dC/dA
            let value = input[col] * d_z.clone();
            weights[col] = value;
        }

        // compute cost function's derivative with respect to each input.
        // dC/dX = sum(dZ/dX * dA/dZ * dC/dA)
        let mut last_layer = Vector::new(input.0.len());
        for i in 0..last_layer.0.len() {
            for j in 0..self.biases.0.len() {
                last_layer[i] += self.weights[i][j] * d_z[j];
            }
        }

        DeltaCost {
            weights,
            biases,
            last_layer,
        }
    }
}

impl Debug for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.size())
    }
}

impl Network {
    /// Create a new network with random weights and biases.
    /// `layer_sizes` represents the number of neurons in each layer.
    ///
    /// # Example
    ///
    /// - Create a new network with 2 input neurons, 3 hidden neurons and 1 output neuron.
    ///
    /// ```
    /// use neural_network::Network;
    /// let network = Network::new(vec![2, 3, 1]);
    /// ```
    ///
    pub fn new(layer_spec: Vec<LayerSpec>) -> Network {
        // Go through all layers in the layer_sizes vector and connect them.
        // The first layer in layer_sizes is the input layer, so we skip it.
        let mut layers = Vec::new();
        for i in 1..layer_spec.len() {
            let layer = layer_spec[i];
            let last_layer = layer_spec[i - 1];
            layers.push(Layer::random_with_size(
                layer.neurons,
                last_layer.neurons,
                layer.activation,
            ));
        }

        Network {
            layers,
            input_size: layer_spec.first().unwrap().neurons,
        }
    }

    pub fn print_layout(&self) {
        print!("Network layout: ");
        print!("{}", self.layers[0].weights.cols());
        for layer in &self.layers {
            print!(" -> {}", layer.size());
        }
        println!();
    }

    pub fn feed_forward(&self, input: Vector) -> Vector {
        let mut result = input;
        for layer in &self.layers {
            result = layer.forward(&result);
        }
        result
    }

    pub fn back_propagate(&self, mut input: Vector, target: Vector) -> Vec<DeltaCost> {
        let mut result = VecDeque::new();
        let mut layer_inputs = Vec::new();
        for layer in &self.layers {
            layer_inputs.push(input.clone());
            input = layer.forward(&input);
        }

        let mut cost1: Vector = self.cost1(input, &target);
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let input = &layer_inputs[i];

            let delta_cost = layer.back_propagate(input, &cost1);
            cost1 = delta_cost.last_layer.clone();
            result.push_front(delta_cost);
        }

        result.into()
    }

    pub fn train_parallel(
        &mut self,
        data: &[TrainingData],
        learning_rate: f64,
        thread_count: usize,
    ) {
        //TODO: does this work when data.len() % thread_count != 0?
        let block_size = data.len() / thread_count;

        let mut thread_deltas = Vec::new();
        std::thread::scope(|s| {
            let mut threads = Vec::with_capacity(thread_count);
            for i in 0..thread_count {
                let this = &self;

                threads.push(s.spawn(move || {
                    let start = i * block_size;
                    let end = start + block_size;
                    this.calc_deltas(&data[start..end], learning_rate)
                }));
            }

            for thread in threads {
                thread_deltas.push(thread.join().unwrap());
            }
        });

        for deltas in thread_deltas {
            for (layer, delta) in self.layers.iter_mut().zip(deltas) {
                layer.weights -= delta.weights / thread_count as f64;
                layer.biases -= delta.biases / thread_count as f64;
            }
        }
    }

    pub fn train(&mut self, data: &[TrainingData], learning_rate: f64) {
        let deltas = self.calc_deltas(data, learning_rate);
        for (layer, delta) in self.layers.iter_mut().zip(deltas) {
            layer.weights -= &delta.weights;
            layer.biases -= &delta.biases;
        }
    }

    fn calc_deltas(&self, data: &[TrainingData], learning_rate: f64) -> Vec<DeltaCost> {
        let mut deltas = Vec::new();
        let data_len = data.len() as f64;
        for training_data in data {
            if deltas.is_empty() {
                deltas =
                    self.back_propagate(training_data.input.clone(), training_data.target.clone());
            } else {
                let mut new_deltas =
                    self.back_propagate(training_data.input.clone(), training_data.target.clone());
                for (new_delta, delta) in new_deltas.iter_mut().zip(&mut deltas) {
                    new_delta.weights *= learning_rate / data_len;
                    new_delta.biases *= learning_rate / data_len;
                    delta.weights += &new_delta.weights;
                    delta.biases += &new_delta.biases;
                }
            }
        }
        deltas
    }

    /// Evaluate the cost of one output compared to the expected output.
    /// The average cost function results across a dataset can be used to evaluate the network's
    /// performance. The cost function is defined as: C = (output - expected)^2.
    ///
    /// # Example
    ///
    /// - Simulate a network output of 0.5 and compare it to the expected output of 0.0.
    ///   The cost function is (0.5 - 0.0)^2 = 0.25.
    ///
    /// ```
    /// use neural_network::{Network, Vector};
    /// let network = Network::new(vec![2, 3, 1]);
    ///
    /// let simulated_output = vec![0.5].into();
    /// let expected_output = vec![0.0].into();
    ///
    /// let cost = network.cost(simulated_output, expected_output);
    /// assert_eq!(cost, 0.25);
    /// ```
    ///
    pub fn cost(&self, output: &Vector, expected: &Vector) -> f64 {
        let mut result = 0.0;
        for (o, e) in output.0.iter().zip(expected.0.iter()) {
            result += (o - e) * (o - e) / output.0.len() as f64;
        }
        result
    }

    pub fn cost1(&self, mut output: Vector, target: &Vector) -> Vector {
        assert_eq!(target.0.len(), output.0.len());

        for (i, t) in target.0.iter().enumerate() {
            output.0[i] = 2.0 * (output.0[i] - t) / output.0.len() as f64;
        }
        output
    }

}
