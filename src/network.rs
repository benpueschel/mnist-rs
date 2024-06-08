use std::{
    collections::VecDeque, fmt::{self, Debug}, sync::{Arc, Mutex}
};

use crate::math::{self, Matrix, Vector};

pub struct DeltaCost {
    pub weights: Matrix,
    pub biases: Vector,
    last_layer: Vector,
}

pub struct TrainingData {
    pub input: Vector,
    pub target: Vector,
}

pub trait ActivationFunction {
    fn activation(x: f64) -> f64;
    fn derivative(x: f64) -> f64;
}

pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn activation(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(x: f64) -> f64 {
        let s = Self::activation(x);
        s * (1.0 - s)
    }
}

pub struct Layer {
    weights: Matrix,
    biases: Vector,
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Layer {
    pub fn new(weights: Matrix, biases: Vector) -> Layer {
        Layer { weights, biases }
    }

    /// Create a new layer with random weights and biases.
    /// * `neurons` Number of neurons in this layer.
    /// * `last_layer` Number of neurons in the last layer.
    pub fn random_with_size(neurons: usize, last_layer: usize) -> Layer {
        let weights = Matrix::new(neurons, last_layer).randomize();
        let biases = Vector::new(neurons).randomize();

        Layer::new(weights, biases)
    }

    /// Returns the number of neurons in this layer.
    pub fn size(&self) -> usize {
        self.biases.0.len()
    }

    pub fn forward(&self, input: &Vector) -> Vector {
        ((&self.weights * input) + &self.biases).map(math::sigmoid)
    }

    pub fn cost(&self, input: &Vector, target: &Vector) -> Vector {
        assert_eq!(target.0.len(), self.biases.0.len());

        let mut result = self.forward(input);

        for (i, t) in target.0.iter().enumerate() {
            let diff = result.0[i] - t;
            result.0[i] = diff * diff;
        }
        result
    }

    pub fn cost1(&self, input: &Vector, target: &Vector) -> Vector {
        assert_eq!(target.0.len(), self.biases.0.len());

        let mut result = self.forward(input);

        for (i, t) in target.0.iter().enumerate() {
            result.0[i] = 2.0 * (result.0[i] - t);
        }
        result
    }

    pub fn back_propagate(&self, input: &Vector, target: &Vector) -> DeltaCost {
        let cost1 = self.cost1(input, target);
        let values = self.forward(input);

        // compute cost function's derivative with respect to each weight.
        let mut weights = self.weights.clone();
        for col in 0..weights.cols() {
            for row in 0..weights.rows() {
                // dC/dW = dC/dA * dA/dZ * dZ/dW
                let value = input.0[col] * math::sigmoid1(values.0[row]) * cost1.0[row];
                weights.set(col, row, value);
            }
        }

        // compute cost function's derivative with respect to each bias.
        let mut biases = self.biases.clone();
        for i in 0..biases.0.len() {
            // dC/dB = dC/dA * dA/dZ * dZ/dB
            // dZ/dB = 1, less work for us :)
            biases.0[i] = math::sigmoid1(values.0[i]) * cost1.0[i];
        }

        // compute cost function's derivative with respect to each input.
        let mut last_layer = Vector::new(input.0.len());
        for i in 0..last_layer.0.len() {
            // dC/dX = sum(dZ/dX * dA/dZ * dC/dA)
            for j in 0..weights.rows() {
                last_layer.0[i] += weights.at(i, j) * math::sigmoid1(values.0[j]) * cost1.0[j];
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
    pub fn new(layer_sizes: Vec<usize>) -> Network {
        // Go through all layers in the layer_sizes vector and connect them.
        // The first layer in layer_sizes is the input layer, so we skip it.
        let mut layers = Vec::new();
        for i in 1..layer_sizes.len() {
            println!("Creating layer {} with {} neurons", i, layer_sizes[i]);
            layers.push(Layer::random_with_size(layer_sizes[i], layer_sizes[i - 1]));
        }

        println!("{:?}", layers);

        Network { layers }
    }

    pub fn feed_forward(&self, input: Vector) -> Vector {
        let mut result = input;
        for layer in &self.layers {
            result = layer.forward(&result);
        }
        result
    }

    pub fn back_propagate(&self, mut input: Vector, mut target: Vector) -> Vec<DeltaCost> {
        let mut result = VecDeque::new();
        let mut layer_inputs = Vec::new();
        for layer in &self.layers {
            layer_inputs.push(input.clone());
            input = layer.forward(&input);
        }

        for (i, layer) in self.layers.iter().enumerate().rev() {
            let input = &layer_inputs[i];
            let delta_cost = layer.back_propagate(input, &target);
            target = delta_cost.last_layer.clone();
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
        let block_size = data.len() / thread_count;

        let mut deltas = Vec::new();
        let i = Arc::new(Mutex::new(0));
        std::thread::scope(|s| {
            let mut threads = Vec::with_capacity(thread_count);
            for _ in 0..thread_count {
                //TODO: does this work when data.len() % thread_count != 0?
                threads.push(s.spawn(|| {
                    let mut i = i.lock().unwrap();
                    let start = *i * block_size;
                    let end = start + block_size;
                    println!("Thread {} processing data[{}..{}]", *i, start, end);
                    *i += 1;
                    self.calc_deltas(&data[start..end], learning_rate)
                }));
            }

            for thread in threads {
                deltas.extend(thread.join().unwrap());
            }
        });

        for (layer, delta) in self.layers.iter_mut().zip(&deltas) {
            layer.weights -= &delta.weights;
            layer.biases -= &delta.biases;
        }
    }

    pub fn train(&mut self, data: &[TrainingData], learning_rate: f64) {
        let mut deltas = self.calc_deltas(data, learning_rate);
        for (layer, delta) in self.layers.iter_mut().zip(&mut deltas) {
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
            result += (o - e) * (o - e);
        }
        result
    }
}
