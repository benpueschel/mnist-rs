use std::{
    collections::VecDeque,
    fmt::{self, Debug},
};

use crate::{downcast::DynEq, math::Vector};

use self::layer::{Gradient, Layer};

pub mod layer;
pub mod serializer;

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingData {
    pub input: Vector,
    pub target: Vector,
}

#[macro_export]
macro_rules! network {
    ($($x:expr),+ $(,)?) => {
        {
            use crate::network::layer::Layer;
            let layers: Vec<Box<dyn Layer>> = vec![
                $(Box::new($x),)+
            ];
            Network::new(layers)
        }
    };
}

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Debug for dyn Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Layer")
            .field("name", &self.display())
            .finish()
    }
}

impl PartialEq for Network {
    fn eq(&self, other: &Self) -> bool {
        if self.layers.len() != other.layers.len() {
            return false;
        }

        for (a, b) in self.layers.iter().zip(&other.layers) {
            if *(a as &dyn DynEq) != *(b as &dyn DynEq) {
                return false;
            }
        }

        true
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
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Network {
        Network { layers }
    }

    pub fn print_layout(&self) {
        print!("Network layout: ");
        for i in 0..self.layers.len() - 1 {
            let layer = &self.layers[i];
            print!("{} -> ", layer.display());
        }
        match self.layers.last() {
            Some(layer) => println!("{}", layer.display()),
            None => println!(),
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

    pub fn back_propagate(&self, mut input: Vector, target: Vector) -> Vec<Gradient> {
        let mut result = VecDeque::new();
        let mut layer_inputs = Vec::new();
        for layer in &self.layers {
            layer_inputs.push(input.clone());
            input = layer.forward(&input);
        }

        let mut cost1: Vector = self.cost1(input, &target);
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let input = &layer_inputs[i];

            let gradient = layer.backward(input, cost1);
            cost1 = gradient.output_gradient.clone();
            result.push_front(gradient);
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
                    this.calc_gradients(&data[start..end], learning_rate)
                }));
            }

            for thread in threads {
                thread_deltas.push(thread.join().unwrap());
            }
        });

        for gradients in thread_deltas {
            for (layer, gradient) in self.layers.iter_mut().zip(gradients) {
                layer.update(gradient, learning_rate / thread_count as f64);
            }
        }
    }

    pub fn train(&mut self, data: &[TrainingData], learning_rate: f64) {
        let gradients = self.calc_gradients(data, learning_rate);
        for (layer, gradient) in self.layers.iter_mut().zip(gradients) {
            layer.update(gradient, learning_rate);
        }
    }

    fn calc_gradients(&self, data: &[TrainingData], learning_rate: f64) -> Vec<Gradient> {
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
