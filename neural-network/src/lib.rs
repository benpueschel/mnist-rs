pub mod downcast;
pub mod layer;

pub mod mnist;

use std::{
    collections::VecDeque,
    fmt::{self, Debug}, io::Result,
};

use crate::downcast::DynEq;
use math::Vector;
use serialization::Serialized;

use self::layer::{Gradient, Layer};


#[derive(Debug, Clone, PartialEq)]
pub struct TrainingData {
    pub input: Vector,
    pub target: Vector,
}

#[macro_export]
macro_rules! create_network {
    ($($x:expr),+ $(,)?) => {
        {
            use $crate::layer::Layer;
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
    /// Create a new network with the given layers.
    /// NOTE: You probably shouldn't use this directly. 
    /// Use the [`create_network!`] macro instead.
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


pub fn serialize_network(network: &Network, path: &str) -> Result<()> {
    std::fs::write(path, network.serialize_binary())
}

pub fn deserialize_network(path: &str) -> Result<Network> {
    let data = std::fs::read(path)?;
    Ok(Network::deserialize_binary(&data).0)
}

macro_rules! deserialize_layers {
    { $tag:expr, $data:expr,$($layer:ty),+ } => {
        match $tag {
            $(
                stringify!($layer) => {
                    let (layer, len) = <$layer>::deserialize_binary($data);
                    (Box::new(layer) as Box<dyn Layer>, len)
                },
            )+
            x => panic!("Invalid layer tag {}", x),
        }
    };
}

impl Serialized for Network {
    fn serialize_binary(&self) -> Vec<u8> {
        let mut data = vec![];
        data.extend((self.layers.len() as u64).to_be_bytes());

        for layer in &self.layers {
            let tag = layer.name().as_bytes().to_vec();
            data.extend((tag.len() as u64).to_be_bytes());
            data.extend(tag);
            data.extend(layer.serialize_binary());
        }

        data
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        let num_layers = serialization::u64_from_bytes(&data[0..]) as usize;
        let mut offset = 8;
        let mut layers: Vec<Box<dyn Layer>> = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let tag = deserialize_tag(&data[offset..]);
            offset += tag.len() + 8;
            use layer::{Activation, Dense};
            // TODO: move to proc macro which should deal with this for us (hopefully)
            let (layer, len) = deserialize_layers! {
                tag.as_str(), &data[offset..], Activation, Dense
            };
            layers.push(layer);
            offset += len;
        }
        (Network { layers }, offset)
    }
    fn tag() -> &'static str {
        "Network"
    }
}

pub fn deserialize_tag(data: &[u8]) -> String {
    let len = serialization::u64_from_bytes(&data[0..]) as usize;
    String::from_utf8(data[8..8 + len].to_vec()).unwrap()
}

#[cfg(test)]
mod test {
    use super::layer::{Activation, Dense};
    use super::*;

    #[test]
    pub fn test_dense() {
        let dense = Dense::new(12, 37);
        let serialized = dense.serialize_binary();
        println!("{}", serialized.len());
        let (deserialized, _) = Dense::deserialize_binary(&serialized);
        assert_eq!(dense, deserialized);
    }

    #[test]
    pub fn test_activation() {
        let activation = Activation::ReLU;
        let serialized = activation.serialize_binary();
        let (deserialized, _) = Activation::deserialize_binary(&serialized);
        assert_eq!(activation, deserialized);
    }

    #[test]
    pub fn test_network() {
        let network = create_network![
            Dense::new(12, 37),
            Activation::ReLU,
            Dense::new(37, 20),
            Activation::Sigmoid,
        ];
        let serialized = network.serialize_binary();
        let (deserialized, _) = Network::deserialize_binary(&serialized);
        assert_eq!(network, deserialized);
    }

    #[test]
    pub fn test_serialization() {
        let network = create_network![
            Dense::new(12, 37),
            Activation::ReLU,
            Dense::new(37, 20),
            Activation::Sigmoid,
        ];
        static PATH: &str = "test";

        serialize_network(&network, PATH).expect("Failed to serialize network");
        let deserialized = deserialize_network(PATH).expect("Failed to deserialize network");
        std::fs::remove_file(PATH).expect("Failed to remove test file");
        assert_eq!(network, deserialized);
    }
}