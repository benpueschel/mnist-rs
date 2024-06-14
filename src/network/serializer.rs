use crate::math::{Matrix, Vector};

use super::{layer::{Activation, Dense, Layer}, Network};
use std::io::Result;

pub trait Serialized {
    fn serialize_binary(&self) -> Vec<u8>;
    fn deserialize_binary(data: &[u8]) -> (Self, usize) where Self: Sized;
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
        let num_layers = u64_from_bytes(&data[0..]) as usize;
        let mut offset = 8;
        let mut layers: Vec<Box<dyn Layer>> = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let tag = deserialize_tag(&data[offset..]);
            offset += tag.len() + 8;
            // TODO: move to proc macro which should deal with this for us (hopefully)
            let (layer, len) = deserialize_layers! {
                tag.as_str(), &data[offset..], Activation, Dense
            };
            layers.push(layer);
            offset += len;
        }
        (Network { layers }, offset)
    }
}

fn deserialize_tag(data: &[u8]) -> String {
    let len = u64_from_bytes(&data[0..]) as usize;
    String::from_utf8(data[8..8 + len].to_vec()).unwrap()
}

impl Serialized for Activation {
    fn serialize_binary(&self) -> Vec<u8> {
        match self {
            Activation::Sigmoid => 0_u64,
            Activation::ReLU => 1_u64,
            Activation::Tanh => 2_u64,
        }.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        (match u64_from_bytes(&data[0..]) {
            0 => Activation::Sigmoid,
            1 => Activation::ReLU,
            2 => Activation::Tanh,
            x => panic!("Invalid activation function {}", x),
        }, 8)
    }
}

/// Layout:
///  0-7 : number of rows (u64)
///  8-15: number of cols (u64)
/// 16-..: weights ([[f64]])
/// ..-..: number of biases (u64) (yes it's redundant, i know duh)
/// ..-..: biases  ([[f64]])
impl Serialized for Dense {
    fn serialize_binary(&self) -> Vec<u8> {
        let mut data = vec![];
        data.extend(self.weights.serialize_binary());
        data.extend(self.biases.serialize_binary());
        data
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        let (weights, weights_offset) = Matrix::deserialize_binary(&data[0..]);
        let (biases, biases_offset) = Vector::deserialize_binary(&data[weights_offset..]);
        let offset = weights_offset + biases_offset;
        (Dense { weights, biases }, offset)
    }
}

fn f64_from_bytes(bytes: &[u8]) -> f64 {
    let b = bytes;
    f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

fn u64_from_bytes(bytes: &[u8]) -> u64 {
    let b = bytes;
    u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

impl Serialized for Vector {
    fn serialize_binary(&self) -> Vec<u8> {
        let len = self.0.len();
        let mut data = Vec::with_capacity(8 + len * 8);
        data.extend(len.to_be_bytes());
        for i in 0..len {
            data.extend(self.at(i).to_be_bytes());
        }
        data
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        let len = u64_from_bytes(&data[0..]) as usize;
        let mut result = Vector::new(len);
        let mut offset = 8;
        for i in 0..len {
            result.set(i, f64_from_bytes(&data[offset..]));
            offset += 8;
        }
        (result, offset)
    }
}

impl Serialized for Matrix {
    fn serialize_binary(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(16 + self.cols() * self.rows() * 8);
        data.extend((self.rows() as u64).to_be_bytes());
        data.extend((self.cols() as u64).to_be_bytes());
        for i in 0..self.cols() {
            for j in 0..self.rows() {
                data.extend(self.at(i, j).to_be_bytes());
            }
        }
        data
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        let rows = u64_from_bytes(&data[0..]) as usize;
        let cols = u64_from_bytes(&data[8..]) as usize;
        let mut offset = 16;
        let mut result = Matrix::new(rows, cols);
        for i in 0..cols {
            for j in 0..rows {
                result.set(i, j, f64_from_bytes(&data[offset..]));
                offset += 8;
            }
        }
        (result, offset)
    }
}

#[cfg(test)]
mod test {
    use crate::network::Network;
    use crate::network::layer::{Activation, Dense};
    use crate::network;

    use super::*;

    #[test]
    pub fn test_deserialize_u64() {
        let num = rand::random::<u64>();
        assert_eq!(num, u64_from_bytes(&num.to_be_bytes()));
    }

    #[test]
    pub fn test_deserialize_f64() {
        let num = rand::random::<f64>();
        assert_eq!(num, f64_from_bytes(&num.to_be_bytes()));
    }


    #[test]
    pub fn test_dense() {
        let dense = Dense::new(12, 37);
        let serialized = dense.serialize_binary();
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
        let network = network![
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
        let network = network![
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

