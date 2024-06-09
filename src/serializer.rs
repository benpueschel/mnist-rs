use std::io::Result;

use crate::{
    math::{Matrix, Vector},
    network::{ActivationFunction, Layer, Network},
};

/// Binary format:
/// - 0-3 ([u32]): magic number
/// - 4-7 ([u32]): number of layers
/// - 8-11 ([u32]): number of input neurons
/// - ... ([[u32]]): number of neurons per layer
/// - ... ([[u32]]): activation function per layer
/// - ... ([[f64]]): biases
/// - ... ([[f64]]): weights

static MAGIC_NUMBER: [u8; 4] = 0x12092004_u32.to_be_bytes();

pub fn serialize(network: &Network, path: &str) -> Result<()> {
    let layers = (network.layers.len() as u32).to_be_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&MAGIC_NUMBER);
    data.extend_from_slice(&layers);
    data.extend_from_slice(&(network.input_size as u32).to_be_bytes());

    for layer in &network.layers {
        let neurons = (layer.size() as u32).to_be_bytes();
        data.extend_from_slice(&neurons);
    }

    for layer in &network.layers {
        let activation = serialize_activation(layer.activation);
        data.extend_from_slice(&activation.to_be_bytes());
    }

    data.extend(serialize_biases(network).iter());
    data.extend(serialize_weights(network).iter());

    std::fs::write(path, data)
}

pub fn deserialize(path: &str) -> Result<Network> {
    let data = std::fs::read(path)?;

    assert_eq!(&data[0..4], &MAGIC_NUMBER, "Invalid magic number");

    let num_layers = u32_from_bytes(&data[4..8]) as usize;
    let input_neurons = u32_from_bytes(&data[8..12]);
    let mut offset = 12;

    let mut layer_neurons = Vec::with_capacity(num_layers);
    let mut activation_functions = Vec::with_capacity(num_layers);

    for _ in 0..num_layers {
        let neurons = u32_from_bytes(&data[offset..offset + 4]);
        layer_neurons.push(neurons);
        offset += 4;
    }

    for _ in 0..num_layers {
        let serialized = u32_from_bytes(&data[offset..offset + 4]);
        activation_functions.push(deserialize_activation(serialized));
        offset += 4;
    }

    let (biases, bias_offset) = deserialize_biases(&data[offset..], layer_neurons.as_slice());
    offset += bias_offset;

    let (weights, _) =
        deserialize_weights(&data[offset..], input_neurons, layer_neurons.as_slice());

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        layers.push(Layer::new(
            weights[i].clone(),
            biases[i].clone(),
            activation_functions[i],
        ));
    }

    Ok(Network {
        layers,
        input_size: input_neurons as usize,
    })
}

fn serialize_biases(network: &Network) -> Vec<u8> {
    let mut data = Vec::new();
    for layer in &network.layers {
        data.extend(layer.biases.0.iter().map(|x| x.to_be_bytes()).flatten());
    }
    data
}

fn deserialize_biases(data: &[u8], layers: &[u32]) -> (Vec<Vector>, usize) {
    let mut biases = Vec::with_capacity(layers.len());
    let mut offset = 0;
    for layer in layers {
        let layer = *layer as usize;

        let mut layer_biases = Vector::new(layer);
        for i in 0..layer {
            layer_biases[i] = f64_from_bytes(&data[offset..offset + 8]);
            offset += 8;
        }
        biases.push(layer_biases);
    }
    (biases, offset)
}

fn serialize_weights(network: &Network) -> Vec<u8> {
    let mut data = Vec::new();
    for layer in &network.layers {
        let weights = &layer.weights;
        for i in 0..weights.cols() {
            for j in 0..weights.rows() {
                data.extend(weights[i][j].to_be_bytes());
            }
        }
    }
    data
}

fn deserialize_weights(data: &[u8], input_neurons: u32, layers: &[u32]) -> (Vec<Matrix>, usize) {
    let mut weights = Vec::with_capacity(layers.len());
    let mut last_layer = input_neurons as usize;
    let mut offset = 0;
    for layer in layers {
        let layer = *layer as usize;

        let mut layer_weights = Matrix::new(layer, last_layer);
        for j in 0..last_layer {
            for i in 0..layer {
                layer_weights[j][i] = f64_from_bytes(&data[offset..offset + 8]);
                offset += 8;
            }
        }
        weights.push(layer_weights);

        last_layer = layer;
    }
    (weights, offset)
}

fn serialize_activation(activation: ActivationFunction) -> u32 {
    match activation {
        ActivationFunction::Sigmoid => 0,
        ActivationFunction::ReLU => 1,
        ActivationFunction::Softmax => 2,
    }
}

fn deserialize_activation(activation: u32) -> ActivationFunction {
    match activation {
        0 => ActivationFunction::Sigmoid,
        1 => ActivationFunction::ReLU,
        2 => ActivationFunction::Softmax,
        x => panic!("Invalid activation function {}", x),
    }
}

fn u32_from_bytes(data: &[u8]) -> u32 {
    u32::from_be_bytes([data[0], data[1], data[2], data[3]])
}

fn f64_from_bytes(data: &[u8]) -> f64 {
    f64::from_be_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ])
}

#[cfg(test)]
mod test {
    use crate::network::ActivationFunction::*;
    use crate::network::LayerSpec;
    use crate::network::Network;

    use super::*;

    static LAYOUT: [LayerSpec; 4] = [
        LayerSpec {
            neurons: 3,
            activation: Sigmoid,
        },
        LayerSpec {
            neurons: 14,
            activation: ReLU,
        },
        LayerSpec {
            neurons: 8,
            activation: Softmax,
        },
        LayerSpec {
            neurons: 7,
            activation: ReLU,
        },
    ];

    fn num_neurons() -> Vec<u32> {
        LAYOUT[1..]
            .iter()
            .map(|layer| layer.neurons as u32)
            .collect::<Vec<_>>()
    }

    fn input_neurons() -> u32 {
        LAYOUT[0].neurons as u32
    }

    #[test]
    pub fn test_deserialize_u32() {
        let num = rand::random::<u32>();
        assert_eq!(num, u32_from_bytes(&num.to_be_bytes()));
    }

    #[test]
    pub fn test_deserialize_f64() {
        let num = rand::random::<f64>();
        assert_eq!(num, f64_from_bytes(&num.to_be_bytes()));
    }

    #[test]
    pub fn test_biases() {
        let network = Network::new(LAYOUT.to_vec());
        let data = serialize_biases(&network);
        let (biases, _) = deserialize_biases(&data, &num_neurons());

        let biases_should = network
            .layers
            .iter()
            .map(|layer| layer.biases.clone())
            .collect::<Vec<_>>();
        assert_eq!(biases.len(), biases_should.len());
        assert_eq!(biases, biases_should);
    }

    #[test]
    pub fn test_weights() {
        let network = Network::new(LAYOUT.to_vec());
        let data = serialize_weights(&network);
        let (weights, _) = deserialize_weights(&data, input_neurons(), &num_neurons());

        let weights_should = network
            .layers
            .iter()
            .map(|layer| layer.weights.clone())
            .collect::<Vec<_>>();

        assert_eq!(weights.len(), weights_should.len());
        assert_eq!(weights, weights_should);
    }

    #[test]
    pub fn test_activation() {
        let network = Network::new(LAYOUT.to_vec());
        for layer in &network.layers {
            let activation = serialize_activation(layer.activation);
            let deserialized = deserialize_activation(activation);
            assert_eq!(layer.activation, deserialized);
        }
    }

    #[test]
    pub fn test_serialization() {
        let network = Network::new(LAYOUT.to_vec());
        static PATH: &str = "test";

        serialize(&network, PATH).expect("Failed to serialize network");
        let deserialized = deserialize(PATH).expect("Failed to deserialize network");
        std::fs::remove_file(PATH).expect("Failed to remove test file");
        assert_eq!(network, deserialized);
    }
}
