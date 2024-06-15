use math::{Matrix, Vector};
use serialization::Serialized;
use serialize_macro::Serialize;
use crate::downcast::DynEq;

#[derive(Default)]
pub struct Gradient {
    pub weights: Matrix,
    pub biases: Vector,
    pub output_gradient: Vector,
}

pub trait LayerName {
    fn name(&self) -> String;
    fn display(&self) -> String {
        self.name()
    }
}

pub trait Layer: LayerName + Sync + Send + Serialized + DynEq {
    fn forward(&self, input: &Vector) -> Vector;
    fn backward(&self, input: &Vector, output_gradient: Vector) -> Gradient;
    fn update(&mut self, gradient: Gradient, learning_rate: f64);
    /// TODO: this is really suboptimal but we need some consistent way to identify layers.
    fn layer_id(&self) -> usize;
}

impl PartialEq for dyn Layer {
    fn eq(&self, other: &Self) -> bool {
        self.as_dyn_eq() == other.as_dyn_eq()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Activation {
    Sigmoid,
    ReLU,
    Tanh,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Dense {
    pub weights: Matrix,
    pub biases: Vector,
}

impl LayerName for Activation {
    fn name(&self) -> String {
        format!("Activation")
    }
    fn display(&self) -> String {
        format!("{:?}", self)
    }
}

impl LayerName for Dense {
    fn name(&self) -> String {
        format!("Dense")
    }
    fn display(&self) -> String {
        format!("Dense({}x{})", self.weights.cols(), self.weights.rows())
    }
}

impl Layer for Activation {
    fn forward(&self, input: &Vector) -> Vector {
        let input = input.clone();
        // apply the activation function to each element of the input vector.
        match self {
            Activation::Sigmoid => input.map(|x| 1.0 / (1.0 + (-x).exp())),
            Activation::ReLU => input.map(|x| if x > 0.0 { x } else { 0.0 }),
            Activation::Tanh => input.map(|x| x.tanh()),
        }
    }

    fn backward(&self, input: &Vector, output_gradient: Vector) -> Gradient {
        // compute the derivative of the activation function with respect to each input.
        let gradient = match self {
            // a'(x) = a(x) * (1 - a(x))
            Activation::Sigmoid => self.forward(input).map(|x| x * (1.0 - x)),
            // a'(x) = 1 if x > 0, else 0
            Activation::ReLU => input.clone().map(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            // a'(x) = sech(x)^2 = 1 / cosh(x)^2
            Activation::Tanh => self.forward(input).map(|x| 1.0 / x.cosh().powi(-2)),
        };
        Gradient {
            output_gradient: gradient * &output_gradient,
            ..Default::default()
        }
    }

    fn update(&mut self, _gradient: Gradient, _learning_rate: f64) {}
    fn layer_id(&self) -> usize {
        1
    }
}

impl Dense {

    pub fn new(input_size: usize, output_size: usize) -> Dense {
        let weights = Matrix::new(output_size, input_size).randomize();
        let biases = Vector::new(output_size).randomize();
        Dense { weights, biases }
    }

}

impl Layer for Dense {
    fn forward(&self, input: &Vector) -> Vector {
        &self.weights * input + &self.biases
    }

    fn backward(&self, input: &Vector, output_gradient: Vector) -> Gradient {
        // note: dC/dA = cost1 -> C' for the last layer,
        // sum(dZ/dX * dA/dZ * dC/dA) for the rest.

        let d_z = output_gradient;

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
        Gradient {
            weights,
            biases,
            output_gradient: last_layer,
        }
    }

    fn update(&mut self, gradient: Gradient, learning_rate: f64) {
        self.weights -= gradient.weights * learning_rate;
        self.biases -= gradient.biases * learning_rate;
    }
    fn layer_id(&self) -> usize {
        2
    }
}
