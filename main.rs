/*
// do cargo later pls
[dependencies]
ndarray = "0.15.3"
ndarray-rand = "0.15.0"
rand = "0.8.4"
*/

use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::f64::consts::E;

struct NeuralNetwork {
    synaptic_weights: Array2<f64>,
}

impl NeuralNetwork {
    fn new() -> NeuralNetwork {
        let mut rng = rand::thread_rng();
        let weights = Array2::random_using((3, 1), Uniform::new(-1.0, 1.0), &mut rng);
        NeuralNetwork {
            synaptic_weights: weights,
        }
    }

    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    fn sigmoid_derivative(&self, x: f64) -> f64 {
        x * (1.0 - x)
    }

    fn train(&mut self, training_set_inputs: &Array2<f64>, training_set_outputs: &Array2<f64>, n: usize) {
        for _ in 0..n {
            let o = self.think(training_set_inputs);
            let e = training_set_outputs - &o;
            let a = training_set_inputs.t().dot(&(e * &self.__sigmoid_derivative(&o)));
            self.synaptic_weights += &a;
        }
    }

    fn think(&self, inputs: &Array1<f64>) -> Array1<f64> {
        self.__sigmoid(inputs.dot(&self.synaptic_weights))
    }

    fn __sigmoid(&self, x: Array1<f64>) -> Array1<f64> {
        x.mapv(|val| self.sigmoid(val))
    }

    fn __sigmoid_derivative(&self, x: Array1<f64>) -> Array1<f64> {
        x * &(1.0 - &x)
    }
}

fn main() {
    let mut nn = NeuralNetwork::new();

    println!("Random starting synaptic weight:");
    println!("{:?}", nn.synaptic_weights);

    let training_set_input = array![[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
    let training_set_output = array![[0.0, 1.0, 1.0, 0.0, 1.0]].reversed_axes();

    nn.train(&training_set_input, &training_set_output, 10000);

    println!("New synaptic weights:");
    println!("{:?}", nn.synaptic_weights);

    let input = array![1.0, 0.0, 1.0];
    println!("Under New situation [1, 0, 1]:");
    println!("{:?}", nn.think(&input));
}
