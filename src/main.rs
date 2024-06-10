use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
};

use math::Vector;
use network::{Network, TrainingData};

use crate::screen::ScreenInfo;

pub mod math;
pub mod network;
pub mod screen;
pub mod serializer;

pub mod mnist;

static PEEK_COUNT: usize = 5;
static BATCHES: usize = 20;
static THREAD_COUNT: usize = 10;
static LEARNING_RATE: f64 = 0.01;
static NETWORK_PATH: &str = "network.ben";

fn main() {
    let exit = Arc::new(Mutex::new(false));
    let exit_clone = exit.clone();

    ctrlc::set_handler(move || {
        let mut exit = exit_clone.lock().unwrap();
        *exit = true;
    })
    .expect("Error setting Ctrl-C handler");

    println!("Loading MNIST dataset...");
    let (train_set, test_set) = mnist::load_datasets("data").unwrap();
    let image_size = train_set.image_size.0 * train_set.image_size.1;

    let mut network = load_network(image_size);
    print!("Network layout: ");
    network.print_layout();
    println!();

    let mut training_data = Vec::from(&train_set);
    rand::seq::SliceRandom::shuffle(training_data.as_mut_slice(), &mut rand::thread_rng());

    let mut test_data = Vec::from(&test_set);
    rand::seq::SliceRandom::shuffle(test_data.as_mut_slice(), &mut rand::thread_rng());

    let start = std::time::Instant::now();
    let mut screen_info = ScreenInfo::default();
    screen::clear_screen();

    for i in 0.. {
        screen_info.epoch = i;
        screen_info.total_batches = BATCHES;
        screen_info.batch = 0;
        screen_info.status = "Testing...";
        screen::display_info(&screen_info);

        let stats = test_network(&network, &test_data, true);
        screen_info.test_accuracy = stats.accuracy;
        screen_info.test_loss = stats.avg_cost;
        screen_info.test_confidence = stats.confidence;
        screen_info.error_least_confident = stats.error_least_confident;
        screen_info.error_least_confident_data = stats.error_least_confident_data;
        screen::display_info(&screen_info);
        screen_info.status = "Training...";

        for i in 0..BATCHES {
            screen_info.batch = i + 1;
            screen_info.elapsed = start.elapsed();

            let batch_size = training_data.len() / BATCHES;
            let start = i * batch_size;
            let end = start + batch_size;

            network.train_parallel(&training_data[start..end], LEARNING_RATE, THREAD_COUNT);
            // network.train(&training_data[start..end], LEARNING_RATE);

            screen::display_info(&screen_info);

            if *exit.lock().unwrap() {
                // screen::move_cursor();

                serializer::serialize(&network, NETWORK_PATH).unwrap();

                assert_eq!(serializer::deserialize(NETWORK_PATH).unwrap(), network);

                println!("Network saved to network.ben");
                println!("Exiting...");
                return;
            }
        }
    }
}

pub fn load_network(image_size: usize) -> Network {
    use network::ActivationFunction::*;
    let network;

    loop {
        print!("Create new network? (y/n): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim().to_lowercase();

        if input == "n" {
            network = serializer::deserialize(NETWORK_PATH).unwrap();
            println!("Loaded network from network.ben");
            break;
        } else if input == "y" {
            let network_layout = vec![
                (image_size, ReLU).into(),
                // (80, ReLU).into(),
                // (20, Sigmoid).into(),
                // (20, Sigmoid).into(),
                (20, ReLU).into(),
                (20, ReLU).into(),
                (10, Sigmoid).into(),
            ];
            network = network::Network::new(network_layout.clone());
            println!("Created new network.");
            break;
        }
    }

    network
}

pub struct TestResult {
    pub avg_cost: f64,
    pub accuracy: f64,
    pub confidence: f64,
    pub error_least_confident: Option<(Vector, f64)>,
    pub error_least_confident_data: Option<TrainingData>,
}

pub fn test_network(network: &Network, dataset: &Vec<TrainingData>, peek: bool) -> TestResult {
    let mut avg_cost = 0.0;
    let mut accuracy = 0.0;
    let mut confidence = 0.0;
    let mut error_least_confident = None;
    let mut error_least_confident_data = None;

    let mut peek_count = 0;
    for data in dataset.iter() {
        let output = network.feed_forward(data.input.clone());
        let predicted = output.argmax();
        let current_cost = network.cost(&output, &data.target);
        let current_confidence = math::softmax(output.clone())[predicted];

        if predicted == data.target.argmax() {
            accuracy += 1.0 / dataset.len() as f64;
        } else if peek && peek_count < PEEK_COUNT {
            match error_least_confident {
                Some((_, confidence)) if confidence < current_confidence => (),
                _ => {
                    error_least_confident = Some((output.clone(), current_confidence));
                    error_least_confident_data = Some(data.clone());
                }
            }
            peek_count += 1;
        }

        avg_cost += current_cost / dataset.len() as f64;
        confidence += current_confidence / dataset.len() as f64;
    }

    TestResult {
        avg_cost,
        accuracy,
        confidence,
        error_least_confident,
        error_least_confident_data,
    }
}
