pub mod math;
pub mod network;

pub mod mnist;

fn main() {
    println!("Hello, world!");

    let (train, test) = mnist::load_datasets("data").unwrap();
    let image_size = train.image_size.0 * train.image_size.1;

    let mut network = network::Network::new(vec![image_size, 16, 16, 10]);
    println!("Network created");
    print!("Network layout: ");
    print!("{}", image_size);
    for layer in &network.layers {
        print!(" -> {}", layer.size());
    }
    println!();

    let training_data = Vec::from(&train);
    let mut iter = 0;
    loop {
        println!("Iteration: {}", iter);
        println!("Testing network...");
        let mut avg_cost = 0.0;
        let mut accuracy = 0.0;
        let mut confidence = 0.0;
        let test = Vec::from(&test);
        for data in &test {
            let output = network.feed_forward(data.input.clone());
            avg_cost += network.cost(&output, &data.target) / test.len() as f64;
            if output.argmax() == data.target.argmax() {
                accuracy += 1.0 / test.len() as f64;
            }
            confidence += output.0[output.argmax()] / test.len() as f64;
        }
        println!("Average cost: {}", avg_cost);
        println!("Accuracy: {}", accuracy);
        println!("Confidence: {}", confidence);

        static BATCH_SIZE: usize = 5000;
        for i in 0..training_data.len() / BATCH_SIZE {
            println!("Training batch: {}", i);
            // network.train(&training_data[BATCH_SIZE * i..BATCH_SIZE * (i+ 1)], 0.1);
            network.train(&training_data[BATCH_SIZE * i..BATCH_SIZE * (i + 1)], 0.1);
        }
        iter += 1;
    }
}
