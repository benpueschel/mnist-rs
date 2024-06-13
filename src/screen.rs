use std::time::Duration;

use colored::Colorize;

use crate::{
    math::{self, Vector},
    network::TrainingData,
};

pub fn clear_screen() {
    print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
}

pub fn move_cursor() {
    print!("{esc}[1;1H", esc = 27 as char);
}

#[derive(Default)]
pub struct ScreenInfo {
    pub epoch: usize,
    pub batch: usize,
    pub total_batches: usize,
    pub elapsed: Duration,
    pub test_accuracy: f64,
    pub test_loss: f64,
    pub test_confidence: f64,
    pub error_least_confident: Option<(Vector, f64)>,
    pub error_least_confident_data: Option<TrainingData>,
    pub status: &'static str,
    pub exit: bool,
}

pub fn display_info(info: &ScreenInfo) {
    //clear_screen();
    move_cursor();

    match &info.error_least_confident {
        Some(x) => {
            let output = &x.0;
            let data = info.error_least_confident_data.as_ref().unwrap();

            let predicted = output.argmax();
            let actual = data.target.argmax();
            let confidence = math::softmax(output.clone())[predicted];

            print_image(&data.input, (28, 28));
            println!(
                "digit: {}, prediction: {}",
                actual, predicted
            );
            println!("confidence: {:.5?}", confidence);
            println!("output: {:.5?}", output.0);
        }
        None => {
            let _ = (0..32).map(|_| println!());
        }
    };
    println!();
    println!(
        "Epoch {:#03} (batch {:#03}/{:#03}):",
        info.epoch, info.batch, info.total_batches
    );
    println!(" Average loss: {:.10}", info.test_loss);
    println!(" Accuracy:     {:.10}", info.test_accuracy);
    println!(" Confidence:   {:.5}", info.test_confidence);
    println!(" Time elapsed: {:.1}s", info.elapsed.as_secs_f32());
    if info.exit {
        println!(" Waiting to exit...");
    }
    println!();
    println!("Status: {}", info.status);
}

fn print_image(data: &Vector, size: (usize, usize)) {
    for i in 0..size.0 {
        for j in 0..size.1 {
            let value = data.at(i * size.1 + j);
            let value = (value * 255.0) as u8;
            // use ▉ as a character to represent the pixel, use colored crate for grayscale output
            let pixel = "▉".truecolor(value, value, value);
            print!("{}{}", pixel, pixel);
        }
        println!();
    }
}
