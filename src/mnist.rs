use std::{fs, io::Result};

use crate::{
    math::Vector,
    network::TrainingData,
};

pub struct Dataset {
    pub image_size: (usize, usize),
    pub data: Vec<u8>,
    pub labels: Vec<u8>,
}

impl From<&Dataset> for Vec<TrainingData> {
    fn from(dataset: &Dataset) -> Vec<TrainingData> {
        let image_size = dataset.image_size.0 * dataset.image_size.1;
        let mut result = Vec::with_capacity(dataset.labels.len());

        for i in 0..dataset.labels.len() {
            let label = dataset.labels[i] as usize;
            let offset = i * image_size;

            let mut input = Vector::new(image_size);
            for j in 0..image_size {
                input.0[j] = dataset.data[offset + j] as f64 / 255.0;
            }

            let mut target = Vector::new(10);
            target.set(label.saturating_sub(1), 1.0);

            result.push(TrainingData { input, target });
        }

        result
    }
}

pub fn load_datasets(path: &str) -> Result<(Dataset, Dataset)> {
    let train_image_path = format!("{}/train-images-idx3-ubyte", path);
    let train_label_path = format!("{}/train-labels-idx1-ubyte", path);
    let test_image_path = format!("{}/t10k-images-idx3-ubyte", path);
    let test_label_path = format!("{}/t10k-labels-idx1-ubyte", path);

    let train = load_dataset(&train_image_path, &train_label_path)?;
    let test = load_dataset(&test_image_path, &test_label_path)?;

    assert_eq!(
        train.image_size, test.image_size,
        "Train and test image sizes do not match"
    );

    Ok((train, test))
}

pub fn load_dataset(image_path: &str, label_path: &str) -> Result<Dataset> {
    // image data layout:
    // 0 - 3: magic number (0x00000803 -> 2051)
    // 4 - 7: number of images
    // 8 - 11: number of rows
    // 12 - 15: number of columns
    // 16 - ...: image data
    println!("Loading image data from: {}", image_path);
    let image_data = fs::read(image_path)?;

    assert_eq!(image_data[0..4], [0, 0, 8, 3], "Invalid image magic number");

    let num_images = u32_from_bytes(&image_data[4..8]) as usize;
    let num_rows = u32_from_bytes(&image_data[8..12]) as usize;
    let num_cols = u32_from_bytes(&image_data[12..16]) as usize;
    let image_size = num_rows * num_cols;
    let data = image_data[16..].to_vec();

    assert_eq!(
        data.len(),
        num_images * image_size,
        "Image data length does not match the expected size"
    );

    // label data layout:
    // 0 - 3: magic number (0x00000801 -> 2049)
    // 4 - 7: number of labels
    // 8 - ...: label data
    println!("Loading label data from: {}", label_path);
    let label_data = fs::read(label_path)?;

    assert_eq!(label_data[0..4], [0, 0, 8, 1], "Invalid label magic number");

    let num_labels = u32_from_bytes(&label_data[4..8]) as usize;
    assert_eq!(
        num_labels, num_images,
        "Number of labels does not match the number of images"
    );
    let labels = label_data[8..].to_vec();

    Ok(Dataset {
        image_size: (num_rows, num_cols),
        data,
        labels,
    })
}

fn u32_from_bytes(bytes: &[u8]) -> u32 {
    i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
}


