# mnist-rs
A neural network solving the MNIST database fully written in Rust.

This is an educational example to understand the math behind gradient descent, the core behind machine learning. 
If you want to learn more about the topic, here are some great resources:
- I highly recommend the [awesome video series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) and [interactive lessons](https://www.3blue1brown.com/topics/neural-networks) by 3Blue1Brown.
- The free online book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) by Michael Nielsen also gives great insights into neural networks.
- [Neural Network from Scratch](https://www.youtube.com/watch?v=pauPCy_s0Ok) by The Independent Code is
  another great watch to implement a simple Neural Network

## Usage
Use `cargo run` to start the program. 
The program will load the dataset located at `data/` and ask if you want to train a new network or load the pre-trained network located at `network.ben`.
The network will then be tested and trained in batches. 
To exit and save the network, hit Ctrl+C. 
The network will finish training the current batch, save the network and exit gracefully.
