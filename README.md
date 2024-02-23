# Rust Speech-to-Text with Candle Framework

This repository contains a Rust program for performing real-time speech-to-text conversion using the Candle(https://github.com/huggingface/candle/tree/main) machine learning framework. The program utilizes the Metal GPU acceleration on MacBook devices for efficient processing. It leverages the CPAL(https://crates.io/crates/cpal) library for accessing the microphone input.

## Idea Behind the Project

The primary motivation behind creating this project was to delve deeper into low-level programming languages like Rust and to explore the capabilities of machine learning frameworks written in Rust, such as Candle. By building a real-world application that combines Rust's efficiency with Candle's speech-to-text capabilities, I aimed to gain insights into memory management, concurrency, and GPU acceleration.

Furthermore, I wanted to simplify the original Whisper Microphone example from the Candle Examples repository to create a more focused and beginner-friendly project. By breaking down the example into smaller components and using only one speech-to-text model (Whisper Tiny.en), I aimed to provide a clearer understanding of how to integrate Candle into Rust projects for real-time applications.

## Features

- Real-time speech-to-text conversion.
- Utilizes the Candle machine learning framework.
- Takes advantage of Metal GPU acceleration on MacBook devices.
- Accesses microphone input using the CPAL library.
- Simplified code structure for easy understanding and modification.

## Requirements

- Rust programming language installed on your system.
- Metal-compatible MacBook device.
- Access to a microphone.

## Installation and Usage

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/singhaki/fun_learning_rust_whisper.git
    ```

2. Navigate to the cloned directory:

    ```bash
    cd rust-speech-to-text-candle
    ```

3. Run the program using Cargo:

    ```bash
    cargo run --features metal --release
    ```

4. Speak into your microphone, and the program will convert speech to text in real time.

## Contributions

Contributions are welcome! If you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Candle framework by Hugging Face for providing efficient machine learning capabilities in Rust.(https://github.com/huggingface/candle/tree/main)
- CPAL library for enabling access to microphone input. (https://crates.io/crates/cpal)
- [Original Whisper Microphone example](https://github.com/huggingface/candle-examples/tree/main/examples/whisper-microphone) from Candle Examples repository, which served as inspiration for this project.
