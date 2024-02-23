/*
In Rust, the anyhow crate is used for error handling. It provides a consistent and idiomatic way to handle errors in Rust programs. It is used to create error contexts that can be propagated through the program, making it easier to handle and manage errors in a structured and efficient manner.
*/
use anyhow::{Error as E, Result};

// install candle dependencies
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::iter;
use tokenizers::Tokenizer;

// audio_utils module holds only audio recording from your microphone
mod audio_utils;

//main whisper model with it's decoder
mod whisper;

/*The cpal crate makes it easier to work with audio in Rust applications, allowing developers to focus on their specific use cases without having to worry about the underlying audio handling details.
*/
use cpal::traits::{DeviceTrait, HostTrait};
use std::sync::{Arc, Mutex};

pub fn main() -> Result<()> {
    // Importing necessary traits and modules from the tracing_chrome and tracing_subscriber crates.This code sets up a tracing subscriber with a ChromeLayer for structured logging and debugging.
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    // Creating a guard variable to keep the ChromeLayer alive.
    let _guard = {
        // Creating a new ChromeLayer using the ChromeLayerBuilder.
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();

        // Retrieving the global subscriber registry and attaching the ChromeLayer.
        tracing_subscriber::registry().with(chrome_layer).init();

        // Returning Some(guard) to ensure the guard is kept alive.
        Some(guard)
    };

    // set gpu device for using mac gpu
    let device = Device::new_metal(0).unwrap();

    // setup config and load whisper tiny english model
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            "openai/whisper-tiny.en".to_string(),
            RepoType::Model,
            "refs/pr/15".to_string(),
        ));
        let (config, tokenizer, model) = {
            let config = repo.get("config.json")?;
            let tokenizer = repo.get("tokenizer.json")?;
            let model = repo.get("model.safetensors")?;
            (config, tokenizer, model)
        };
        (config, tokenizer, model)
    };
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let model = {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        whisper::Model::Normal(m::model::Whisper::load(&vb, config.clone())?)
    };
    let language_token = None;
    let mut dc = whisper::Decoder::new(
        model,
        tokenizer,
        299792458,
        &device,
        language_token,
        false,
        false,
    )?;

    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    // Set up the input device and stream with the default input config.
    let host = cpal::default_host();
    let _device = "default";
    let _device = if _device == "default" {
        host.default_input_device()
    } else {
        host.input_devices()?
            .find(|x| x.name().map(|y| y == _device).unwrap_or(false))
    }
    .expect("failed to find input device");

    let _config = _device
        .default_input_config()
        .expect("Failed to get default input config");

    let channel_count = _config.channels() as usize;

    let audio_ring_buffer = Arc::new(Mutex::new(Vec::new()));
    let audio_ring_buffer_2 = audio_ring_buffer.clone();

    // spwawn a thread to read audio using the cpal crate data from microphone is loaded in audio buffer
    std::thread::spawn(move || loop {
        let data = audio_utils::record_audio(&_device, &_config, 300).unwrap();
        audio_ring_buffer.lock().unwrap().extend_from_slice(&data);
        let max_len = data.len() * 16;
        let data_len = data.len();
        let len = audio_ring_buffer.lock().unwrap().len();
        if len > max_len {
            let mut data = audio_ring_buffer.lock().unwrap();
            let new_data = data[data_len..].to_vec();
            *data = new_data;
        }
    });

    // loop to process the audio data forever (until the user stops the program)
    println!("Transcribing audio...");
    for (i, _) in iter::repeat(()).enumerate() {
        std::thread::sleep(std::time::Duration::from_millis(1000));
        let data = audio_ring_buffer_2.lock().unwrap().clone();
        let pcm_data: Vec<_> = data[..data.len() / channel_count as usize]
            .iter()
            .map(|v| *v as f32 / 32768.)
            .collect();
        let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            &device,
        )?;

        // on the first iteration, we detect the language and set the language token.
        if i == 0 {
            let language_token = None;
            println!("language_token: {:?}", language_token);
            dc.set_language_token(language_token);
        }
        dc.run(
            &mel,
            Some((
                i as f64,
                i as f64 + data.len() as f64 / m::SAMPLE_RATE as f64,
            )),
        )?;
        dc.reset_kv_cache();
    }

    Ok(())
}
