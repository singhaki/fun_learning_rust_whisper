use anyhow::Result;
use cpal::traits::{DeviceTrait, StreamTrait};
use std::sync::{Arc, Mutex};

pub fn record_audio(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    milliseconds: u64,
) -> Result<Vec<i16>> {
    let writer = Arc::new(Mutex::new(Vec::new()));
    let writer_2 = writer.clone();
    let stream = device.build_input_stream(
        &config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let processed = data
                .iter()
                .map(|v| (v * 32768.0) as i16)
                .collect::<Vec<i16>>();
            writer_2.lock().unwrap().extend_from_slice(&processed);
        },
        move |err| {
            eprintln!("an error occurred on stream: {}", err);
        },
        None,
    )?;
    stream.play()?;
    std::thread::sleep(std::time::Duration::from_millis(milliseconds));
    drop(stream);
    let data = writer.lock().unwrap().clone();
    let step = 3;
    let data: Vec<i16> = data.iter().step_by(step).copied().collect();
    Ok(data)
}
