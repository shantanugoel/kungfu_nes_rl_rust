use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{
    traits::{Consumer, Producer, Split},
    CachingCons, CachingProd, HeapRb,
};
use std::sync::Arc;

const DEFAULT_SAMPLE_RATE: u32 = 44_100;
const DEFAULT_LATENCY_MS: f32 = 40.0;

pub struct AudioOutput {
    stream: cpal::Stream,
    sample_rate: u32,
    channels: u16,
    producer: CachingProd<Arc<HeapRb<f32>>>,
}

impl AudioOutput {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no audio output device available")?;
        let config_range = device
            .supported_output_configs()
            .context("failed to query audio output configs")?;
        let mut chosen = None;
        for cfg in config_range {
            if cfg.sample_format() == cpal::SampleFormat::F32 {
                chosen = Some(cfg);
                break;
            }
        }
        let supported_range = chosen.context("no supported f32 output config found")?;
        let desired_rate: cpal::SampleRate = DEFAULT_SAMPLE_RATE;
        let min_rate = supported_range.min_sample_rate();
        let max_rate = supported_range.max_sample_rate();
        let clamped_rate: cpal::SampleRate = desired_rate.clamp(min_rate, max_rate);
        let supported_config = supported_range.with_sample_rate(clamped_rate);
        let sample_format = supported_config.sample_format();
        let stream_config: cpal::StreamConfig = supported_config.into();

        let channels = stream_config.channels;
        let sample_rate = stream_config.sample_rate;
        let latency_frames =
            ((DEFAULT_LATENCY_MS / 1000.0) * sample_rate as f32 * channels as f32).ceil() as usize;
        let rb_capacity = (latency_frames * 2).max(sample_rate as usize);
        let rb = Arc::new(HeapRb::<f32>::new(rb_capacity));
        let (producer, mut consumer) = rb.split();
        let err_fn = |err| eprintln!("audio stream error: {err:?}");
        let stream = match sample_format {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &stream_config,
                move |out: &mut [f32], _| {
                    Self::fill_output(out, &mut consumer);
                },
                err_fn,
                None,
            )?,
            _ => {
                anyhow::bail!("unsupported sample format: {sample_format:?}");
            }
        };

        stream.play()?;

        Ok(Self {
            stream,
            sample_rate,
            channels,
            producer,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn push_samples(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let channels = self.channels as usize;
        for &sample in samples {
            for _ in 0..channels {
                if self.producer.try_push(sample).is_err() {
                    break;
                }
            }
        }
    }

    fn fill_output(out: &mut [f32], consumer: &mut CachingCons<Arc<HeapRb<f32>>>) {
        let filled = consumer.pop_slice(out);
        if filled < out.len() {
            out[filled..].fill(0.0);
        }
    }
}

impl Drop for AudioOutput {
    fn drop(&mut self) {
        let _ = self.stream.pause();
    }
}
