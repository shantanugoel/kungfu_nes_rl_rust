#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub const STATE_DIM: usize = 49;
pub type Features = [f32; STATE_DIM];

pub mod env;
pub mod dqn;

pub use env::{Action, EnvConfig, GameState, NesEnv, RewardBreakdown, RewardConfig, StepResult, ram};
pub use dqn::{AgentConfig, DqnAgent, DqnNet, ReplayBuffer, Transition, TrainMeta, BatchTensors, AdamW};
