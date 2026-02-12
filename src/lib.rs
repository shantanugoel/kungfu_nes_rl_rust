#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub const STATE_DIM: usize = 49;
pub type Features = [f32; STATE_DIM];

pub mod dqn;
pub mod env;
pub mod train_parallel;

pub use dqn::{
    AdamW, AgentConfig, BatchTensors, DqnAgent, DqnNet, ReplayBuffer, TrainMeta, Transition,
};
pub use env::{
    Action, EnvConfig, GameState, NesEnv, RewardBreakdown, RewardConfig, StepResult, ram,
};
