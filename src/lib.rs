pub const STATE_DIM: usize = 98;
pub type Features = [f32; STATE_DIM];

pub mod dqn;
pub mod env;
pub mod eval;
pub mod train_parallel;

pub use dqn::{AgentConfig, BatchTensors, DqnAgent, DqnNet, ReplayBuffer, TrainMeta, Transition};
pub use env::{
    Action, EnvConfig, GameState, NesEnv, RewardBreakdown, RewardConfig, StepResult, ram,
};
pub use eval::{EvalStats, run_eval};
