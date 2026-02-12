use anyhow::{Context, Result};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Linear, Module, ParamsAdamW, VarBuilder, VarMap};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;

use crate::env::Action;
use crate::{Features, STATE_DIM};

// =============================================================================
// Agent Hyperparameters
// =============================================================================

pub struct AgentConfig {
    pub hidden_size: usize,
    pub dueling_hidden_size: usize,
    pub gamma: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay_steps: u64,
    pub tau: f64,
    pub max_grad_norm: f64,
    pub initial_lr: f64,
    pub weight_decay: f64,
    pub lr_decay_start: u64,
    pub lr_decay_factor: f64,
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub learn_start: usize,
    pub train_freq: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            hidden_size: 256,
            dueling_hidden_size: 128,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 2_000_000,
            tau: 0.005,
            max_grad_norm: 10.0,
            initial_lr: 1e-4,
            weight_decay: 1e-5,
            lr_decay_start: 1_000_000,
            lr_decay_factor: 0.5,
            replay_capacity: 1_000_000,
            batch_size: 64,
            learn_start: 10_000,
            train_freq: 4,
        }
    }
}

// =============================================================================
// DQN Neural Network (candle)
// =============================================================================

/// DQN with dueling architecture for RAM-based state input
/// Input: STATE_DIM features → Hidden layers → Action::COUNT Q-values
pub struct DqnNet {
    fc1: Linear,
    fc2: Linear,
    // Dueling streams
    value_fc: Linear,
    value_out: Linear,
    advantage_fc: Linear,
    advantage_out: Linear,
}

impl DqnNet {
    pub fn new(vs: VarBuilder, config: &AgentConfig) -> Result<Self> {
        let h = config.hidden_size;
        let dh = config.dueling_hidden_size;
        let fc1 = candle_nn::linear(STATE_DIM, h, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(h, h, vs.pp("fc2"))?;

        let value_fc = candle_nn::linear(h, dh, vs.pp("val_fc"))?;
        let value_out = candle_nn::linear(dh, 1, vs.pp("val_out"))?;
        let advantage_fc = candle_nn::linear(h, dh, vs.pp("adv_fc"))?;
        let advantage_out = candle_nn::linear(dh, Action::COUNT, vs.pp("adv_out"))?;

        Ok(Self {
            fc1,
            fc2,
            value_fc,
            value_out,
            advantage_fc,
            advantage_out,
        })
    }

    /// Forward pass: state → Q-values for all actions
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.fc1.forward(x)?.relu()?;
        let h = self.fc2.forward(&h)?.relu()?;

        // Value stream
        let v = self.value_fc.forward(&h)?.relu()?;
        let v = self.value_out.forward(&v)?;

        // Advantage stream
        let a = self.advantage_fc.forward(&h)?.relu()?;
        let a = self.advantage_out.forward(&a)?;

        // Q = V + (A - mean(A))
        let a_mean = a.mean_keepdim(candle_core::D::Minus1)?;
        let q = v.broadcast_add(&a.broadcast_sub(&a_mean)?)?;
        Ok(q)
    }
}

// =============================================================================
// Experience Replay Buffer
// =============================================================================

#[derive(Clone, Serialize, Deserialize)]
pub struct Transition {
    #[serde(with = "serde_big_array::BigArray")]
    pub state: Features,
    pub action: usize,
    pub reward: f32,
    #[serde(with = "serde_big_array::BigArray")]
    pub next_state: Features,
    pub done: bool,
}

#[derive(Serialize, Deserialize)]
pub struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, t: Transition) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(t);
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let reader = std::io::BufReader::new(file);
        let replay = bincode::deserialize_from(reader)?;
        Ok(replay)
    }

    /// Sample a random batch, return tensors ready for training
    pub fn sample(
        &self,
        batch_size: usize,
        dev: &Device,
        rng: &mut SmallRng,
    ) -> Result<BatchTensors> {
        let len = self.buffer.len();
        assert!(len >= batch_size);

        let mut states = Vec::with_capacity(batch_size * STATE_DIM);
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut next_states = Vec::with_capacity(batch_size * STATE_DIM);
        let mut dones = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = rng.random_range(0..len);
            let t = &self.buffer[idx];
            states.extend_from_slice(&t.state);
            actions.push(t.action as i64);
            rewards.push(t.reward);
            next_states.extend_from_slice(&t.next_state);
            dones.push(if t.done { 0.0f32 } else { 1.0f32 });
        }

        Ok(BatchTensors {
            states: Tensor::from_vec(states, (batch_size, STATE_DIM), dev)?,
            actions: Tensor::from_vec(actions, batch_size, dev)?,
            rewards: Tensor::from_vec(rewards, batch_size, dev)?,
            next_states: Tensor::from_vec(next_states, (batch_size, STATE_DIM), dev)?,
            not_dones: Tensor::from_vec(dones, batch_size, dev)?,
        })
    }
}

pub struct BatchTensors {
    pub states: Tensor,
    pub actions: Tensor,
    pub rewards: Tensor,
    pub next_states: Tensor,
    pub not_dones: Tensor,
}

#[derive(Serialize, Deserialize)]
pub struct TrainMeta {
    pub best_reward: f64,
    pub episode: u64,
    pub total_steps: u64,
    pub epsilon: f64,
    pub agent_steps: u64,
}

pub fn save_checkpoint(
    agent: &DqnAgent,
    best_reward: f64,
    episode: u64,
    total_steps: u64,
    dir: &str,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    agent
        .online_varmap
        .save(Path::new(dir).join("model.safetensors"))?;
    agent
        .target_varmap
        .save(Path::new(dir).join("target.safetensors"))?;
    agent.save_optimizer(Path::new(dir).join("optimizer.safetensors"))?;
    agent.replay.save(Path::new(dir).join("replay.bin"))?;

    let meta = TrainMeta {
        best_reward,
        episode,
        total_steps,
        epsilon: agent.epsilon,
        agent_steps: agent.steps,
    };
    let file = File::create(Path::new(dir).join("meta.json"))?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer(writer, &meta)?;
    Ok(())
}

pub fn save_recent_rewards(recent_rewards: &VecDeque<f64>, dir: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let rewards: Vec<f64> = recent_rewards.iter().copied().collect();
    let file = File::create(Path::new(dir).join("recent_rewards.json"))?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer(writer, &rewards)?;
    Ok(())
}

// =============================================================================
// DQN Agent
// =============================================================================

pub struct DqnAgent {
    pub online_varmap: VarMap,
    pub target_varmap: VarMap,
    online_net: DqnNet,
    target_net: DqnNet,
    optimizer: AdamW,
    device: Device,
    gamma: f64,
    pub epsilon: f64,
    epsilon_start: f64,
    epsilon_end: f64,
    epsilon_decay_steps: u64,
    tau: f64,
    max_grad_norm: f64,
    initial_lr: f64,
    lr_decay_start: u64,
    lr_decay_factor: f64,
    pub replay: ReplayBuffer,
    batch_size: usize,
    learn_start: usize,
    train_freq: u64,
    pub total_env_steps: u64,
    pub steps: u64,
    rng: SmallRng,
}

pub struct AdamW {
    vars: Vec<VarAdamW>,
    pub(crate) step_t: usize,
    pub(crate) params: ParamsAdamW,
}

struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

impl AdamW {
    pub fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    pub fn backward_step(&mut self, loss: &Tensor, max_grad_norm: Option<f64>) -> Result<()> {
        let grads = loss.backward()?;
        if let Some(max_norm) = max_grad_norm {
            let mut total_norm_sq = 0f64;
            for var in self.vars.iter() {
                if let Some(g) = grads.get(&var.var) {
                    let norm = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                    total_norm_sq += norm;
                }
            }
            let total_norm = total_norm_sq.sqrt();
            if total_norm > max_norm {
                let clip_coef = max_norm / (total_norm + 1e-6);
                self.step_with_clip(&grads, clip_coef)
            } else {
                self.step(&grads)
            }
        } else {
            self.step(&grads)
        }
    }

    fn step_with_clip(&mut self, grads: &GradStore, clip_coef: f64) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                let g = (g * clip_coef)?;
                let next_m = ((m.as_tensor() * beta1)? + (&g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }

    pub fn save_state<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "adamw.step_t".to_string(),
            Tensor::from_slice(&[self.step_t as u32], 1, &Device::Cpu)?,
        );
        for (i, var) in self.vars.iter().enumerate() {
            tensors.insert(
                format!("adamw.m.{i}"),
                var.first_moment.as_tensor().detach(),
            );
            tensors.insert(
                format!("adamw.v.{i}"),
                var.second_moment.as_tensor().detach(),
            );
        }
        candle_core::safetensors::save(&tensors, path.as_ref())?;
        Ok(())
    }

    pub fn load_state<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let tensors = candle_core::safetensors::load(path.as_ref(), &Device::Cpu)?;
        let step_t = tensors
            .get("adamw.step_t")
            .context("Missing adamw.step_t in optimizer state")?
            .to_vec1::<u32>()?;
        self.step_t = step_t.first().copied().unwrap_or(0) as usize;
        for (i, var) in self.vars.iter().enumerate() {
            let m = tensors
                .get(&format!("adamw.m.{i}"))
                .context("Missing adamw.m tensor in optimizer state")?;
            let v = tensors
                .get(&format!("adamw.v.{i}"))
                .context("Missing adamw.v tensor in optimizer state")?;
            var.first_moment.set(m)?;
            var.second_moment.set(v)?;
        }
        Ok(())
    }
}

impl DqnAgent {
    pub fn new(device: &Device, config: AgentConfig) -> Result<Self> {
        let online_varmap = VarMap::new();
        let target_varmap = VarMap::new();

        let online_vb = VarBuilder::from_varmap(&online_varmap, DType::F32, device);
        let target_vb = VarBuilder::from_varmap(&target_varmap, DType::F32, device);

        let online_net = DqnNet::new(online_vb, &config)?;
        let target_net = DqnNet::new(target_vb, &config)?;

        let opt_params = ParamsAdamW {
            lr: config.initial_lr,
            weight_decay: config.weight_decay,
            ..Default::default()
        };
        let optimizer = AdamW::new(online_varmap.all_vars(), opt_params)?;

        let mut agent = Self {
            online_varmap,
            target_varmap,
            online_net,
            target_net,
            optimizer,
            device: device.clone(),
            gamma: config.gamma,
            epsilon: config.epsilon_start,
            epsilon_start: config.epsilon_start,
            epsilon_end: config.epsilon_end,
            epsilon_decay_steps: config.epsilon_decay_steps,
            tau: config.tau,
            max_grad_norm: config.max_grad_norm,
            initial_lr: config.initial_lr,
            lr_decay_start: config.lr_decay_start,
            lr_decay_factor: config.lr_decay_factor,
            replay: ReplayBuffer::new(config.replay_capacity),
            batch_size: config.batch_size,
            learn_start: config.learn_start,
            train_freq: config.train_freq,
            total_env_steps: 0,
            steps: 0,
            rng: SmallRng::from_os_rng(),
        };
        agent.hard_update_target()?;
        Ok(agent)
    }

    pub fn resume_from(&mut self, resume_dir: &Path) -> Result<(TrainMeta, ReplayBuffer)> {
        let model_path = resume_dir.join("model.safetensors");
        let target_path = resume_dir.join("target.safetensors");
        let optimizer_path = resume_dir.join("optimizer.safetensors");
        let replay_bin_path = resume_dir.join("replay.bin");
        let meta_path = resume_dir.join("meta.json");

        self.online_varmap.load(&model_path)?;
        self.target_varmap.load(&target_path)?;
        if let Err(err) = self.load_optimizer(&optimizer_path) {
            eprintln!(
                "⚠️  Optimizer state load failed ({err}). Continuing with fresh optimizer state."
            );
        }
        let replay = ReplayBuffer::load(&replay_bin_path)?;

        let file = File::open(&meta_path)?;
        let reader = std::io::BufReader::new(file);
        let meta: TrainMeta = serde_json::from_reader(reader)?;
        Ok((meta, replay))
    }

    /// Select action using epsilon-greedy
    pub fn select_action(&mut self, state: &[f32]) -> Result<usize> {
        if self.rng.random::<f64>() < self.epsilon {
            Ok(self.rng.random_range(0..Action::COUNT))
        } else {
            let s = Tensor::from_slice(state, (1, STATE_DIM), &self.device)?;
            let q = self.online_net.forward(&s)?;
            let action = q
                .argmax(candle_core::D::Minus1)?
                .squeeze(0)?
                .to_scalar::<u32>()? as usize;
            Ok(action)
        }
    }

    /// Store transition in replay buffer
    pub fn remember(&mut self, t: Transition) {
        self.replay.push(t);
    }

    /// Train on a batch from replay buffer (Double DQN)
    pub fn train_step(&mut self) -> Result<f32> {
        if self.replay.len() < self.learn_start {
            return Ok(0.0);
        }
        self.steps += 1;
        if !self.steps.is_multiple_of(self.train_freq) {
            return Ok(0.0);
        }

        let batch = self
            .replay
            .sample(self.batch_size, &self.device, &mut self.rng)?;

        // Online net: Q(s, a) for the actions we actually took
        let q_all = self.online_net.forward(&batch.states)?;
        let actions_unsqueezed = batch.actions.unsqueeze(1)?;
        let q_values = q_all.gather(&actions_unsqueezed, 1)?.squeeze(1)?;

        // Double DQN: use online net to SELECT best action, target net to EVALUATE
        let next_q_online = self.online_net.forward(&batch.next_states)?;
        let best_next_actions = next_q_online.argmax(candle_core::D::Minus1)?.unsqueeze(1)?;

        let next_q_target = self.target_net.forward(&batch.next_states)?;
        let next_q = next_q_target
            .gather(&best_next_actions.to_dtype(DType::I64)?, 1)?
            .squeeze(1)?;

        // Target: r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
        let discounted = next_q.affine(self.gamma, 0.0)?;
        let target = batch.rewards.add(&discounted.mul(&batch.not_dones)?)?;

        // Huber loss (smooth L1) — more robust than MSE for RL
        let diff = q_values.sub(&target.detach())?;
        let abs_diff = diff.abs()?;
        let ones = Tensor::ones_like(&abs_diff)?;
        // Huber: where |d| < 1: 0.5*d^2, else |d| - 0.5
        let loss = abs_diff
            .lt(&ones)?
            .where_cond(
                &(diff.sqr()?.affine(0.5, 0.0)?),
                &(abs_diff.affine(1.0, -0.5)?),
            )?
            .mean_all()?;

        // Backprop with gradient clipping
        self.optimizer
            .backward_step(&loss, Some(self.max_grad_norm))?;

        // Soft update target network every gradient step
        self.soft_update_target()?;

        // Linear epsilon decay based on total env steps
        let progress = (self.total_env_steps as f64) / (self.epsilon_decay_steps as f64);
        self.epsilon =
            self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress.min(1.0);

        // LR decay after lr_decay_start steps
        if self.total_env_steps > self.lr_decay_start {
            let decayed_lr = self.initial_lr * self.lr_decay_factor;
            self.optimizer.set_learning_rate(decayed_lr);
        }

        loss.to_scalar::<f32>().map_err(Into::into)
    }

    /// Copy online weights → target (hard copy)
    pub fn hard_update_target(&mut self) -> Result<()> {
        let online_data = self
            .online_varmap
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock online varmap for hard update"))?;
        let mut target_data = self
            .target_varmap
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock target varmap for hard update"))?;
        for (name, target_v) in target_data.iter_mut() {
            let online_v = online_data.get(name).ok_or_else(|| {
                anyhow::anyhow!("Missing var {name} in online varmap during hard update")
            })?;
            target_v.set(&online_v.as_tensor().detach())?;
        }
        Ok(())
    }

    /// Soft update: target = tau * online + (1-tau) * target
    pub fn soft_update_target(&mut self) -> Result<()> {
        let tau = self.tau as f32;
        let online_data = self
            .online_varmap
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock online varmap for soft update"))?;
        let mut target_data = self
            .target_varmap
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock target varmap for soft update"))?;
        for (name, target_v) in target_data.iter_mut() {
            let online_v = online_data.get(name).ok_or_else(|| {
                anyhow::anyhow!("Missing var {name} in online varmap during soft update")
            })?;
            let new_val = online_v
                .as_tensor()
                .affine(tau as f64, 0.0)?
                .add(&target_v.as_tensor().affine((1.0 - tau) as f64, 0.0)?)?;
            target_v.set(&new_val.detach())?;
        }
        Ok(())
    }

    /// Save model weights
    pub fn save(&self, path: &str) -> Result<()> {
        self.online_varmap.save(path)?;
        eprintln!("💾 Model saved to {path}");
        Ok(())
    }

    /// Load model weights
    pub fn load(&mut self, path: &str) -> Result<()> {
        self.online_varmap.load(path)?;
        self.hard_update_target()?;
        eprintln!("📂 Model loaded from {path}");
        Ok(())
    }

    pub fn save_optimizer<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.optimizer.save_state(path)
    }

    pub fn load_optimizer<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.optimizer.load_state(path)
    }
}
