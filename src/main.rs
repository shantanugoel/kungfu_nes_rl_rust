// =============================================================================
// Kung Fu Master (Spartan X) NES — DQN Reinforcement Learning Agent in Rust
// =============================================================================
// Build & Run:
//   cargo build --release
//   cargo run --release -- train --rom kung_fu.nes --timesteps 2000000
//   cargo run --release -- play  --rom kung_fu.nes --model checkpoints/best.safetensors
//   cargo run --release -- explore --rom kung_fu.nes

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use rand::Rng;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;
use tetanes_core::mem::Read;
use tetanes_core::prelude::*;
fn update_pause_from_window(env: &mut NesEnv, window: &minifb::Window) {
    if window.is_key_pressed(minifb::Key::Space, minifb::KeyRepeat::No) {
        env.toggle_pause();
    }
}

// =============================================================================
// Section 1: RAM Addresses — YOUR confirmed addresses
// =============================================================================

mod ram {
    /// Your confirmed RAM map for your specific ROM revision
    pub const PLAYER_LIVES: u16 = 0x005C;
    pub const PLAYER_X: u16 = 0x00D4;
    pub const PLAYER_Y: u16 = 0x00B6;
    pub const PLAYER_HP: u16 = 0x04A6;
    pub const PLAYER_POSE: u16 = 0x036E;
    pub const PLAYER_STATE: u16 = 0x036F;

    // Enemy slots (4 active)
    pub const ENEMY_X: [u16; 4] = [0x00CE, 0x00CF, 0x00D0, 0x00D1];
    pub const ENEMY_TYPE: [u16; 4] = [0x0087, 0x0088, 0x0089, 0x008A];
    pub const ENEMY_Y: [u16; 4] = [0x00B0, 0x00B1, 0x00B2, 0x00B3];
    pub const ENEMY_FACING: [u16; 4] = [0x00C0, 0x00C1, 0x00C2, 0x00C3];
    pub const ENEMY_POSE: [u16; 4] = [0x00DF, 0x00E0, 0x00E1, 0x00E2];

    pub const KILL_COUNTER: u16 = 0x03B1;

    // Score: 6 BCD digits
    pub const SCORE_DIGITS: [u16; 6] = [0x0531, 0x0532, 0x0533, 0x0534, 0x0535, 0x0536];

    // Timer: 4 BCD digits
    pub const TIMER_DIGITS: [u16; 4] = [0x0390, 0x0391, 0x0392, 0x0393];

    // TODO: Discover these for your ROM using explore mode
    // WARNING: 0x0534 collides with SCORE_DIGITS[3] (score_100).
    pub const BOSS_HP: Option<u16> = None; // Unknown in this ROM
    pub const FLOOR: u16 = 0x0058; // Verify! Try scanning during floor transition
}

// =============================================================================
// Section 2: Action Space
// =============================================================================

/// 15 meaningful button combinations for Kung Fu
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Action {
    Noop = 0,
    Right = 1,
    Left = 2,
    Punch = 3,  // B button
    Kick = 4,   // A button
    Crouch = 5, // Down
    Jump = 6,   // Up
    RightPunch = 7,
    RightKick = 8,
    LeftPunch = 9,
    LeftKick = 10,
    CrouchPunch = 11,
    CrouchKick = 12,
    JumpPunch = 13,
    JumpKick = 14,
}

impl Action {
    pub const COUNT: usize = 15;

    pub fn from_index(i: usize) -> Self {
        assert!(i < Self::COUNT);
        // SAFETY: repr(u8) and we checked bounds
        unsafe { std::mem::transmute(i as u8) }
    }

    /// Convert to tetanes Joypad button flags
    pub fn to_joypad(self) -> tetanes_core::input::JoypadBtnState {
        use tetanes_core::input::{JoypadBtn, JoypadBtnState};
        let mut state = JoypadBtnState::empty();
        match self {
            Action::Noop => {}
            Action::Right => {
                state.set(JoypadBtn::Right.into(), true);
            }
            Action::Left => {
                state.set(JoypadBtn::Left.into(), true);
            }
            Action::Punch => {
                state.set(JoypadBtn::TurboB.into(), true);
            } // B = punch
            Action::Kick => {
                state.set(JoypadBtn::TurboA.into(), true);
            } // A = kick
            Action::Crouch => {
                state.set(JoypadBtn::Down.into(), true);
            }
            Action::Jump => {
                state.set(JoypadBtn::Up.into(), true);
            }
            Action::RightPunch => {
                state.set(JoypadBtn::Right.into(), true);
                state.set(JoypadBtn::TurboB.into(), true);
            }
            Action::RightKick => {
                state.set(JoypadBtn::Right.into(), true);
                state.set(JoypadBtn::TurboA.into(), true);
            }
            Action::LeftPunch => {
                state.set(JoypadBtn::Left.into(), true);
                state.set(JoypadBtn::TurboB.into(), true);
            }
            Action::LeftKick => {
                state.set(JoypadBtn::Left.into(), true);
                state.set(JoypadBtn::TurboA.into(), true);
            }
            Action::CrouchPunch => {
                state.set(JoypadBtn::Down.into(), true);
                state.set(JoypadBtn::TurboB.into(), true);
            }
            Action::CrouchKick => {
                state.set(JoypadBtn::Down.into(), true);
                state.set(JoypadBtn::TurboA.into(), true);
            }
            Action::JumpPunch => {
                state.set(JoypadBtn::Up.into(), true);
                state.set(JoypadBtn::TurboB.into(), true);
            }
            Action::JumpKick => {
                state.set(JoypadBtn::Up.into(), true);
                state.set(JoypadBtn::TurboA.into(), true);
            }
        }
        state
    }
}

// =============================================================================
// Section 3: NES Environment
// =============================================================================

/// Game state extracted from RAM each frame
#[derive(Debug, Clone, Default)]
pub struct GameState {
    pub player_x: u8,
    pub player_y: u8,
    pub player_hp: u8,
    pub player_lives: u8,
    pub player_state: u8,
    pub score: u32,
    pub kill_count: u8,
    pub enemy_x: [u8; 4],
    pub enemy_y: [u8; 4],
    pub enemy_type: [u8; 4],
    pub enemy_facing: [u8; 4],
    pub enemy_active: [bool; 4],
    pub boss_hp: u8,
    pub floor: u8,
    pub timer: u8,
}

impl GameState {
    /// Convert to normalized f32 feature vector for the neural network
    /// Returns ~30 features, all in [0.0, 1.0]
    pub fn to_features(&self) -> Vec<f32> {
        let mut f = Vec::with_capacity(34);

        f.push(self.player_x as f32 / 255.0);
        f.push(self.player_y as f32 / 255.0);
        let hp = if self.player_hp == 0xFF {
            0.0
        } else {
            self.player_hp as f32
        };
        f.push(hp / 176.0); // Max HP is ~176
        f.push(self.player_lives as f32 / 5.0);
        f.push(self.player_state as f32 / 255.0);
        f.push(self.floor as f32 / 5.0);

        // 4 enemy slots × 5 features = 20
        for i in 0..4 {
            f.push(self.enemy_x[i] as f32 / 255.0);
            f.push(self.enemy_y[i] as f32 / 255.0);
            f.push(self.enemy_type[i] as f32 / 7.0); // Types 0-7
            f.push(self.enemy_facing[i] as f32); // 0 or 1
            f.push(if self.enemy_active[i] { 1.0 } else { 0.0 });
        }

        // Relative enemy positions (distance to player)
        for i in 0..4 {
            let dx = (self.enemy_x[i] as f32 - self.player_x as f32) / 255.0;
            f.push(dx);
        }

        f.push(self.boss_hp as f32 / 255.0);
        f.push(self.timer as f32 / 255.0);
        f.push(self.kill_count as f32 / 255.0);

        f
    }
}

pub const STATE_DIM: usize = 34; // Must match to_features() length

pub struct NesEnv {
    deck: ControlDeck,
    prev_state: GameState,
    total_reward: f64,
    steps: u64,
    frame_skip: u32,
    sticky_prob: f64,
    last_action: Action,
    paused: bool,
}

impl NesEnv {
    pub fn new(rom_path: PathBuf, frame_skip: u32, sticky_prob: f64) -> Result<Self> {
        let mut deck = ControlDeck::new();
        deck.load_rom_path(&rom_path)
            .with_context(|| format!("Failed to load ROM: {}", rom_path.display()))?;

        Ok(Self {
            deck,
            prev_state: GameState::default(),
            total_reward: 0.0,
            steps: 0,
            frame_skip,
            sticky_prob,
            last_action: Action::Noop,
            paused: false,
        })
    }

    /// Read a byte from NES CPU address space (0x0000-0x07FF is RAM)
    fn peek(&self, addr: u16) -> u8 {
        self.deck.bus().peek(addr)
    }

    /// Read BCD score from 6 digit bytes
    fn read_score(&self) -> u32 {
        let mut score = 0u32;
        let multipliers = [100_000, 10_000, 1_000, 100, 10, 1];
        for (i, &addr) in ram::SCORE_DIGITS.iter().enumerate() {
            let digit = self.peek(addr) & 0x0F; // BCD: mask to low nibble
            score += digit as u32 * multipliers[i];
        }
        score
    }

    /// Read BCD timer from 4 digit bytes
    fn read_timer(&self) -> u16 {
        let mut timer = 0u16;
        let multipliers = [1000, 100, 10, 1];
        for (i, &addr) in ram::TIMER_DIGITS.iter().enumerate() {
            let digit = self.peek(addr) & 0x0F;
            timer += digit as u16 * multipliers[i];
        }
        timer
    }

    /// Extract full game state from RAM
    fn read_state(&self) -> GameState {
        let mut state = GameState::default();
        state.player_x = self.peek(ram::PLAYER_X);
        state.player_y = self.peek(ram::PLAYER_Y);
        state.player_hp = self.peek(ram::PLAYER_HP);
        state.player_lives = self.peek(ram::PLAYER_LIVES);
        state.player_state = self.peek(ram::PLAYER_STATE);
        state.score = self.read_score();
        state.kill_count = self.peek(ram::KILL_COUNTER);
        state.boss_hp = ram::BOSS_HP.map(|addr| self.peek(addr)).unwrap_or(0);
        state.floor = self.peek(ram::FLOOR);
        state.timer = (self.read_timer().min(u8::MAX as u16)) as u8;

        for i in 0..4 {
            state.enemy_x[i] = self.peek(ram::ENEMY_X[i]);
            state.enemy_y[i] = self.peek(ram::ENEMY_Y[i]);
            state.enemy_type[i] = self.peek(ram::ENEMY_TYPE[i]);
            state.enemy_facing[i] = self.peek(ram::ENEMY_FACING[i]);
            let pose = self.peek(ram::ENEMY_POSE[i]);
            state.enemy_active[i] = pose != 0 && pose != 0x7F;
        }
        state
    }

    /// Apply joypad input to player 1
    fn set_input(&mut self, action: Action) {
        let btn_state = action.to_joypad();
        self.set_input_state(btn_state);
    }

    /// Apply a full button state to player 1
    fn set_input_state(&mut self, btn_state: tetanes_core::input::JoypadBtnState) {
        use tetanes_core::input::JoypadBtnState;
        let joypad = self.deck.joypad_mut(Player::One);
        for button in [
            JoypadBtnState::LEFT,
            JoypadBtnState::RIGHT,
            JoypadBtnState::UP,
            JoypadBtnState::DOWN,
            JoypadBtnState::A,
            JoypadBtnState::B,
            JoypadBtnState::TURBO_A,
            JoypadBtnState::TURBO_B,
            JoypadBtnState::START,
            JoypadBtnState::SELECT,
        ] {
            joypad.set_button(button, btn_state.contains(button));
        }
    }

    /// Reset the emulator
    pub fn reset(&mut self) -> Result<Vec<f32>> {
        self.deck.reset(ResetKind::Soft);

        // Random no-op start for stochasticity
        let mut rng = rand::rng();
        let noops = rng.random_range(1..30);
        for _ in 0..noops {
            self.deck.clock_frame()?;
        }

        // Press Start to begin game
        self.press_start(60)?;

        self.prev_state = self.read_state();
        self.total_reward = 0.0;
        self.steps = 0;
        self.last_action = Action::Noop;
        self.paused = false;

        Ok(self.prev_state.to_features())
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn step_pause(&mut self) -> Result<()> {
        if !self.paused {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
        Ok(())
    }

    pub fn press_start(&mut self, frames: u32) -> Result<()> {
        use tetanes_core::input::JoypadBtnState;
        let mut btn_state = JoypadBtnState::empty();
        btn_state.set(JoypadBtnState::START, true);
        for _ in 0..frames {
            self.set_input_state(btn_state);
            self.deck.clock_frame()?;
        }
        self.set_input_state(JoypadBtnState::empty());
        Ok(())
    }

    /// Step the environment: apply action, advance frames, compute reward
    pub fn step(&mut self, action: Action) -> Result<StepResult> {
        let mut rng = rand::rng();
        self.steps += 1;

        // Sticky action: with some probability, repeat previous action
        let effective_action = if rng.random::<f64>() < self.sticky_prob {
            self.last_action
        } else {
            action
        };
        self.last_action = effective_action;

        // Apply action and advance frame_skip frames
        let mut frame_reward = 0.0;
        let mut done = false;

        for _ in 0..self.frame_skip {
            self.set_input(effective_action);
            self.deck.clock_frame()?;

            let state = self.read_state();

            // --- Reward computation ---
            let reward = self.compute_reward(&state);
            frame_reward += reward;

            // Check done: lives depleted
            if state.player_lives == 0 && self.prev_state.player_lives > 0 {
                done = true;
                frame_reward -= 25.0; // Death penalty
            }

            self.prev_state = state;

            if done {
                break;
            }
        }

        self.total_reward += frame_reward;

        // Count active enemies
        let active_enemies = (0..4).filter(|&i| self.prev_state.enemy_active[i]).count() as u8;

        Ok(StepResult {
            state: self.prev_state.to_features(),
            reward: frame_reward as f32,
            done,
            score: self.prev_state.score,
            total_reward: self.total_reward,
            kills: self.prev_state.kill_count,
            active_enemies,
        })
    }

    fn compute_reward(&self, cur: &GameState) -> f64 {
        let prev = &self.prev_state;
        let mut reward = 0.0;

        // 1. Kill counter delta — most reliable combat signal
        let kill_delta = cur.kill_count.wrapping_sub(prev.kill_count);
        if kill_delta > 0 && kill_delta < 10 {
            reward += kill_delta as f64 * 5.0;
        }

        // 2. Score delta (normalized)
        let score_delta = cur.score as i64 - prev.score as i64;
        if score_delta > 0 && score_delta < 50_000 {
            reward += score_delta as f64 / 100.0;
        }

        // 3. Health delta (penalize damage)
        if cur.player_hp != 0xFF && prev.player_hp != 0xFF {
            let hp_delta = cur.player_hp as i32 - prev.player_hp as i32;
            if hp_delta < 0 && hp_delta > -200 {
                reward += hp_delta as f64 * 0.5; // Negative
            }
        }

        // 4. Rightward movement (small)
        let dx = cur.player_x as i32 - prev.player_x as i32;
        if dx.abs() < 128 && dx > 0 {
            reward += dx as f64 * 0.02;
        }

        // 5. Floor completion bonus
        if cur.floor > prev.floor {
            reward += 100.0;
        }

        // 6. Boss damage
        let boss_delta = prev.boss_hp as i32 - cur.boss_hp as i32;
        if boss_delta > 0 && boss_delta < 200 {
            reward += boss_delta as f64 * 2.0;
        }

        // 7. Time penalty
        reward -= 0.001;

        reward
    }

    /// Get current frame buffer for visualization (RGBA, 256×240)
    pub fn frame_buffer(&mut self) -> &[u8] {
        self.deck.frame_buffer()
    }
}

pub struct StepResult {
    pub state: Vec<f32>,
    pub reward: f32,
    pub done: bool,
    pub score: u32,
    pub total_reward: f64,
    pub kills: u8,
    pub active_enemies: u8,
}

// =============================================================================
// Section 4: DQN Neural Network (candle)
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
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(STATE_DIM, 256, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(256, 256, vs.pp("fc2"))?;

        // Dueling: separate value and advantage streams
        let value_fc = candle_nn::linear(256, 128, vs.pp("val_fc"))?;
        let value_out = candle_nn::linear(128, 1, vs.pp("val_out"))?;
        let advantage_fc = candle_nn::linear(256, 128, vs.pp("adv_fc"))?;
        let advantage_out = candle_nn::linear(128, Action::COUNT, vs.pp("adv_out"))?;

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
// Section 5: Experience Replay Buffer
// =============================================================================

#[derive(Clone)]
struct Transition {
    state: Vec<f32>,
    action: usize,
    reward: f32,
    next_state: Vec<f32>,
    done: bool,
}

struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, t: Transition) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(t);
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Sample a random batch, return tensors ready for training
    fn sample(&self, batch_size: usize, dev: &Device) -> Result<BatchTensors> {
        let mut rng = rand::rng();
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

struct BatchTensors {
    states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    next_states: Tensor,
    not_dones: Tensor,
}

// =============================================================================
// Section 6: DQN Agent
// =============================================================================

struct DqnAgent {
    online_varmap: VarMap,
    target_varmap: VarMap,
    online_net: DqnNet,
    target_net: DqnNet,
    optimizer: AdamW,
    device: Device,
    gamma: f64,
    epsilon: f64,
    epsilon_min: f64,
    epsilon_decay: f64,
    tau: f64, // Soft update coefficient
    replay: ReplayBuffer,
    batch_size: usize,
    learn_start: usize, // Min replay size before training starts
    train_freq: u64,    // Train every N steps
    steps: u64,
}

impl DqnAgent {
    fn new(device: &Device) -> Result<Self> {
        let online_varmap = VarMap::new();
        let target_varmap = VarMap::new();

        let online_vb = VarBuilder::from_varmap(&online_varmap, DType::F32, device);
        let target_vb = VarBuilder::from_varmap(&target_varmap, DType::F32, device);

        let online_net = DqnNet::new(online_vb)?;
        let target_net = DqnNet::new(target_vb)?;

        let opt_params = ParamsAdamW {
            lr: 2.5e-4,
            weight_decay: 1e-5,
            ..Default::default()
        };
        let optimizer = AdamW::new(online_varmap.all_vars(), opt_params)?;

        // Copy online weights to target
        let mut agent = Self {
            online_varmap,
            target_varmap,
            online_net,
            target_net,
            optimizer,
            device: device.clone(),
            gamma: 0.99,
            epsilon: 1.0,
            epsilon_min: 0.02,
            epsilon_decay: 0.99999,
            tau: 0.005,
            replay: ReplayBuffer::new(100_000),
            batch_size: 64,
            learn_start: 1000,
            train_freq: 4,
            steps: 0,
        };
        agent.hard_update_target()?;
        Ok(agent)
    }

    /// Select action using epsilon-greedy
    fn select_action(&self, state: &[f32]) -> Result<usize> {
        let mut rng = rand::rng();
        if rng.random::<f64>() < self.epsilon {
            Ok(rng.random_range(0..Action::COUNT))
        } else {
            let s = Tensor::from_slice(state, (1, STATE_DIM), &self.device)?;
            let q = self.online_net.forward(&s)?;
            let action = q.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?[0] as usize;
            Ok(action)
        }
    }

    /// Store transition in replay buffer
    fn remember(&mut self, t: Transition) {
        self.replay.push(t);
    }

    /// Train on a batch from replay buffer (Double DQN)
    fn train_step(&mut self) -> Result<f32> {
        if self.replay.len() < self.learn_start {
            return Ok(0.0);
        }
        self.steps += 1;
        if self.steps % self.train_freq != 0 {
            return Ok(0.0);
        }

        let batch = self.replay.sample(self.batch_size, &self.device)?;

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
        let gamma_t = Tensor::from_slice(&[self.gamma as f32], 1, &self.device)?
            .broadcast_as(next_q.shape())?;

        let target = batch
            .rewards
            .add(&gamma_t.mul(&next_q)?.mul(&batch.not_dones)?)?;

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

        // Backprop
        self.optimizer.backward_step(&loss)?;

        // Soft update target network
        self.soft_update_target()?;

        // Decay epsilon
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);

        loss.to_scalar::<f32>().map_err(Into::into)
    }

    /// Copy online weights → target (hard copy)
    fn hard_update_target(&mut self) -> Result<()> {
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
    fn soft_update_target(&mut self) -> Result<()> {
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
    fn save(&self, path: &str) -> Result<()> {
        self.online_varmap.save(path)?;
        eprintln!("💾 Model saved to {path}");
        Ok(())
    }

    /// Load model weights
    fn load(&mut self, path: &str) -> Result<()> {
        self.online_varmap.load(path)?;
        self.hard_update_target()?;
        eprintln!("📂 Model loaded from {path}");
        Ok(())
    }
}

// =============================================================================
// Section 7: Training Loop
// =============================================================================

fn train(args: &TrainArgs) -> Result<()> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  TRAINING — Kung Fu Master DQN Agent (Rust + candle)");
    eprintln!("═══════════════════════════════════════════════════════════");

    // Select device: Metal on Apple Silicon, else CPU
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    eprintln!("Device: {:?}", device);

    let mut env = NesEnv::new(
        args.rom.clone(),
        args.frame_skip,
        0.25, // sticky action probability
    )?;
    env.press_start(60)?;

    let mut window = if args.render {
        Some(minifb::Window::new(
            "Kung Fu Master — Training",
            256,
            240,
            minifb::WindowOptions {
                resize: true,
                scale: minifb::Scale::X2,
                ..Default::default()
            },
        )?)
    } else {
        None
    };
    if let Some(win) = window.as_mut() {
        win.set_target_fps(60);
    }

    let mut agent = DqnAgent::new(&device)?;

    std::fs::create_dir_all("checkpoints")?;

    let mut best_reward = f64::NEG_INFINITY;
    let mut episode = 0;
    let mut total_steps: u64 = 0;
    let t_start = Instant::now();

    let mut last_render_ep = 0u64;
    let mut last_render_steps = 0u64;
    let mut last_render_reward = 0.0f64;
    let mut last_render_avg = 0.0f64;
    let mut last_render_score = 0u32;
    let mut last_render_kills = 0u8;

    // Episode stats for logging
    let mut recent_rewards: VecDeque<f64> = VecDeque::with_capacity(100);

    while total_steps < args.timesteps {
        episode += 1;
        let mut state = env.reset()?;
        env.press_start(60)?;
        let mut ep_reward = 0.0;
        let mut ep_steps = 0u64;
        let mut ep_loss = 0.0f32;
        let mut loss_count = 0u32;

        loop {
            if let Some(win) = window.as_ref() {
                update_pause_from_window(&mut env, win);
                if !win.is_open() {
                    eprintln!("\nRender window closed. Exiting training loop.");
                    return Ok(());
                }
            }
            env.step_pause()?;
            if env.is_paused() {
                continue;
            }
            let action_idx = agent.select_action(&state)?;
            let result = env.step(Action::from_index(action_idx))?;

            agent.remember(Transition {
                state: state.clone(),
                action: action_idx,
                reward: result.reward,
                next_state: result.state.clone(),
                done: result.done,
            });

            let loss = agent.train_step()?;
            if loss > 0.0 {
                ep_loss += loss;
                loss_count += 1;
            }

            state = result.state;
            ep_reward += result.reward as f64;
            ep_steps += 1;
            total_steps += 1;

            if let Some(win) = window.as_mut() {
                let overlay_ep = if last_render_ep == 0 {
                    episode
                } else {
                    last_render_ep
                };
                let overlay_steps = if last_render_steps == 0 {
                    total_steps
                } else {
                    last_render_steps
                };
                let overlay_reward = if last_render_ep == 0 {
                    ep_reward
                } else {
                    last_render_reward
                };
                let overlay_avg = if last_render_ep == 0 {
                    0.0
                } else {
                    last_render_avg
                };
                let overlay_score = if last_render_ep == 0 {
                    env.prev_state.score
                } else {
                    last_render_score
                };
                let overlay_kills = if last_render_ep == 0 {
                    env.prev_state.kill_count
                } else {
                    last_render_kills
                };
                win.set_title(&format!(
                    "Kung Fu Master — Training | Ep {overlay_ep} | Steps {overlay_steps} | R {overlay_reward:.1} | Avg100 {overlay_avg:.1} | Score {overlay_score} | Kills {overlay_kills}"
                ));
                let fb = env.frame_buffer();
                let mut buf = vec![0u32; 256 * 240];
                for (i, pixel) in buf.iter_mut().enumerate() {
                    let base = i * 4;
                    if base + 2 < fb.len() {
                        let r = fb[base] as u32;
                        let g = fb[base + 1] as u32;
                        let b = fb[base + 2] as u32;
                        *pixel = (r << 16) | (g << 8) | b;
                    }
                }
                win.update_with_buffer(&buf, 256, 240)?;
            }

            if result.done || ep_steps > 10_000 {
                break;
            }

            // Periodic save
            if total_steps % 50_000 == 0 {
                agent.save(&format!("checkpoints/step_{total_steps}.safetensors"))?;
            }
        }

        recent_rewards.push_back(ep_reward);
        if recent_rewards.len() > 100 {
            recent_rewards.pop_front();
        }
        let avg_reward: f64 = recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64;

        let avg_loss = if loss_count > 0 {
            ep_loss / loss_count as f32
        } else {
            0.0
        };

        if ep_reward > best_reward {
            best_reward = ep_reward;
            agent.save("checkpoints/best.safetensors")?;
        }

        let elapsed = t_start.elapsed().as_secs_f64();
        let fps = total_steps as f64 / elapsed;

        if episode % 10 == 0 || ep_reward > best_reward - 1.0 {
            eprintln!(
                "Ep {episode:>5} | Steps {total_steps:>8} | R {ep_reward:>8.1} | \
                 Avg100 {avg_reward:>7.1} | Score {score:>6} | Kills {kills:>3} | \
                 ε {eps:.4} | Loss {loss:.5} | FPS {fps:.0}",
                score = env.prev_state.score,
                kills = env.prev_state.kill_count,
                eps = agent.epsilon,
                loss = avg_loss,
            );
        }

        last_render_ep = episode;
        last_render_steps = total_steps;
        last_render_reward = ep_reward;
        last_render_avg = avg_reward;
        last_render_score = env.prev_state.score;
        last_render_kills = env.prev_state.kill_count;
    }

    agent.save("checkpoints/final.safetensors")?;
    eprintln!(
        "\n✅ Training complete. {total_steps} steps in {:.1}s",
        t_start.elapsed().as_secs_f64()
    );
    Ok(())
}

// =============================================================================
// Section 8: Play / Evaluate with Visualization
// =============================================================================

fn play(args: &PlayArgs) -> Result<()> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  PLAYING — Kung Fu Master DQN Agent");
    eprintln!("═══════════════════════════════════════════════════════════");

    let device = Device::new_metal(0).unwrap_or(Device::Cpu);

    let mut env = NesEnv::new(args.rom.clone(), 4, 0.0)?;
    let mut agent = DqnAgent::new(&device)?;
    agent.load(&args.model.to_string_lossy())?;
    agent.epsilon = 0.0; // Greedy during evaluation
    env.press_start(60)?;

    // Create display window (NES native res: 256×240)
    let mut window = minifb::Window::new(
        "Kung Fu Master — RL Agent",
        256,
        240,
        minifb::WindowOptions {
            resize: true,
            scale: minifb::Scale::X2,
            ..Default::default()
        },
    )?;
    window.set_target_fps(60);

    for ep in 0..args.episodes {
        let mut state = env.reset()?;
        env.press_start(60)?;
        let mut total_reward = 0.0;
        let mut steps = 0u64;

        loop {
            update_pause_from_window(&mut env, &window);
            env.step_pause()?;
            if env.is_paused() {
                continue;
            }
            let action_idx = agent.select_action(&state)?;
            let result = env.step(Action::from_index(action_idx))?;

            state = result.state;
            total_reward += result.reward as f64;
            steps += 1;

            // Render: convert tetanes RGBA frame to minifb u32 (0xAARRGGBB)
            let fb = env.frame_buffer();
            let mut buf = vec![0u32; 256 * 240];
            for (i, pixel) in buf.iter_mut().enumerate() {
                let base = i * 4; // tetanes outputs 4 bytes per pixel
                if base + 2 < fb.len() {
                    let r = fb[base] as u32;
                    let g = fb[base + 1] as u32;
                    let b = fb[base + 2] as u32;
                    *pixel = (r << 16) | (g << 8) | b;
                }
            }
            window.update_with_buffer(&buf, 256, 240)?;

            if result.done || steps > 20_000 || !window.is_open() {
                break;
            }
        }

        eprintln!(
            "Episode {}: reward={total_reward:.1}, steps={steps}, score={}, kills={}",
            ep + 1,
            env.prev_state.score,
            env.prev_state.kill_count,
        );

        if !window.is_open() {
            break;
        }
    }

    Ok(())
}

// =============================================================================
// Section 9: RAM Explorer
// =============================================================================

fn explore(args: &ExploreArgs) -> Result<()> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  RAM EXPLORER — Kung Fu (NES via tetanes-core)");
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("Watch RAM values change as the game runs.");
    eprintln!("Press Start via keyboard if tetanes supports it,");
    eprintln!("or modify this to inject Start presses automatically.");
    eprintln!("═══════════════════════════════════════════════════════════");

    let mut env = NesEnv::new(args.rom.clone(), 1, 0.0)?;

    let mut window = minifb::Window::new(
        "Kung Fu — RAM Explorer",
        256,
        240,
        minifb::WindowOptions {
            resize: true,
            scale: minifb::Scale::X2,
            ..Default::default()
        },
    )?;
    window.set_target_fps(60);

    let watched: Vec<(&str, u16)> = vec![
        ("Player X", ram::PLAYER_X),
        ("Player Y", ram::PLAYER_Y),
        ("Player HP", ram::PLAYER_HP),
        ("Player Lives", ram::PLAYER_LIVES),
        ("Player Pose", ram::PLAYER_POSE),
        ("Player State", ram::PLAYER_STATE),
        ("Kill Count", ram::KILL_COUNTER),
        ("Enemy0 X", ram::ENEMY_X[0]),
        ("Enemy0 Type", ram::ENEMY_TYPE[0]),
        ("Boss HP", ram::BOSS_HP.unwrap_or(0x0000)),
        ("Floor", ram::FLOOR),
    ];

    let mut prev_vals: Vec<u8> = watched.iter().map(|_| 0).collect();
    let mut frame = 0u64;

    // Auto-press Start to get past title
    env.press_start(120)?;

    while window.is_open() {
        update_pause_from_window(&mut env, &window);
        if env.is_paused() {
            env.step_pause()?;
            continue;
        }
        let mut btn_state = tetanes_core::input::JoypadBtnState::empty();
        if window.is_key_down(minifb::Key::Left) {
            btn_state.set(tetanes_core::input::JoypadBtnState::LEFT, true);
        }
        if window.is_key_down(minifb::Key::Right) {
            btn_state.set(tetanes_core::input::JoypadBtnState::RIGHT, true);
        }
        if window.is_key_down(minifb::Key::Up) {
            btn_state.set(tetanes_core::input::JoypadBtnState::UP, true);
        }
        if window.is_key_down(minifb::Key::Down) {
            btn_state.set(tetanes_core::input::JoypadBtnState::DOWN, true);
        }
        if window.is_key_down(minifb::Key::Z) {
            btn_state.set(tetanes_core::input::JoypadBtnState::B, true);
        }
        if window.is_key_down(minifb::Key::X) {
            btn_state.set(tetanes_core::input::JoypadBtnState::A, true);
        }
        if window.is_key_down(minifb::Key::A) {
            btn_state.set(tetanes_core::input::JoypadBtnState::SELECT, true);
        }
        if window.is_key_down(minifb::Key::S) {
            btn_state.set(tetanes_core::input::JoypadBtnState::START, true);
        }

        env.set_input_state(btn_state);
        env.deck.clock_frame()?;
        frame += 1;

        // Render
        let fb = env.frame_buffer();
        let mut buf = vec![0u32; 256 * 240];
        for (i, pixel) in buf.iter_mut().enumerate() {
            let base = i * 4;
            if base + 2 < fb.len() {
                *pixel =
                    ((fb[base] as u32) << 16) | ((fb[base + 1] as u32) << 8) | fb[base + 2] as u32;
            }
        }
        window.update_with_buffer(&buf, 256, 240)?;

        // Print RAM every 30 frames (~0.5 sec)
        if frame % 30 == 0 {
            println!("\r\n--- Frame {frame} ---");
            println!("\r  Score: {}", env.read_score());
            println!("\r  Timer: {}", env.read_timer());
            println!("\r  Kill Count: {}", env.peek(ram::KILL_COUNTER));
            for (i, (name, addr)) in watched.iter().enumerate() {
                let val = env.peek(*addr);
                let changed = if val != prev_vals[i] {
                    " ← CHANGED"
                } else {
                    ""
                };
                println!("\r  {name:<14} [0x{addr:04X}] = {val:3} (0x{val:02X}){changed}");
                prev_vals[i] = val;
            }

            // Decode active enemies
            let type_names = [
                "Gripper",
                "Tiny Grip",
                "Knife Thr",
                "Stick Ftr",
                "Boomerang",
                "Bigman",
                "Magician",
                "Mr. X",
            ];
            for i in 0..4 {
                let pose = env.peek(ram::ENEMY_POSE[i]);
                if pose != 0 && pose != 0x7F {
                    let etype = env.peek(ram::ENEMY_TYPE[i]) as usize;
                    let ex = env.peek(ram::ENEMY_X[i]);
                    let ey = env.peek(ram::ENEMY_Y[i]);
                    let ef = env.peek(ram::ENEMY_FACING[i]);
                    let tname = type_names.get(etype).unwrap_or(&"Unknown");
                    let facing = if ef == 0 { "L" } else { "R" };
                    println!("\r  Enemy[{i}]: {tname:>9} @ ({ex},{ey}) facing={facing}");
                }
            }
        }
    }

    Ok(())
}

// =============================================================================
// Section 10: Random Baseline
// =============================================================================

fn baseline(args: &BaselineArgs) -> Result<()> {
    eprintln!("Running random agent baseline...");

    let mut env = NesEnv::new(args.rom.clone(), 4, 0.0)?;
    let mut rng = rand::rng();
    let mut rewards = Vec::new();

    for ep in 0..10 {
        let _state = env.reset()?;
        let mut total_reward = 0.0;
        let mut steps = 0u64;

        loop {
            env.step_pause()?;
            if env.is_paused() {
                continue;
            }
            let action = Action::from_index(rng.random_range(0..Action::COUNT));
            let result = env.step(action)?;
            total_reward += result.reward as f64;
            steps += 1;

            if result.done || steps > 10_000 {
                break;
            }
        }

        eprintln!(
            "Random ep {}: reward={total_reward:.1}, steps={steps}, score={}, kills={}",
            ep + 1,
            env.prev_state.score,
            env.prev_state.kill_count,
        );
        rewards.push(total_reward);
    }

    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let max = rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("\nBaseline: mean={mean:.1}, max={max:.1}");
    Ok(())
}

// =============================================================================
// Section 10.5: Manual Play (Keyboard)
// =============================================================================

fn manual(args: &ManualArgs) -> Result<()> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  MANUAL — Kung Fu Master (Keyboard)");
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("Arrows: Move | Z: B (Punch) | X: A (Kick) | A: Select | S: Start");
    eprintln!("Space: Pause | Esc: Quit");

    let mut env = NesEnv::new(args.rom.clone(), 1, 0.0)?;
    let _ = env.reset()?;
    env.press_start(60)?;

    let mut window = minifb::Window::new(
        "Kung Fu Master — Manual",
        256,
        240,
        minifb::WindowOptions {
            resize: true,
            scale: minifb::Scale::X2,
            ..Default::default()
        },
    )?;
    window.set_target_fps(60);

    while window.is_open() {
        update_pause_from_window(&mut env, &window);
        if window.is_key_pressed(minifb::Key::Escape, minifb::KeyRepeat::No) {
            break;
        }
        if window.is_key_pressed(minifb::Key::Space, minifb::KeyRepeat::No) {
            env.toggle_pause();
        }

        let mut btn_state = tetanes_core::input::JoypadBtnState::empty();
        if window.is_key_down(minifb::Key::Left) {
            btn_state.set(tetanes_core::input::JoypadBtnState::LEFT, true);
        }
        if window.is_key_down(minifb::Key::Right) {
            btn_state.set(tetanes_core::input::JoypadBtnState::RIGHT, true);
        }
        if window.is_key_down(minifb::Key::Up) {
            btn_state.set(tetanes_core::input::JoypadBtnState::UP, true);
        }
        if window.is_key_down(minifb::Key::Down) {
            btn_state.set(tetanes_core::input::JoypadBtnState::DOWN, true);
        }
        if window.is_key_down(minifb::Key::Z) {
            btn_state.set(tetanes_core::input::JoypadBtnState::B, true);
        }
        if window.is_key_down(minifb::Key::X) {
            btn_state.set(tetanes_core::input::JoypadBtnState::A, true);
        }
        if window.is_key_down(minifb::Key::A) {
            btn_state.set(tetanes_core::input::JoypadBtnState::SELECT, true);
        }
        if window.is_key_down(minifb::Key::S) {
            btn_state.set(tetanes_core::input::JoypadBtnState::START, true);
        }

        if !env.is_paused() {
            env.set_input_state(btn_state);
            env.deck.clock_frame()?;
        } else {
            env.step_pause()?;
        }

        let fb = env.frame_buffer();
        let mut buf = vec![0u32; 256 * 240];
        for (i, pixel) in buf.iter_mut().enumerate() {
            let base = i * 4;
            if base + 2 < fb.len() {
                let r = fb[base] as u32;
                let g = fb[base + 1] as u32;
                let b = fb[base + 2] as u32;
                *pixel = (r << 16) | (g << 8) | b;
            }
        }
        window.update_with_buffer(&buf, 256, 240)?;
    }

    Ok(())
}

// =============================================================================
// Section 11: CLI
// =============================================================================

#[derive(Parser)]
#[command(name = "kungfu-rl", about = "Kung Fu Master NES — DQN RL Agent")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Set up and verify the environment
    Explore(ExploreArgs),
    /// Train the DQN agent
    Train(TrainArgs),
    /// Watch the trained agent play
    Play(PlayArgs),
    /// Play manually with keyboard
    Manual(ManualArgs),
    /// Run random agent baseline
    Baseline(BaselineArgs),
}

#[derive(Parser)]
struct ExploreArgs {
    #[arg(long)]
    rom: PathBuf,
}

#[derive(Parser)]
struct TrainArgs {
    #[arg(long)]
    rom: PathBuf,
    #[arg(long, default_value = "2000000")]
    timesteps: u64,
    #[arg(long, default_value = "4")]
    frame_skip: u32,
    #[arg(long, default_value_t = false)]
    render: bool,
}

#[derive(Parser)]
struct PlayArgs {
    #[arg(long)]
    rom: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "5")]
    episodes: usize,
}

#[derive(Parser)]
struct ManualArgs {
    #[arg(long)]
    rom: PathBuf,
}

#[derive(Parser)]
struct BaselineArgs {
    #[arg(long)]
    rom: PathBuf,
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".to_string()))
        .init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Explore(args) => explore(args),
        Commands::Train(args) => train(args),
        Commands::Play(args) => play(args),
        Commands::Manual(args) => manual(args),
        Commands::Baseline(args) => baseline(args),
    }
}
