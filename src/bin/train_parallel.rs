#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Context, Result};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Linear, Module, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock, mpsc};
use std::time::{Duration, Instant};
use tetanes_core::mem::Read;
use tetanes_core::prelude::*;

// =============================================================================
// RAM Addresses
// =============================================================================

mod ram {
    pub const PLAYER_LIVES: u16 = 0x005C;
    pub const PLAYER_X: u16 = 0x00D4;
    pub const PLAYER_Y: u16 = 0x00B6;
    pub const PLAYER_HP: u16 = 0x04A6;
    #[allow(dead_code)]
    pub const PLAYER_POSE: u16 = 0x036E;
    pub const PLAYER_STATE: u16 = 0x036F;
    pub const PAGE: u16 = 0x0065;
    pub const GAME_MODE: u16 = 0x0062;
    pub const START_TIMER: u16 = 0x003A;
    pub const ENEMY_X: [u16; 4] = [0x00CE, 0x00CF, 0x00D0, 0x00D1];
    pub const ENEMY_TYPE: [u16; 4] = [0x0087, 0x0088, 0x0089, 0x008A];
    pub const ENEMY_Y: [u16; 4] = [0x00B0, 0x00B1, 0x00B2, 0x00B3];
    pub const ENEMY_FACING: [u16; 4] = [0x00C0, 0x00C1, 0x00C2, 0x00C3];
    pub const ENEMY_POSE: [u16; 4] = [0x00DF, 0x00E0, 0x00E1, 0x00E2];
    pub const ENEMY_ENERGY: [u16; 4] = [0x04A0, 0x04A1, 0x04A2, 0x04A3];
    pub const KNIFE_X: [u16; 4] = [0x03D4, 0x03D5, 0x03D6, 0x03D7];
    pub const KNIFE_Y: [u16; 4] = [0x03D0, 0x03D1, 0x03D2, 0x03D3];
    pub const KNIFE_STATE: [u16; 4] = [0x03EC, 0x03ED, 0x03EE, 0x03EF];
    pub const KILL_COUNTER: u16 = 0x03B1;
    pub const SCORE_DIGITS: [u16; 6] = [0x0531, 0x0532, 0x0533, 0x0534, 0x0535, 0x0536];
    pub const TOP_SCORE_DIGITS: Option<[u16; 6]> =
        Some([0x0501, 0x0502, 0x0503, 0x0504, 0x0505, 0x0506]);
    pub const TIMER_DIGITS: [u16; 4] = [0x0390, 0x0391, 0x0392, 0x0393];
    pub const BOSS_HP: Option<u16> = None;
    pub const FLOOR: u16 = 0x005F;
}

// =============================================================================
// Action Space
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Action {
    Noop = 0,
    Right = 1,
    Left = 2,
    Crouch = 3,
    Jump = 4,
    RightPunch = 5,
    RightKick = 6,
    LeftPunch = 7,
    LeftKick = 8,
    CrouchPunch = 9,
    CrouchKick = 10,
    JumpPunch = 11,
    JumpKick = 12,
}

impl Action {
    pub const COUNT: usize = 13;

    pub fn from_index(i: usize) -> Self {
        assert!(i < Self::COUNT);
        // SAFETY: repr(u8) and bounds checked.
        unsafe { std::mem::transmute(i as u8) }
    }

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
            Action::Crouch => {
                state.set(JoypadBtn::Down.into(), true);
            }
            Action::Jump => {
                state.set(JoypadBtn::Up.into(), true);
            }
            Action::RightPunch => {
                state.set(JoypadBtn::Right.into(), true);
                state.set(JoypadBtn::B.into(), true);
            }
            Action::RightKick => {
                state.set(JoypadBtn::Right.into(), true);
                state.set(JoypadBtn::A.into(), true);
            }
            Action::LeftPunch => {
                state.set(JoypadBtn::Left.into(), true);
                state.set(JoypadBtn::B.into(), true);
            }
            Action::LeftKick => {
                state.set(JoypadBtn::Left.into(), true);
                state.set(JoypadBtn::A.into(), true);
            }
            Action::CrouchPunch => {
                state.set(JoypadBtn::Down.into(), true);
                state.set(JoypadBtn::B.into(), true);
            }
            Action::CrouchKick => {
                state.set(JoypadBtn::Down.into(), true);
                state.set(JoypadBtn::A.into(), true);
            }
            Action::JumpPunch => {
                state.set(JoypadBtn::Up.into(), true);
                state.set(JoypadBtn::B.into(), true);
            }
            Action::JumpKick => {
                state.set(JoypadBtn::Up.into(), true);
                state.set(JoypadBtn::A.into(), true);
            }
        }
        state
    }
}

// =============================================================================
// Game State
// =============================================================================

#[derive(Debug, Clone, Default)]
pub struct GameState {
    pub player_x: u8,
    pub player_y: u8,
    pub player_hp: u8,
    pub player_lives: u8,
    pub player_state: u8,
    pub page: u8,
    pub game_mode: u8,
    pub start_timer: u8,
    pub score: u32,
    pub top_score: u32,
    pub kill_count: u8,
    pub enemy_x: [u8; 4],
    pub enemy_y: [u8; 4],
    pub enemy_type: [u8; 4],
    pub enemy_facing: [u8; 4],
    pub enemy_active: [bool; 4],
    pub enemy_energy: [u8; 4],
    pub knife_x: [u8; 4],
    pub knife_y: [u8; 4],
    pub knife_facing: [u8; 4],
    pub knife_active: [bool; 4],
    pub boss_hp: u8,
    pub floor: u8,
    pub timer: u16,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RewardBreakdown {
    pub score: f64,
    pub hp: f64,
    pub energy: f64,
    pub movement: f64,
    pub floor: f64,
    pub boss: f64,
    pub time: f64,
    pub death: f64,
}

pub const STATE_DIM: usize = 49;

impl GameState {
    pub fn to_features(&self) -> [f32; STATE_DIM] {
        let mut f = [0f32; STATE_DIM];
        let mut idx = 0;

        f[idx] = self.player_x as f32 / 255.0;
        idx += 1;
        f[idx] = self.player_y as f32 / 255.0;
        idx += 1;
        let hp = if self.player_hp == 0xFF {
            0.0
        } else {
            self.player_hp as f32
        };
        f[idx] = hp / 48.0;
        idx += 1;
        f[idx] = self.player_lives as f32 / 5.0;
        idx += 1;
        f[idx] = self.player_state as f32 / 255.0;
        idx += 1;
        f[idx] = self.floor as f32 / 5.0;
        idx += 1;

        #[derive(Clone, Copy)]
        struct EF {
            sort_key: f32,
            active: f32,
            dx: f32,
            dy: f32,
            abs_dx: f32,
            etype: f32,
            facing: f32,
        }

        let mut enemies = [EF {
            sort_key: f32::INFINITY,
            active: 0.0,
            dx: 0.0,
            dy: 0.0,
            abs_dx: 0.0,
            etype: 0.0,
            facing: 0.0,
        }; 4];
        for i in 0..4 {
            if self.enemy_active[i] {
                let dx_raw = self.enemy_x[i] as f32 - self.player_x as f32;
                let dy_raw = self.enemy_y[i] as f32 - self.player_y as f32;
                let abs_dx = dx_raw.abs() / 255.0;
                enemies[i] = EF {
                    sort_key: abs_dx,
                    active: 1.0,
                    dx: dx_raw / 255.0,
                    dy: dy_raw / 255.0,
                    abs_dx,
                    etype: self.enemy_type[i] as f32 / 7.0,
                    facing: self.enemy_facing[i] as f32,
                };
            }
        }
        enemies.sort_unstable_by(|a, b| {
            a.sort_key
                .partial_cmp(&b.sort_key)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for enemy in &enemies {
            f[idx] = enemy.active;
            idx += 1;
            f[idx] = enemy.dx;
            idx += 1;
            f[idx] = enemy.dy;
            idx += 1;
            f[idx] = enemy.abs_dx;
            idx += 1;
            f[idx] = enemy.etype;
            idx += 1;
            f[idx] = enemy.facing;
            idx += 1;
        }

        for i in 0..4 {
            let active = self.knife_active[i];
            f[idx] = if active { 1.0 } else { 0.0 };
            idx += 1;
            if active {
                f[idx] = self.knife_x[i] as f32 / 255.0;
                idx += 1;
                f[idx] = self.knife_y[i] as f32 / 255.0;
                idx += 1;
                f[idx] = self.knife_facing[i] as f32;
                idx += 1;
            } else {
                f[idx] = 0.0;
                idx += 1;
                f[idx] = 0.0;
                idx += 1;
                f[idx] = 0.0;
                idx += 1;
            }
        }

        f[idx] = self.boss_hp as f32 / 255.0;
        idx += 1;
        f[idx] = (self.timer.min(9999) as f32) / 9999.0;
        idx += 1;
        f[idx] = self.kill_count as f32 / 255.0;
        f
    }

    fn global_x(&self) -> i32 {
        (self.page as i32) * 256 + self.player_x as i32
    }
}

// =============================================================================
// NES Environment (headless only)
// =============================================================================

const GAME_MODE_TITLE: u8 = 0x00;
const GAME_MODE_COUNTDOWN: u8 = 0x01;
const GAME_MODE_ACTION: u8 = 0x02;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionState {
    Startup,
    WaitForTitle,
    WaitForMode2,
    WaitForCountdown,
    Playing,
}

pub struct NesEnv {
    deck: ControlDeck,
    prev_state: GameState,
    total_reward: f64,
    steps: u64,
    frame_skip: u32,
    sticky_prob: f64,
    last_action: Action,
    clock_enabled: bool,
    real_time: bool,
    next_frame_deadline: Option<Instant>,
    session_state: SessionState,
    countdown_seen: bool,
    debug_state: bool,
    reward_debug: bool,
    reward_breakdown: RewardBreakdown,
    progress_floor: u8,
    progress_start_global_x: i32,
    progress_best: i32,
    rng: SmallRng,
}

impl NesEnv {
    pub fn new(rom_path: PathBuf, frame_skip: u32, sticky_prob: f64) -> Result<Self> {
        let mut deck = ControlDeck::new();
        let headless_mode = tetanes_core::control_deck::HeadlessMode::NO_AUDIO
            | tetanes_core::control_deck::HeadlessMode::NO_VIDEO;
        deck.set_headless_mode(headless_mode);
        deck.load_rom_path(&rom_path)
            .with_context(|| format!("Failed to load ROM: {}", rom_path.display()))?;

        let debug_state = Self::debug_state_enabled();
        let reward_debug = Self::debug_reward_enabled();

        Ok(Self {
            deck,
            prev_state: GameState::default(),
            total_reward: 0.0,
            steps: 0,
            frame_skip,
            sticky_prob,
            last_action: Action::Noop,
            clock_enabled: true,
            real_time: false,
            next_frame_deadline: None,
            session_state: SessionState::Startup,
            countdown_seen: false,
            debug_state,
            reward_debug,
            reward_breakdown: RewardBreakdown::default(),
            progress_floor: 0,
            progress_start_global_x: 0,
            progress_best: 0,
            rng: SmallRng::from_os_rng(),
        })
    }

    fn debug_state_enabled() -> bool {
        match std::env::var("KFM_DEBUG_STATE") {
            Ok(val) => matches!(val.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
            Err(_) => false,
        }
    }

    fn debug_reward_enabled() -> bool {
        match std::env::var("KFM_DEBUG_REWARD") {
            Ok(val) => matches!(val.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
            Err(_) => false,
        }
    }

    pub fn reward_debug_enabled(&self) -> bool {
        self.reward_debug
    }

    pub fn clear_reward_breakdown(&mut self) {
        self.reward_breakdown = RewardBreakdown::default();
    }

    pub fn reward_breakdown(&self) -> RewardBreakdown {
        self.reward_breakdown
    }

    fn is_action_mode(mode: u8) -> bool {
        mode == GAME_MODE_ACTION || mode == GAME_MODE_COUNTDOWN
    }

    fn log_state(&self, tag: &str, state: &GameState) {
        if !self.debug_state {
            return;
        }
        eprintln!(
            "[state:{tag}] phase={phase:?} mode=0x{mode:02X} start_timer={start_timer} lives={lives} hp={hp} score={score} timer={timer} floor={floor}",
            phase = self.session_state,
            mode = state.game_mode,
            start_timer = state.start_timer,
            lives = state.player_lives,
            hp = state.player_hp,
            score = state.score,
            timer = state.timer,
            floor = state.floor,
        );
    }

    pub fn set_clock_enabled(&mut self, enabled: bool) {
        self.clock_enabled = enabled;
    }

    pub fn set_real_time(&mut self, enabled: bool) {
        self.real_time = enabled;
        self.next_frame_deadline = None;
    }

    fn run_state_machine(&mut self, max_frames: u32) -> Result<GameState> {
        const START_PRESS_FRAMES: u32 = 2;
        const START_PRESS_INTERVAL: u32 = 30;

        let mut frames = 0u32;
        let mut since_press = START_PRESS_INTERVAL;
        let mut last_mode: Option<u8> = None;
        let mut last_phase: Option<SessionState> = None;

        loop {
            let state = self.read_state();
            if state.start_timer > 0 {
                self.countdown_seen = true;
            }
            if self.debug_state
                && (last_mode != Some(state.game_mode)
                    || last_phase != Some(self.session_state)
                    || frames.is_multiple_of(120))
            {
                self.log_state("sm", &state);
                last_mode = Some(state.game_mode);
                last_phase = Some(self.session_state);
            }

            match self.session_state {
                SessionState::Startup => {
                    if state.game_mode == GAME_MODE_TITLE {
                        self.session_state = SessionState::WaitForMode2;
                        self.countdown_seen = false;
                    } else if Self::is_action_mode(state.game_mode) {
                        if state.start_timer == 0 {
                            if self.countdown_seen {
                                self.session_state = SessionState::Playing;
                                return Ok(state);
                            }
                            self.press_start(START_PRESS_FRAMES)?;
                            frames = frames.saturating_add(START_PRESS_FRAMES);
                            self.session_state = SessionState::WaitForTitle;
                        } else {
                            self.session_state = SessionState::WaitForCountdown;
                        }
                    } else {
                        self.clock_frame()?;
                        frames = frames.saturating_add(1);
                    }
                }
                SessionState::WaitForTitle => {
                    if state.game_mode == GAME_MODE_TITLE {
                        self.session_state = SessionState::WaitForMode2;
                        since_press = START_PRESS_INTERVAL;
                        self.countdown_seen = false;
                    } else {
                        self.clock_frame()?;
                        frames = frames.saturating_add(1);
                    }
                }
                SessionState::WaitForMode2 => {
                    if state.game_mode == GAME_MODE_TITLE {
                        if since_press >= START_PRESS_INTERVAL {
                            if self.debug_state {
                                eprintln!("[state:sm] press start");
                            }
                            self.press_start(START_PRESS_FRAMES)?;
                            frames = frames.saturating_add(START_PRESS_FRAMES);
                            since_press = 0;
                        } else {
                            self.clock_frame()?;
                            frames = frames.saturating_add(1);
                            since_press += 1;
                        }
                    } else if state.game_mode == GAME_MODE_COUNTDOWN {
                        self.session_state = SessionState::WaitForCountdown;
                    } else if state.game_mode == GAME_MODE_ACTION {
                        if state.start_timer == 0 {
                            if self.countdown_seen {
                                self.session_state = SessionState::Playing;
                                return Ok(state);
                            }
                            if self.debug_state {
                                eprintln!("[state:sm] demo detected, press start");
                            }
                            self.press_start(START_PRESS_FRAMES)?;
                            frames = frames.saturating_add(START_PRESS_FRAMES);
                            self.session_state = SessionState::WaitForTitle;
                            since_press = START_PRESS_INTERVAL;
                        } else {
                            self.session_state = SessionState::WaitForCountdown;
                        }
                    } else {
                        self.clock_frame()?;
                        frames = frames.saturating_add(1);
                    }
                }
                SessionState::WaitForCountdown => {
                    if state.game_mode == GAME_MODE_TITLE {
                        self.session_state = SessionState::WaitForMode2;
                        since_press = START_PRESS_INTERVAL;
                        self.countdown_seen = false;
                    } else if Self::is_action_mode(state.game_mode) {
                        if state.start_timer == 0 {
                            self.session_state = SessionState::Playing;
                            return Ok(state);
                        }
                        self.clock_frame()?;
                        frames = frames.saturating_add(1);
                    } else {
                        self.clock_frame()?;
                        frames = frames.saturating_add(1);
                    }
                }
                SessionState::Playing => {
                    return Ok(state);
                }
            }

            if frames >= max_frames {
                return Ok(state);
            }
        }
    }

    fn clock_frame(&mut self) -> Result<()> {
        if !self.clock_enabled {
            return Ok(());
        }
        self.deck.clock_frame()?;
        if self.real_time {
            self.throttle_frame();
        }
        Ok(())
    }

    fn throttle_frame(&mut self) {
        let frame_duration = Duration::from_nanos(1_000_000_000 / 60);
        let now = Instant::now();
        match self.next_frame_deadline {
            Some(deadline) if deadline > now => {
                std::thread::sleep(deadline - now);
                self.next_frame_deadline = Some(deadline + frame_duration);
            }
            _ => {
                self.next_frame_deadline = Some(now + frame_duration);
            }
        }
    }

    fn peek(&self, addr: u16) -> u8 {
        self.deck.bus().peek(addr)
    }

    fn read_score(&self) -> u32 {
        self.read_score_digits(&ram::SCORE_DIGITS)
    }

    fn read_score_digits(&self, digits: &[u16; 6]) -> u32 {
        let mut score = 0u32;
        let multipliers = [100_000, 10_000, 1_000, 100, 10, 1];
        for (i, &addr) in digits.iter().enumerate() {
            let digit = self.peek(addr) & 0x0F;
            score += digit as u32 * multipliers[i];
        }
        score
    }

    fn read_timer(&self) -> u16 {
        let mut timer = 0u16;
        let multipliers = [1000, 100, 10, 1];
        for (i, &addr) in ram::TIMER_DIGITS.iter().enumerate() {
            let digit = self.peek(addr) & 0x0F;
            timer += digit as u16 * multipliers[i];
        }
        timer
    }

    fn read_state(&self) -> GameState {
        let mut state = GameState::default();
        state.player_x = self.peek(ram::PLAYER_X);
        state.player_y = self.peek(ram::PLAYER_Y);
        state.player_hp = self.peek(ram::PLAYER_HP);
        state.player_lives = self.peek(ram::PLAYER_LIVES);
        state.player_state = self.peek(ram::PLAYER_STATE);
        state.page = self.peek(ram::PAGE);
        state.game_mode = self.peek(ram::GAME_MODE);
        state.start_timer = self.peek(ram::START_TIMER);
        state.score = self.read_score();
        state.top_score = ram::TOP_SCORE_DIGITS
            .as_ref()
            .map(|digits| self.read_score_digits(digits))
            .unwrap_or(0);
        state.kill_count = self.peek(ram::KILL_COUNTER);
        state.boss_hp = ram::BOSS_HP.map(|addr| self.peek(addr)).unwrap_or(0);
        state.floor = self.peek(ram::FLOOR);
        state.timer = self.read_timer();

        for i in 0..4 {
            state.enemy_x[i] = self.peek(ram::ENEMY_X[i]);
            state.enemy_y[i] = self.peek(ram::ENEMY_Y[i]);
            state.enemy_type[i] = self.peek(ram::ENEMY_TYPE[i]);
            state.enemy_facing[i] = self.peek(ram::ENEMY_FACING[i]);
            let pose = self.peek(ram::ENEMY_POSE[i]);
            state.enemy_active[i] = pose != 0 && pose != 0x7F;
            state.enemy_energy[i] = self.peek(ram::ENEMY_ENERGY[i]);

            state.knife_x[i] = self.peek(ram::KNIFE_X[i]);
            state.knife_y[i] = self.peek(ram::KNIFE_Y[i]);
            let knife_state = self.peek(ram::KNIFE_STATE[i]);
            state.knife_active[i] = knife_state != 0;
            state.knife_facing[i] = match knife_state {
                0x11 => 1,
                0x01 => 0,
                _ => 0,
            };
        }
        state
    }

    fn set_input(&mut self, action: Action) {
        let btn_state = action.to_joypad();
        self.set_input_state(btn_state);
    }

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

    fn press_start(&mut self, frames: u32) -> Result<()> {
        use tetanes_core::input::JoypadBtnState;
        let mut btn_state = JoypadBtnState::empty();
        btn_state.set(JoypadBtnState::START, true);
        for _ in 0..frames {
            self.set_input_state(btn_state);
            self.clock_frame()?;
        }
        self.set_input_state(JoypadBtnState::empty());
        Ok(())
    }

    pub fn reset(&mut self) -> Result<[f32; STATE_DIM]> {
        self.deck.reset(ResetKind::Soft);
        self.session_state = SessionState::Startup;
        self.countdown_seen = false;

        let noops = self.rng.random_range(1..30);
        for _ in 0..noops {
            self.clock_frame()?;
        }

        let state = self.run_state_machine(600)?;
        self.log_state("reset", &state);
        if self.session_state != SessionState::Playing {
            return Err(anyhow::anyhow!(
                "Timed out waiting for play state (phase: {phase:?}, mode: 0x{mode:02X}, start_timer: {timer})",
                phase = self.session_state,
                mode = state.game_mode,
                timer = state.start_timer
            ));
        }
        self.prev_state = state;
        self.total_reward = 0.0;
        self.steps = 0;
        self.last_action = Action::Noop;
        self.progress_floor = self.prev_state.floor;
        self.progress_start_global_x = self.prev_state.global_x();
        self.progress_best = 0;
        Ok(self.prev_state.to_features())
    }

    fn compute_reward(&mut self, cur: &GameState) -> f64 {
        let prev = &self.prev_state;
        let mut reward = 0.0;
        let mut energy_reward = 0.0;
        let mut score_reward = 0.0;
        let mut hp_penalty = 0.0;
        let mut movement_reward = 0.0;
        let mut floor_bonus = 0.0;
        let mut boss_reward = 0.0;
        let time_penalty = -0.001;

        for i in 0..4 {
            if prev.enemy_active[i]
                && cur.enemy_active[i]
                && prev.enemy_type[i] == cur.enemy_type[i]
            {
                let prev_energy = prev.enemy_energy[i];
                let cur_energy = cur.enemy_energy[i];
                if prev_energy > 0 && cur_energy < prev_energy && cur_energy != 0xFF {
                    let drop = prev_energy - cur_energy;
                    if drop <= 4 {
                        energy_reward += drop as f64 * 2.0;
                    }
                }
            }
        }

        let score_delta = cur.score as i64 - prev.score as i64;
        if score_delta > 0 && score_delta < 5_000 {
            score_reward += (score_delta as f64 / 50.0).min(2.0);
        }

        if cur.player_hp != 0xFF && prev.player_hp != 0xFF {
            let hp_delta = cur.player_hp as i32 - prev.player_hp as i32;
            if hp_delta < 0 && hp_delta > -200 {
                hp_penalty += hp_delta as f64 * 0.25;
            }
        }

        let cur_global_x = cur.global_x();
        if cur.floor != self.progress_floor {
            self.progress_floor = cur.floor;
            self.progress_start_global_x = cur_global_x;
            self.progress_best = 0;
        } else {
            let progress = if cur.floor % 2 == 0 {
                self.progress_start_global_x - cur_global_x
            } else {
                cur_global_x - self.progress_start_global_x
            };
            if progress > self.progress_best {
                let delta = (progress - self.progress_best).min(128);
                if delta > 0 {
                    movement_reward += delta as f64 * 0.05;
                    self.progress_best = progress;
                }
            }
        }

        if cur.floor > prev.floor {
            floor_bonus += 100.0;
        }

        let boss_delta = prev.boss_hp as i32 - cur.boss_hp as i32;
        if boss_delta > 0 && boss_delta < 200 {
            boss_reward += boss_delta as f64 * 2.0;
        }

        reward += time_penalty;

        reward +=
            energy_reward + score_reward + hp_penalty + movement_reward + floor_bonus + boss_reward;

        if self.reward_debug {
            self.reward_breakdown.energy += energy_reward;
            self.reward_breakdown.score += score_reward;
            self.reward_breakdown.hp += hp_penalty;
            self.reward_breakdown.movement += movement_reward;
            self.reward_breakdown.floor += floor_bonus;
            self.reward_breakdown.boss += boss_reward;
            self.reward_breakdown.time += time_penalty;
        }

        reward
    }

    pub fn step(&mut self, action: Action) -> Result<StepResult> {
        self.steps += 1;

        let effective_action = if self.rng.random::<f64>() < self.sticky_prob {
            self.last_action
        } else {
            action
        };
        self.last_action = effective_action;

        let mut frame_reward = 0.0;
        let mut life_lost = false;
        let mut game_over = false;
        let mut playing = true;

        if self.session_state != SessionState::Playing {
            let state = self.run_state_machine(180)?;
            self.log_state("step-nonplay", &state);
            playing = false;
            let done = self.session_state != SessionState::Playing;
            self.prev_state = state;
            let active_enemies = (0..4).filter(|&i| self.prev_state.enemy_active[i]).count() as u8;
            return Ok(StepResult {
                state: self.prev_state.to_features(),
                reward: 0.0,
                done,
                life_lost: false,
                game_over: false,
                score: self.prev_state.score,
                total_reward: self.total_reward,
                kills: self.prev_state.kill_count,
                active_enemies,
                playing,
            });
        }

        for _ in 0..self.frame_skip {
            self.set_input(effective_action);
            self.clock_frame()?;

            let state = self.read_state();
            let reward = self.compute_reward(&state);
            frame_reward += reward;

            if state.player_lives < self.prev_state.player_lives && self.prev_state.player_lives > 0
            {
                frame_reward -= 100.0;
                if self.reward_debug {
                    self.reward_breakdown.death -= 100.0;
                }
                life_lost = true;
                if state.player_lives == 0 {
                    game_over = true;
                    self.session_state = SessionState::WaitForTitle;
                    self.countdown_seen = false;
                }
                self.prev_state = state;
                break;
            }

            self.prev_state = state;
        }

        if playing {
            self.total_reward += frame_reward;
        }

        let active_enemies = (0..4).filter(|&i| self.prev_state.enemy_active[i]).count() as u8;

        Ok(StepResult {
            state: self.prev_state.to_features(),
            reward: frame_reward as f32,
            done: life_lost,
            life_lost,
            game_over,
            score: self.prev_state.score,
            total_reward: self.total_reward,
            kills: self.prev_state.kill_count,
            active_enemies,
            playing,
        })
    }
}

pub struct StepResult {
    pub state: [f32; STATE_DIM],
    pub reward: f32,
    pub done: bool,
    pub life_lost: bool,
    pub game_over: bool,
    pub score: u32,
    pub total_reward: f64,
    pub kills: u8,
    pub active_enemies: u8,
    pub playing: bool,
}

// =============================================================================
// DQN Neural Network
// =============================================================================

pub struct DqnNet {
    fc1: Linear,
    fc2: Linear,
    value_fc: Linear,
    value_out: Linear,
    advantage_fc: Linear,
    advantage_out: Linear,
}

impl DqnNet {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(STATE_DIM, 256, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(256, 256, vs.pp("fc2"))?;
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

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.fc1.forward(x)?.relu()?;
        let h = self.fc2.forward(&h)?.relu()?;
        let v = self.value_fc.forward(&h)?.relu()?;
        let v = self.value_out.forward(&v)?;
        let a = self.advantage_fc.forward(&h)?.relu()?;
        let a = self.advantage_out.forward(&a)?;
        let a_mean = a.mean_keepdim(candle_core::D::Minus1)?;
        let q = v.broadcast_add(&a.broadcast_sub(&a_mean)?)?;
        Ok(q)
    }
}

// =============================================================================
// Replay Buffer & Transition
// =============================================================================

#[derive(Clone, Serialize, Deserialize)]
struct Transition {
    #[serde(with = "serde_big_array::BigArray")]
    state: [f32; STATE_DIM],
    action: usize,
    reward: f32,
    #[serde(with = "serde_big_array::BigArray")]
    next_state: [f32; STATE_DIM],
    done: bool,
}

#[derive(Serialize, Deserialize)]
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

    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let reader = std::io::BufReader::new(file);
        let replay = bincode::deserialize_from(reader)?;
        Ok(replay)
    }

    fn sample(&self, batch_size: usize, dev: &Device, rng: &mut SmallRng) -> Result<BatchTensors> {
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

#[derive(Serialize, Deserialize)]
struct TrainMeta {
    best_reward: f64,
    episode: u64,
    total_steps: u64,
    epsilon: f64,
    agent_steps: u64,
}

fn save_checkpoint(
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

fn save_recent_rewards(recent_rewards: &VecDeque<f64>, dir: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let rewards: Vec<f64> = recent_rewards.iter().copied().collect();
    let file = File::create(Path::new(dir).join("recent_rewards.json"))?;
    let writer = std::io::BufWriter::new(file);
    serde_json::to_writer(writer, &rewards)?;
    Ok(())
}

// =============================================================================
// AdamW Optimizer
// =============================================================================

struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

impl AdamW {
    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
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

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn backward_step(&mut self, loss: &Tensor, max_grad_norm: Option<f64>) -> Result<()> {
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

    fn save_state<P: AsRef<Path>>(&self, path: P) -> Result<()> {
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

    fn load_state<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
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

// =============================================================================
// Shared Weight Snapshot (main thread → workers)
// =============================================================================

struct WeightSnapshot {
    version: u64,
    data: Vec<(String, Vec<f32>, Vec<usize>)>,
}

type SharedWeights = Arc<RwLock<WeightSnapshot>>;

fn snapshot_varmap(varmap: &VarMap, version: u64) -> Result<WeightSnapshot> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("lock failed"))?;
    let mut entries = Vec::new();
    for (name, var) in data.iter() {
        let tensor = var.as_tensor().to_device(&Device::Cpu)?.detach();
        let shape: Vec<usize> = tensor.shape().dims().to_vec();
        let flat: Vec<f32> = tensor.flatten_all()?.to_vec1::<f32>()?;
        entries.push((name.clone(), flat, shape));
    }
    Ok(WeightSnapshot {
        version,
        data: entries,
    })
}

fn load_snapshot_into_varmap(snapshot: &WeightSnapshot, varmap: &VarMap) -> Result<()> {
    let mut data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("lock failed"))?;
    for (name, flat, shape) in &snapshot.data {
        if let Some(var) = data.get_mut(name) {
            let t = Tensor::from_vec(flat.clone(), shape.as_slice(), &Device::Cpu)?;
            var.set(&t)?;
        }
    }
    Ok(())
}

// =============================================================================
// Worker Thread
// =============================================================================

struct WorkerStats {
    episodes: u64,
    steps: u64,
    total_reward: f64,
    last_score: u32,
    last_kills: u8,
}

fn worker_thread(
    id: usize,
    rom_path: PathBuf,
    frame_skip: u32,
    sticky_prob: f64,
    tx: mpsc::SyncSender<Transition>,
    shared_weights: SharedWeights,
    shared_epsilon: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    stats_out: Arc<RwLock<WorkerStats>>,
) {
    if let Err(e) = worker_loop(
        id,
        rom_path,
        frame_skip,
        sticky_prob,
        tx,
        shared_weights,
        shared_epsilon,
        stop,
        stats_out,
    ) {
        eprintln!("[worker {id}] error: {e}");
    }
}

fn worker_loop(
    _id: usize,
    rom_path: PathBuf,
    frame_skip: u32,
    sticky_prob: f64,
    tx: mpsc::SyncSender<Transition>,
    shared_weights: SharedWeights,
    shared_epsilon: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    stats_out: Arc<RwLock<WorkerStats>>,
) -> Result<()> {
    let mut env = NesEnv::new(rom_path, frame_skip, sticky_prob)?;
    let mut rng = SmallRng::from_os_rng();

    let local_varmap = VarMap::new();
    let local_vb = VarBuilder::from_varmap(&local_varmap, DType::F32, &Device::Cpu);
    let local_net = DqnNet::new(local_vb)?;
    let mut local_weight_version = 0u64;

    {
        let snap = shared_weights.read().unwrap();
        if snap.version > 0 {
            load_snapshot_into_varmap(&snap, &local_varmap)?;
            local_weight_version = snap.version;
        }
    }

    let mut episodes = 0u64;
    let mut worker_steps = 0u64;
    let mut need_reset = true;
    let mut state = [0.0f32; STATE_DIM];

    while !stop.load(Ordering::Relaxed) {
        if need_reset {
            state = match env.reset() {
                Ok(s) => s,
                Err(_) => continue,
            };
            need_reset = false;
        }
        episodes += 1;
        let mut ep_reward = 0.0f64;
        let mut ep_steps = 0u64;

        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }

            if worker_steps.is_multiple_of(500) {
                let snap = shared_weights.read().unwrap();
                if snap.version > local_weight_version {
                    let _ = load_snapshot_into_varmap(&snap, &local_varmap);
                    local_weight_version = snap.version;
                }
            }

            let eps = f64::from_bits(shared_epsilon.load(Ordering::Relaxed));
            let action_idx = if rng.random::<f64>() < eps {
                rng.random_range(0..Action::COUNT)
            } else {
                let s = Tensor::from_slice(&state, (1, STATE_DIM), &Device::Cpu)?;
                let q = local_net.forward(&s)?;
                q.argmax(candle_core::D::Minus1)?
                    .squeeze(0)?
                    .to_scalar::<u32>()? as usize
            };

            let result = env.step(Action::from_index(action_idx))?;

            if result.playing {
                let t = Transition {
                    state,
                    action: action_idx,
                    reward: result.reward,
                    next_state: result.state,
                    done: result.done,
                };
                if tx.send(t).is_err() {
                    return Ok(());
                }
                ep_reward += result.reward as f64;
                ep_steps += 1;
                worker_steps += 1;
            }

            state = result.state;

            if result.done || ep_steps > 10_000 {
                if result.game_over {
                    need_reset = true;
                }
                break;
            }
        }

        if let Ok(mut stats) = stats_out.write() {
            stats.episodes = episodes;
            stats.steps = worker_steps;
            stats.total_reward = ep_reward;
            stats.last_score = env.prev_state.score;
            stats.last_kills = env.prev_state.kill_count;
        }
    }

    Ok(())
}

// =============================================================================
// DQN Agent (training only, main thread)
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
    epsilon_start: f64,
    epsilon_end: f64,
    epsilon_decay_steps: u64,
    tau: f64,
    max_grad_norm: f64,
    initial_lr: f64,
    lr_decay_start: u64,
    lr_decay_factor: f64,
    replay: ReplayBuffer,
    batch_size: usize,
    learn_start: usize,
    train_freq: u64,
    total_env_steps: u64,
    steps: u64,
    rng: SmallRng,
}

impl DqnAgent {
    fn new(device: &Device) -> Result<Self> {
        let online_varmap = VarMap::new();
        let target_varmap = VarMap::new();
        let online_vb = VarBuilder::from_varmap(&online_varmap, DType::F32, device);
        let target_vb = VarBuilder::from_varmap(&target_varmap, DType::F32, device);
        let online_net = DqnNet::new(online_vb)?;
        let target_net = DqnNet::new(target_vb)?;

        let initial_lr = 1e-4;
        let opt_params = ParamsAdamW {
            lr: initial_lr,
            weight_decay: 1e-5,
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
            gamma: 0.99,
            epsilon: 1.0,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 2_000_000,
            tau: 0.005,
            max_grad_norm: 10.0,
            initial_lr,
            lr_decay_start: 1_000_000,
            lr_decay_factor: 0.5,
            replay: ReplayBuffer::new(1_000_000),
            batch_size: 64,
            learn_start: 10_000,
            train_freq: 4,
            total_env_steps: 0,
            steps: 0,
            rng: SmallRng::from_os_rng(),
        };
        agent.hard_update_target()?;
        Ok(agent)
    }

    fn resume_from(&mut self, resume_dir: &Path) -> Result<(TrainMeta, ReplayBuffer)> {
        let model_path = resume_dir.join("model.safetensors");
        let target_path = resume_dir.join("target.safetensors");
        let optimizer_path = resume_dir.join("optimizer.safetensors");
        let replay_bin_path = resume_dir.join("replay.bin");
        let meta_path = resume_dir.join("meta.json");

        self.online_varmap.load(&model_path)?;
        self.target_varmap.load(&target_path)?;
        if let Err(err) = self.load_optimizer(&optimizer_path) {
            eprintln!(
                "Optimizer state load failed ({err}). Continuing with fresh optimizer state."
            );
        }
        let replay = ReplayBuffer::load(&replay_bin_path)?;

        let file = File::open(&meta_path)?;
        let reader = std::io::BufReader::new(file);
        let meta: TrainMeta = serde_json::from_reader(reader)?;
        Ok((meta, replay))
    }

    fn train_step(&mut self) -> Result<f32> {
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

        let q_all = self.online_net.forward(&batch.states)?;
        let actions_unsqueezed = batch.actions.unsqueeze(1)?;
        let q_values = q_all.gather(&actions_unsqueezed, 1)?.squeeze(1)?;

        let next_q_online = self.online_net.forward(&batch.next_states)?;
        let best_next_actions = next_q_online.argmax(candle_core::D::Minus1)?.unsqueeze(1)?;

        let next_q_target = self.target_net.forward(&batch.next_states)?;
        let next_q = next_q_target
            .gather(&best_next_actions.to_dtype(DType::I64)?, 1)?
            .squeeze(1)?;

        let discounted = next_q.affine(self.gamma, 0.0)?;
        let target = batch.rewards.add(&discounted.mul(&batch.not_dones)?)?;

        let diff = q_values.sub(&target.detach())?;
        let abs_diff = diff.abs()?;
        let ones = Tensor::ones_like(&abs_diff)?;
        let loss = abs_diff
            .lt(&ones)?
            .where_cond(
                &(diff.sqr()?.affine(0.5, 0.0)?),
                &(abs_diff.affine(1.0, -0.5)?),
            )?
            .mean_all()?;

        self.optimizer
            .backward_step(&loss, Some(self.max_grad_norm))?;

        self.soft_update_target()?;

        let progress = (self.total_env_steps as f64) / (self.epsilon_decay_steps as f64);
        self.epsilon =
            self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress.min(1.0);

        if self.total_env_steps > self.lr_decay_start {
            let decayed_lr = self.initial_lr * self.lr_decay_factor;
            self.optimizer.set_learning_rate(decayed_lr);
        }

        loss.to_scalar::<f32>().map_err(Into::into)
    }

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

    fn save(&self, path: &str) -> Result<()> {
        self.online_varmap.save(path)?;
        eprintln!("Model saved to {path}");
        Ok(())
    }

    #[allow(dead_code)]
    fn load(&mut self, path: &str) -> Result<()> {
        self.online_varmap.load(path)?;
        self.hard_update_target()?;
        eprintln!("Model loaded from {path}");
        Ok(())
    }

    fn save_optimizer<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.optimizer.save_state(path)
    }

    fn load_optimizer<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.optimizer.load_state(path)
    }
}

// =============================================================================
// CLI
// =============================================================================

#[derive(Parser)]
#[command(
    name = "train-parallel",
    about = "Parallel DQN training for Kung Fu Master"
)]
struct Args {
    #[arg(long)]
    rom: PathBuf,
    #[arg(long, default_value = "2000000")]
    timesteps: u64,
    #[arg(long, default_value = "4")]
    frame_skip: u32,
    #[arg(long, default_value_t = false)]
    cpu: bool,
    #[arg(long)]
    workers: Option<usize>,
    #[arg(long)]
    resume: Option<PathBuf>,
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".to_string()))
        .init();

    let args = Args::parse();

    let num_workers = args.workers.unwrap_or_else(|| {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        (cpus - 1).max(2)
    });

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  PARALLEL TRAINING — Kung Fu Master DQN ({num_workers} workers)");
    eprintln!("═══════════════════════════════════════════════════════════");

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    };
    eprintln!("Device: {:?}", device);

    let mut agent = DqnAgent::new(&device)?;
    let mut best_reward = f64::NEG_INFINITY;
    let mut total_steps: u64 = 0;
    let mut episode: u64 = 0;

    if let Some(resume_dir) = args.resume.as_ref() {
        let (meta, replay) = agent.resume_from(resume_dir)?;
        best_reward = meta.best_reward;
        episode = meta.episode;
        total_steps = meta.total_steps;
        agent.epsilon = meta.epsilon;
        agent.steps = meta.agent_steps;
        agent.total_env_steps = total_steps;
        agent.replay = replay;
        eprintln!(
            "Resumed from {} (steps={}, episode={}, epsilon={:.4})",
            resume_dir.display(),
            total_steps,
            episode,
            agent.epsilon
        );
    }

    std::fs::create_dir_all("checkpoints")?;

    let weight_version = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let initial_snapshot = snapshot_varmap(&agent.online_varmap, 1)?;
    let shared_weights: SharedWeights = Arc::new(RwLock::new(initial_snapshot));
    let shared_epsilon = Arc::new(AtomicU64::new(agent.epsilon.to_bits()));
    let stop = Arc::new(AtomicBool::new(false));

    let (tx, rx) = mpsc::sync_channel::<Transition>(4096);

    let mut worker_stats: Vec<Arc<RwLock<WorkerStats>>> = Vec::new();
    let mut handles = Vec::new();

    for i in 0..num_workers {
        let rom = args.rom.clone();
        let tx = tx.clone();
        let sw = shared_weights.clone();
        let se = shared_epsilon.clone();
        let st = stop.clone();
        let stats = Arc::new(RwLock::new(WorkerStats {
            episodes: 0,
            steps: 0,
            total_reward: 0.0,
            last_score: 0,
            last_kills: 0,
        }));
        worker_stats.push(stats.clone());

        let frame_skip = args.frame_skip;
        let handle = std::thread::Builder::new()
            .name(format!("worker-{i}"))
            .spawn(move || {
                worker_thread(i, rom, frame_skip, 0.25, tx, sw, se, st, stats);
            })?;
        handles.push(handle);
    }
    drop(tx);

    eprintln!("Workers spawned. Training...\n");

    let t_start = Instant::now();
    let mut recent_rewards: VecDeque<f64> = VecDeque::with_capacity(100);
    if let Some(resume_dir) = args.resume.as_ref() {
        let rewards_path = resume_dir.join("recent_rewards.json");
        if rewards_path.exists() {
            let file = File::open(&rewards_path)?;
            let reader = std::io::BufReader::new(file);
            let rewards: Vec<f64> = serde_json::from_reader(reader)?;
            recent_rewards = VecDeque::from(rewards);
        }
    }
    let mut ep_loss = 0.0f32;
    let mut loss_count = 0u32;
    let mut last_log = Instant::now();
    let mut last_sync = Instant::now();
    let mut last_save_steps = 0u64;

    let weight_sync_interval_ms = 2000u128;
    let log_interval_ms = 5000u128;

    while total_steps < args.timesteps {
        let mut drained = 0u64;
        loop {
            match rx.try_recv() {
                Ok(t) => {
                    agent.replay.push(t);
                    total_steps += 1;
                    agent.total_env_steps = total_steps;
                    drained += 1;

                    let loss = agent.train_step()?;
                    if loss > 0.0 {
                        ep_loss += loss;
                        loss_count += 1;
                    }

                    if drained >= 256 {
                        break;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    eprintln!("All workers disconnected.");
                    stop.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }

        if drained == 0 {
            match rx.recv_timeout(std::time::Duration::from_millis(10)) {
                Ok(t) => {
                    agent.replay.push(t);
                    total_steps += 1;
                    agent.total_env_steps = total_steps;

                    let loss = agent.train_step()?;
                    if loss > 0.0 {
                        ep_loss += loss;
                        loss_count += 1;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    eprintln!("All workers disconnected.");
                    break;
                }
            }
        }

        shared_epsilon.store(agent.epsilon.to_bits(), Ordering::Relaxed);

        if last_sync.elapsed().as_millis() >= weight_sync_interval_ms {
            let ver = weight_version.fetch_add(1, Ordering::Relaxed) + 1;
            let snap = snapshot_varmap(&agent.online_varmap, ver)?;
            *shared_weights.write().unwrap() = snap;
            last_sync = Instant::now();
        }

        if last_log.elapsed().as_millis() >= log_interval_ms {
            let elapsed = t_start.elapsed().as_secs_f64();
            let fps = total_steps as f64 / elapsed;
            let avg_loss = if loss_count > 0 {
                ep_loss / loss_count as f32
            } else {
                0.0
            };

            let mut total_worker_eps = 0u64;
            let mut total_worker_steps = 0u64;
            let mut best_score = 0u32;
            let mut best_kills = 0u8;
            let mut total_worker_reward = 0.0f64;
            let mut worker_reward_samples = 0u64;
            for ws in &worker_stats {
                if let Ok(s) = ws.read() {
                    total_worker_eps += s.episodes;
                    total_worker_steps += s.steps;
                    total_worker_reward += s.total_reward;
                    worker_reward_samples += 1;
                    if s.last_score > best_score {
                        best_score = s.last_score;
                    }
                    if s.last_kills > best_kills {
                        best_kills = s.last_kills;
                    }
                }
            }

            let avg_worker_reward = if worker_reward_samples > 0 {
                total_worker_reward / worker_reward_samples as f64
            } else {
                0.0
            };
            recent_rewards.push_back(avg_worker_reward);
            if recent_rewards.len() > 100 {
                recent_rewards.pop_front();
            }

            let avg_reward = if recent_rewards.is_empty() {
                0.0
            } else {
                recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64
            };

            if recent_rewards.len() >= 100 && avg_reward > best_reward {
                best_reward = avg_reward;
                agent.save("checkpoints/best.safetensors")?;
                save_checkpoint(&agent, best_reward, episode, total_steps, "checkpoints")?;
                save_recent_rewards(&recent_rewards, "checkpoints")?;
            }

            episode = total_worker_eps;

            eprintln!(
                "Steps {total_steps:>8} | Replay {:>6} | BestR {best_reward:>7.1} | Score {best_score:>6} | Kills {best_kills:>3} | ε {eps:.4} | Loss {avg_loss:.5} | FPS {fps:.0} | W_eps {total_worker_eps} | W_steps {total_worker_steps}",
                agent.replay.len(),
                eps = agent.epsilon,
            );

            ep_loss = 0.0;
            loss_count = 0;
            last_log = Instant::now();
        }

        if total_steps - last_save_steps >= 50_000 {
            let mut total_worker_eps = 0u64;
            for ws in &worker_stats {
                if let Ok(s) = ws.read() {
                    total_worker_eps += s.episodes;
                }
            }
            episode = total_worker_eps;
            agent.save(&format!("checkpoints/step_{total_steps}.safetensors"))?;
            save_checkpoint(&agent, best_reward, episode, total_steps, "checkpoints")?;
            save_recent_rewards(&recent_rewards, "checkpoints")?;
            last_save_steps = total_steps;
        }
    }

    stop.store(true, Ordering::Relaxed);
    agent.save("checkpoints/final.safetensors")?;
    let mut total_worker_eps = 0u64;
    for ws in &worker_stats {
        if let Ok(s) = ws.read() {
            total_worker_eps += s.episodes;
        }
    }
    episode = total_worker_eps;
    save_checkpoint(&agent, best_reward, episode, total_steps, "checkpoints")?;
    save_recent_rewards(&recent_rewards, "checkpoints")?;

    for h in handles {
        let _ = h.join();
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    eprintln!(
        "\nTraining complete. {total_steps} steps in {elapsed:.1}s ({:.0} FPS)",
        total_steps as f64 / elapsed
    );

    Ok(())
}
