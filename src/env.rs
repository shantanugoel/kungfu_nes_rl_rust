use anyhow::{Context, Result};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tetanes_core::mem::Read;
use tetanes_core::prelude::*;

use crate::{Features, STATE_DIM};

// =============================================================================
// Reward Tuning Knobs
// =============================================================================

pub struct RewardConfig {
    pub energy_damage_multiplier: f64,
    pub max_valid_score_delta: i64,
    pub score_divisor: f64,
    pub max_score_reward: f64,
    pub reward_scale: f64,
    pub hp_damage_multiplier: f64,
    pub hp_delta_sanity_bound: i32,
    pub movement_reward_per_pixel: f64,
    pub max_movement_delta: i32,
    pub max_valid_movement_delta: i32,
    pub floor_completion_bonus: f64,
    pub boss_damage_multiplier: f64,
    pub time_penalty: f64,
    pub death_penalty: f64,
    pub max_energy_drop: u8,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            energy_damage_multiplier: 1.0,
            max_valid_score_delta: 5_000,
            score_divisor: 100.0,
            max_score_reward: 5.0,
            reward_scale: 0.1,
            hp_damage_multiplier: 0.50,
            hp_delta_sanity_bound: -200,
            movement_reward_per_pixel: 0.15,
            max_movement_delta: 128,
            max_valid_movement_delta: 15,
            floor_completion_bonus: 20.0,
            boss_damage_multiplier: 2.0,
            time_penalty: -0.002,
            death_penalty: -20.0,
            max_energy_drop: 4,
        }
    }
}

// =============================================================================
// Environment Constants
// =============================================================================

pub struct EnvConfig {
    pub frame_skip: u32,
    pub sticky_action_prob: f64,
    pub reset_state_machine_max_frames: u32,
    pub step_state_machine_max_frames: u32,
    pub auto_walk_target_x: u8,
    pub auto_walk_timeout_frames: u32,
    pub random_noop_range: std::ops::Range<u32>,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            frame_skip: 4,
            sticky_action_prob: 0.25,
            reset_state_machine_max_frames: 600,
            step_state_machine_max_frames: 180,
            auto_walk_target_x: 0x7F,
            auto_walk_timeout_frames: 500,
            random_noop_range: 1..30,
        }
    }
}

// =============================================================================
// RAM Addresses
// =============================================================================

pub mod ram {
    pub const PLAYER_LIVES: u16 = 0x005C;
    pub const PLAYER_X: u16 = 0x00D4;
    pub const PLAYER_Y: u16 = 0x00B6;
    pub const BOSS_X: u16 = 0x00D3;
    pub const BOSS_Y: u16 = 0x00B5;
    pub const PLAYER_HP: u16 = 0x04A6;
    pub const PLAYER_POSE: u16 = 0x036E;
    pub const PLAYER_STATE: u16 = 0x036F;
    pub const PAGE: u16 = 0x0065;
    pub const GAME_MODE: u16 = 0x0062;
    pub const START_TIMER: u16 = 0x003A;

    pub const ENEMY_X: [u16; 4] = [0x00CE, 0x00CF, 0x00D0, 0x00D1];
    pub const ENEMY_TYPE: [u16; 4] = [0x0087, 0x0088, 0x0089, 0x008A];
    pub const ENEMY_Y: [u16; 4] = [0x00B0, 0x00B1, 0x00B2, 0x00B3];
    pub const ENEMY_ATTACK: [u16; 4] = [0x00B7, 0x00B8, 0x00B9, 0x00BA];
    pub const ENEMY_FACING: [u16; 4] = [0x00C0, 0x00C1, 0x00C2, 0x00C3];
    pub const BOSS_FACING: u16 = 0x00C5;
    pub const ENEMY_POSE: [u16; 4] = [0x00DF, 0x00E0, 0x00E1, 0x00E2];
    pub const BOSS_ATTACK: u16 = 0x00BC;
    pub const BOSS_ACTIVE: u16 = 0x00E4;
    pub const ENEMY_ENERGY: [u16; 4] = [0x04A0, 0x04A1, 0x04A2, 0x04A3];

    pub const KNIFE_X: [u16; 4] = [0x03D4, 0x03D5, 0x03D6, 0x03D7];
    pub const KNIFE_Y: [u16; 4] = [0x03D0, 0x03D1, 0x03D2, 0x03D3];
    pub const KNIFE_STATE: [u16; 4] = [0x03EC, 0x03ED, 0x03EE, 0x03EF];
    pub const KNIFE_THROW_SEQ: [u16; 4] = [0x03F0, 0x03F1, 0x03F2, 0x03F3];

    pub const KILL_COUNTER: u16 = 0x03B1;
    pub const SHRUG_COUNTER: u16 = 0x0373;

    pub const SCORE_DIGITS: [u16; 6] = [0x0531, 0x0532, 0x0533, 0x0534, 0x0535, 0x0536];

    pub const TOP_SCORE_DIGITS: Option<[u16; 6]> =
        Some([0x0501, 0x0502, 0x0503, 0x0504, 0x0505, 0x0506]);

    pub const TIMER_DIGITS: [u16; 4] = [0x0390, 0x0391, 0x0392, 0x0393];

    pub const BOSS_HP: Option<u16> = Some(0x04A5);
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
    pub player_pose: u8,
    pub page: u8,
    pub game_mode: u8,
    pub start_timer: u8,
    pub score: u32,
    pub top_score: u32,
    pub kill_count: u8,
    pub shrug_counter: u8,
    pub enemy_x: [u8; 4],
    pub enemy_y: [u8; 4],
    pub enemy_type: [u8; 4],
    pub enemy_facing: [u8; 4],
    pub enemy_active: [bool; 4],
    pub enemy_energy: [u8; 4],
    pub enemy_attack: [u8; 4],
    pub knife_x: [u8; 4],
    pub knife_y: [u8; 4],
    pub knife_facing: [u8; 4],
    pub knife_active: [bool; 4],
    pub knife_throw_seq: [u8; 4],
    pub boss_x: u8,
    pub boss_y: u8,
    pub boss_facing: u8,
    pub boss_attack: u8,
    pub boss_active: u8,
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

impl GameState {
    pub fn to_features_with_prev(&self, prev: Option<&GameState>) -> Features {
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
        let facing_right = if self.player_state & 0x01 == 0 {
            0.0
        } else {
            1.0
        };
        let stance_code = ((self.player_state & 0x06) >> 1) as usize;
        let attack_code = ((self.player_state >> 4) & 0x0F) as usize;
        debug_assert!(
            stance_code < 3,
            "unexpected player stance code: {}",
            stance_code
        );
        debug_assert!(
            attack_code < 3,
            "unexpected player attack code: {}",
            attack_code
        );

        f[idx] = facing_right;
        idx += 1;
        let mut stance_oh = [0.0f32; 3];
        stance_oh[stance_code.min(2)] = 1.0;
        for &v in &stance_oh {
            f[idx] = v;
            idx += 1;
        }
        let mut attack_oh = [0.0f32; 3];
        attack_oh[attack_code.min(2)] = 1.0;
        for &v in &attack_oh {
            f[idx] = v;
            idx += 1;
        }
        f[idx] = self.player_pose as f32 / 255.0;
        idx += 1;
        f[idx] = self.floor as f32 / 5.0;
        idx += 1;
        f[idx] = (self.page.min(6) as f32) / 6.0;
        idx += 1;

        let mut player_dx = 0.0f32;
        let mut player_dy = 0.0f32;
        if let Some(prev_state) = prev {
            let mut dx_raw = self.player_x as f32 - prev_state.player_x as f32;
            let mut dy_raw = self.player_y as f32 - prev_state.player_y as f32;
            if dx_raw > 128.0 {
                dx_raw -= 256.0;
            } else if dx_raw < -128.0 {
                dx_raw += 256.0;
            }
            if dy_raw > 128.0 {
                dy_raw -= 256.0;
            } else if dy_raw < -128.0 {
                dy_raw += 256.0;
            }
            player_dx = dx_raw / 128.0;
            player_dy = dy_raw / 128.0;
        }
        f[idx] = player_dx;
        idx += 1;
        f[idx] = player_dy;
        idx += 1;

        #[derive(Clone, Copy)]
        struct EnemyFeatures {
            sort_key: f32,
            active: f32,
            dx: f32,
            dy: f32,
            enemy_type: [f32; 8],
            facing: f32,
            energy: f32,
            attack: f32,
        }

        let mut enemies: [EnemyFeatures; 4] = [EnemyFeatures {
            sort_key: f32::INFINITY,
            active: 0.0,
            dx: 0.0,
            dy: 0.0,
            enemy_type: [0.0; 8],
            facing: 0.0,
            energy: 0.0,
            attack: 0.0,
        }; 4];
        for (i, enemy_slot) in enemies.iter_mut().enumerate() {
            if self.enemy_active[i] {
                let mut dx_raw = self.enemy_x[i] as f32 - self.player_x as f32;
                let dy_raw = self.enemy_y[i] as f32 - self.player_y as f32;
                // Ensure it remains 0 or 1 in case raw value changes
                let facing = if self.enemy_facing[i] == 0 { 0.0 } else { 1.0 };

                if dx_raw > 128.0 {
                    dx_raw -= 256.0;
                } else if dx_raw < -128.0 {
                    dx_raw += 256.0;
                }
                let abs_dx = dx_raw.abs() / 255.0;
                let mut enemy_type = [0.0f32; 8];
                let type_idx = (self.enemy_type[i] as usize).min(7);
                enemy_type[type_idx] = 1.0;
                let energy = if self.enemy_energy[i] == 0xFF {
                    0.0
                } else {
                    (self.enemy_energy[i].min(4) as f32) / 4.0
                };
                let attack_raw = self.enemy_attack[i];
                let attack = if attack_raw == 0 || attack_raw == 0x7F {
                    0.0
                } else {
                    (attack_raw.min(0x0F) as f32) / 15.0
                };
                *enemy_slot = EnemyFeatures {
                    sort_key: abs_dx,
                    active: 1.0,
                    dx: dx_raw / 128.0,
                    dy: dy_raw / 128.0,
                    enemy_type,
                    facing,
                    energy,
                    attack,
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
            for &t in &enemy.enemy_type {
                f[idx] = t;
                idx += 1;
            }
            f[idx] = enemy.facing;
            idx += 1;
            f[idx] = enemy.energy;
            idx += 1;
            f[idx] = enemy.attack;
            idx += 1;
        }

        #[derive(Clone, Copy)]
        struct KnifeFeatures {
            sort_key: f32,
            active: f32,
            dx: f32,
            dy: f32,
            facing: f32,
            throw_seq: f32,
        }

        let mut knives: [KnifeFeatures; 4] = [KnifeFeatures {
            sort_key: f32::INFINITY,
            active: 0.0,
            dx: 0.0,
            dy: 0.0,
            facing: 0.0,
            throw_seq: 0.0,
        }; 4];

        for (i, knife_slot) in knives.iter_mut().enumerate() {
            if self.knife_active[i] {
                let mut dx_raw = self.knife_x[i] as f32 - self.player_x as f32;
                let dy_raw = self.knife_y[i] as f32 - self.player_y as f32;
                let facing = if self.knife_facing[i] == 0 { 0.0 } else { 1.0 };
                let throw_seq = (self.knife_throw_seq[i].min(3) as f32) / 3.0;

                if dx_raw > 128.0 {
                    dx_raw -= 256.0;
                } else if dx_raw < -128.0 {
                    dx_raw += 256.0;
                }
                let abs_dx = dx_raw.abs() / 255.0;
                *knife_slot = KnifeFeatures {
                    sort_key: abs_dx,
                    active: 1.0,
                    dx: dx_raw / 128.0,
                    dy: dy_raw / 128.0,
                    facing,
                    throw_seq,
                };
            }
        }

        knives.sort_unstable_by(|a, b| {
            a.sort_key
                .partial_cmp(&b.sort_key)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for knife in &knives {
            f[idx] = knife.active;
            idx += 1;
            f[idx] = knife.dx;
            idx += 1;
            f[idx] = knife.dy;
            idx += 1;
            f[idx] = knife.facing;
            idx += 1;
            f[idx] = knife.throw_seq;
            idx += 1;
        }

        let boss_active = self.boss_active != 0 && self.boss_active != 0x7F;
        let boss_active_f = if boss_active { 1.0 } else { 0.0 };
        let mut boss_dx = 0.0;
        let mut boss_dy = 0.0;
        let mut boss_facing = 0.0;
        let mut boss_attack = 0.0;
        let mut boss_hp = 0.0;
        if boss_active {
            let mut dx_raw = self.boss_x as f32 - self.player_x as f32;
            let mut dy_raw = self.boss_y as f32 - self.player_y as f32;
            if dx_raw > 128.0 {
                dx_raw -= 256.0;
            } else if dx_raw < -128.0 {
                dx_raw += 256.0;
            }
            if dy_raw > 128.0 {
                dy_raw -= 256.0;
            } else if dy_raw < -128.0 {
                dy_raw += 256.0;
            }
            boss_dx = dx_raw / 128.0;
            boss_dy = dy_raw / 128.0;
            boss_facing = if self.boss_facing == 0 { 0.0 } else { 1.0 };
            boss_attack = (self.boss_attack.min(9) as f32) / 9.0;
            let raw_hp = if self.boss_hp == 0xFF {
                0
            } else {
                self.boss_hp.min(48)
            };
            boss_hp = raw_hp as f32 / 48.0;
        }

        f[idx] = boss_active_f;
        idx += 1;
        f[idx] = boss_dx;
        idx += 1;
        f[idx] = boss_dy;
        idx += 1;
        f[idx] = boss_facing;
        idx += 1;
        f[idx] = boss_attack;
        idx += 1;
        f[idx] = boss_hp;
        idx += 1;
        f[idx] = (self.timer.min(9999) as f32) / 9999.0;
        idx += 1;
        let shrug = if self.shrug_counter == 0xFF {
            0
        } else {
            self.shrug_counter.min(4)
        };
        f[idx] = shrug as f32 / 4.0;
        idx += 1;

        debug_assert_eq!(
            idx, STATE_DIM,
            "Feature index mismatch: got {}, expected {}",
            idx, STATE_DIM
        );

        f
    }

    pub fn to_features(&self) -> Features {
        self.to_features_with_prev(None)
    }

    pub fn global_x(&self) -> i32 {
        (self.page as i32) * 256 + self.player_x as i32
    }
}

// =============================================================================
// NES Environment
// =============================================================================

const GAME_MODE_TITLE: u8 = 0x00;
const GAME_MODE_COUNTDOWN: u8 = 0x01;
const GAME_MODE_ACTION: u8 = 0x02;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionState {
    Startup,
    WaitForTitle,
    WaitForMode2,
    WaitForCountdown,
    Playing,
}

pub struct NesEnv {
    deck: ControlDeck,
    pub prev_state: GameState,
    pub total_reward: f64,
    steps: u64,
    last_action: Action,
    paused: bool,
    clock_enabled: bool,
    real_time: bool,
    next_frame_deadline: Option<Instant>,
    pub(crate) session_state: SessionState,
    countdown_seen: bool,
    debug_state: bool,
    reward_debug: bool,
    reward_breakdown: RewardBreakdown,
    progress_floor: u8,
    progress_start_global_x: i32,
    progress_best: i32,
    rng: SmallRng,
    pub min_page_reached: u8,
    pub env_config: EnvConfig,
    pub reward_config: RewardConfig,
}

impl NesEnv {
    pub fn new(
        rom_path: PathBuf,
        headless: bool,
        env_config: EnvConfig,
        reward_config: RewardConfig,
    ) -> Result<Self> {
        let mut deck = ControlDeck::new();
        let headless_mode = if headless {
            tetanes_core::control_deck::HeadlessMode::NO_AUDIO
                | tetanes_core::control_deck::HeadlessMode::NO_VIDEO
        } else {
            tetanes_core::control_deck::HeadlessMode::empty()
        };
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
            last_action: Action::Noop,
            paused: false,
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
            min_page_reached: 6,
            env_config,
            reward_config,
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

    pub fn clock_frame(&mut self) -> Result<()> {
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

    pub fn peek(&self, addr: u16) -> u8 {
        self.deck.bus().peek(addr)
    }

    pub fn read_score(&self) -> u32 {
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

    pub fn read_timer(&self) -> u16 {
        let mut timer = 0u16;
        let multipliers = [1000, 100, 10, 1];
        for (i, &addr) in ram::TIMER_DIGITS.iter().enumerate() {
            let digit = self.peek(addr) & 0x0F;
            timer += digit as u16 * multipliers[i];
        }
        timer
    }

    fn read_state(&self) -> GameState {
        let mut state = GameState {
            player_x: self.peek(ram::PLAYER_X),
            player_y: self.peek(ram::PLAYER_Y),
            player_hp: self.peek(ram::PLAYER_HP),
            player_lives: self.peek(ram::PLAYER_LIVES),
            player_state: self.peek(ram::PLAYER_STATE),
            player_pose: self.peek(ram::PLAYER_POSE),
            page: self.peek(ram::PAGE),
            game_mode: self.peek(ram::GAME_MODE),
            start_timer: self.peek(ram::START_TIMER),
            score: self.read_score(),
            top_score: ram::TOP_SCORE_DIGITS
                .as_ref()
                .map(|digits| self.read_score_digits(digits))
                .unwrap_or(0),
            kill_count: self.peek(ram::KILL_COUNTER),
            shrug_counter: self.peek(ram::SHRUG_COUNTER),
            boss_x: self.peek(ram::BOSS_X),
            boss_y: self.peek(ram::BOSS_Y),
            boss_facing: self.peek(ram::BOSS_FACING),
            boss_attack: self.peek(ram::BOSS_ATTACK),
            boss_active: self.peek(ram::BOSS_ACTIVE),
            boss_hp: ram::BOSS_HP.map(|addr| self.peek(addr)).unwrap_or(0),
            floor: self.peek(ram::FLOOR),
            timer: self.read_timer(),
            ..Default::default()
        };

        for i in 0..4 {
            state.enemy_x[i] = self.peek(ram::ENEMY_X[i]);
            state.enemy_y[i] = self.peek(ram::ENEMY_Y[i]);
            state.enemy_type[i] = self.peek(ram::ENEMY_TYPE[i]);
            state.enemy_facing[i] = self.peek(ram::ENEMY_FACING[i]);
            let pose = self.peek(ram::ENEMY_POSE[i]);
            state.enemy_active[i] = pose != 0 && pose != 0x7F;
            state.enemy_energy[i] = self.peek(ram::ENEMY_ENERGY[i]);
            state.enemy_attack[i] = self.peek(ram::ENEMY_ATTACK[i]);

            state.knife_x[i] = self.peek(ram::KNIFE_X[i]);
            state.knife_y[i] = self.peek(ram::KNIFE_Y[i]);
            let knife_state = self.peek(ram::KNIFE_STATE[i]);
            state.knife_active[i] = knife_state != 0;
            state.knife_facing[i] = match knife_state {
                0x11 => 1,
                0x01 => 0,
                _ => 0,
            };
            state.knife_throw_seq[i] = self.peek(ram::KNIFE_THROW_SEQ[i]);
        }
        state
    }

    fn set_input(&mut self, action: Action) {
        let btn_state = action.to_joypad();
        self.set_input_state(btn_state);
    }

    pub fn set_input_state(&mut self, btn_state: tetanes_core::input::JoypadBtnState) {
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

    pub fn reset(&mut self) -> Result<Features> {
        self.deck.reset(ResetKind::Soft);
        self.session_state = SessionState::Startup;
        self.countdown_seen = false;

        let noops = self
            .rng
            .random_range(self.env_config.random_noop_range.clone());
        for _ in 0..noops {
            self.clock_frame()?;
        }

        let mut state = self.run_state_machine(self.env_config.reset_state_machine_max_frames)?;
        self.log_state("reset", &state);
        if self.session_state != SessionState::Playing {
            return Err(anyhow::anyhow!(
                "Timed out waiting for play state (phase: {phase:?}, mode: 0x{mode:02X}, start_timer: {timer})",
                phase = self.session_state,
                mode = state.game_mode,
                timer = state.start_timer
            ));
        }
        let mut timeout = 0;
        while state.player_x > self.env_config.auto_walk_target_x
            && timeout < self.env_config.auto_walk_timeout_frames
        {
            self.clock_frame()?;
            state = self.read_state();
            timeout += 1;
        }

        self.prev_state = state;
        self.min_page_reached = self.prev_state.page;
        self.total_reward = 0.0;
        self.steps = 0;
        self.last_action = Action::Noop;
        self.paused = false;
        self.progress_floor = self.prev_state.floor;
        self.progress_start_global_x = self.prev_state.global_x();
        self.progress_best = 0;

        Ok(self.prev_state.to_features_with_prev(None))
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
            self.clock_frame()?;
        }
        self.set_input_state(JoypadBtnState::empty());
        Ok(())
    }

    pub fn step(&mut self, action: Action) -> Result<StepResult> {
        self.steps += 1;
        let prev_obs_state = self.prev_state.clone();

        let effective_action = if self.rng.random::<f64>() < self.env_config.sticky_action_prob {
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
            let state = self.run_state_machine(self.env_config.step_state_machine_max_frames)?;
            self.log_state("step-nonplay", &state);
            playing = false;
            let done = self.session_state != SessionState::Playing;
            self.prev_state = state;
            let active_enemies = (0..4).filter(|&i| self.prev_state.enemy_active[i]).count() as u8;
            return Ok(StepResult {
                state: self.prev_state.to_features_with_prev(None),
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

        for _ in 0..self.env_config.frame_skip {
            self.set_input(effective_action);
            self.clock_frame()?;

            let state = self.read_state();

            let reward = self.compute_reward(&state);
            frame_reward += reward;

            if state.player_lives < self.prev_state.player_lives && self.prev_state.player_lives > 0
            {
                frame_reward += self.reward_config.death_penalty;
                if self.reward_debug {
                    self.reward_breakdown.death +=
                        self.reward_config.death_penalty * self.reward_config.reward_scale;
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
            frame_reward *= self.reward_config.reward_scale;
            self.total_reward += frame_reward;
        }

        let active_enemies = (0..4).filter(|&i| self.prev_state.enemy_active[i]).count() as u8;

        Ok(StepResult {
            state: self.prev_state.to_features_with_prev(Some(&prev_obs_state)),
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

    fn compute_reward(&mut self, cur: &GameState) -> f64 {
        let rc = &self.reward_config;
        let prev = &self.prev_state;
        let mut reward = 0.0;
        let mut energy_reward = 0.0;
        let mut score_reward = 0.0;
        let mut hp_penalty = 0.0;
        let mut movement_reward = 0.0;
        let mut floor_bonus = 0.0;
        let mut boss_reward = 0.0;

        for i in 0..4 {
            if prev.enemy_active[i]
                && cur.enemy_active[i]
                && prev.enemy_type[i] == cur.enemy_type[i]
            {
                let prev_energy = prev.enemy_energy[i];
                let cur_energy = cur.enemy_energy[i];
                if prev_energy > 0 && cur_energy < prev_energy && cur_energy != 0xFF {
                    let drop = prev_energy - cur_energy;
                    if drop <= rc.max_energy_drop {
                        energy_reward += drop as f64 * rc.energy_damage_multiplier;
                    }
                }
            }
        }

        let score_delta = cur.score as i64 - prev.score as i64;
        if score_delta > 0 && score_delta < rc.max_valid_score_delta {
            score_reward += (score_delta as f64 / rc.score_divisor).min(rc.max_score_reward);
        }

        if cur.player_hp != 0xFF && prev.player_hp != 0xFF {
            let hp_delta = cur.player_hp as i32 - prev.player_hp as i32;
            if hp_delta < 0 && hp_delta > rc.hp_delta_sanity_bound {
                hp_penalty += hp_delta as f64 * rc.hp_damage_multiplier;
            }
        }

        let cur_global_x = cur.global_x();
        if cur.floor != self.progress_floor {
            self.progress_floor = cur.floor;
            self.progress_start_global_x = cur_global_x;
            self.progress_best = 0;
        } else {
            let progress = if cur.floor.is_multiple_of(2) {
                self.progress_start_global_x - cur_global_x
            } else {
                cur_global_x - self.progress_start_global_x
            };
            if progress > self.progress_best {
                let delta = (progress - self.progress_best).min(rc.max_movement_delta);
                if delta > 0 && delta < rc.max_valid_movement_delta {
                    movement_reward += delta as f64 * rc.movement_reward_per_pixel;
                    self.progress_best = progress;
                }
            }
        }

        if cur.floor > prev.floor {
            floor_bonus += rc.floor_completion_bonus;
        }

        let boss_active = cur.boss_active != 0 && cur.boss_active != 0x7F;
        let boss_delta = prev.boss_hp as i32 - cur.boss_hp as i32;
        if boss_active && boss_delta > 0 && boss_delta < 200 {
            boss_reward += boss_delta as f64 * rc.boss_damage_multiplier;
        }

        reward += rc.time_penalty;

        if cur.page < self.min_page_reached && cur.page > 0 {
            self.min_page_reached = cur.page;
        }

        reward +=
            energy_reward + score_reward + hp_penalty + movement_reward + floor_bonus + boss_reward;

        if self.reward_debug {
            let scale = rc.reward_scale;
            self.reward_breakdown.energy += energy_reward * scale;
            self.reward_breakdown.score += score_reward * scale;
            self.reward_breakdown.hp += hp_penalty * scale;
            self.reward_breakdown.movement += movement_reward * scale;
            self.reward_breakdown.floor += floor_bonus * scale;
            self.reward_breakdown.boss += boss_reward * scale;
            self.reward_breakdown.time += rc.time_penalty * scale;
        }

        reward
    }

    pub fn frame_buffer(&mut self) -> &[u8] {
        self.deck.frame_buffer()
    }
}

pub struct StepResult {
    pub state: Features,
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
