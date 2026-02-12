use anyhow::Result;
use clap::{Parser, Subcommand};
use kungfu_rl_rs::Features;
use kungfu_rl_rs::dqn::{AgentConfig, DqnAgent, Transition, save_checkpoint, save_recent_rewards};
use kungfu_rl_rs::env::{Action, EnvConfig, NesEnv, RewardConfig, ram};
use kungfu_rl_rs::train_parallel::{self, TrainParallelArgs};
use rand::Rng;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

fn blit_rgba_to_u32(fb: &[u8], out: &mut [u32]) {
    for (dst, src) in out.iter_mut().zip(fb.chunks_exact(4)) {
        *dst = ((src[0] as u32) << 16) | ((src[1] as u32) << 8) | (src[2] as u32);
    }
}

fn update_pause_from_window(env: &mut NesEnv, window: &minifb::Window) {
    if window.is_key_pressed(minifb::Key::Space, minifb::KeyRepeat::No) {
        env.toggle_pause();
    }
}

// =============================================================================
// Training
// =============================================================================

fn train(args: &TrainArgs) -> Result<()> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  TRAINING — Kung Fu Master DQN Agent (Rust + candle)");
    eprintln!("═══════════════════════════════════════════════════════════");

    let device = if args.cpu {
        candle_core::Device::Cpu
    } else {
        candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
    };
    eprintln!("Device: {:?}", device);

    let env_config = EnvConfig {
        frame_skip: args.frame_skip,
        ..Default::default()
    };
    let mut env = NesEnv::new(
        args.rom.clone(),
        !args.render,
        env_config,
        RewardConfig::default(),
    )?;
    env.set_clock_enabled(!args.no_clock);
    env.set_real_time(args.real_time);

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
    let mut agent = DqnAgent::new(&device, AgentConfig::default())?;

    std::fs::create_dir_all(&args.checkpoint_dir)?;

    let mut best_reward = f64::NEG_INFINITY;
    let mut episode: u64 = 0;
    let mut total_steps: u64 = 0;
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
            "📦 Resumed from {} (steps={}, episode={}, epsilon={:.4})",
            resume_dir.display(),
            total_steps,
            episode,
            agent.epsilon
        );
    }
    let t_start = Instant::now();

    let mut all_time_top_score: u32 = 0;

    let mut last_render_ep = 0u64;
    let mut last_render_steps = 0u64;
    let mut last_render_reward = 0.0f64;
    let mut last_render_avg = 0.0f64;
    let mut last_render_score = 0u32;
    let mut last_render_top = 0u32;
    let mut last_render_kills = 0u32;

    let mut recent_rewards: VecDeque<f64> = VecDeque::with_capacity(100);
    if let Some(resume_dir) = args.resume.as_ref() {
        let rewards_path = resume_dir.join("recent_rewards.json");
        if rewards_path.exists() {
            let file = std::fs::File::open(&rewards_path)?;
            let reader = std::io::BufReader::new(file);
            let rewards: Vec<f64> = serde_json::from_reader(reader)?;
            recent_rewards = VecDeque::from(rewards);
        }
    }

    let mut buf = vec![0u32; 256 * 240];
    let mut last_title_update = Instant::now();

    let mut need_reset = true;
    let mut state: Features = [0.0f32; kungfu_rl_rs::STATE_DIM];
    let mut ep_kills: u32;
    let mut prev_kill_count: u8 = 0;
    let mut ep_score: u32;
    let mut prev_score: u32 = 0;
    let mut game_total_score: u32 = 0;

    while total_steps < args.timesteps {
        episode += 1;

        if need_reset {
            state = env.reset()?;
            prev_kill_count = env.prev_state.kill_count;
            prev_score = env.prev_state.score;
            game_total_score = 0;
            need_reset = false;
        }

        ep_kills = 0;
        ep_score = 0;
        let mut ep_reward = 0.0;
        let mut ep_steps = 0u64;
        let mut ep_loss = 0.0f32;
        let mut loss_count = 0u32;
        if env.reward_debug_enabled() {
            env.clear_reward_breakdown();
        }

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
            if result.playing {
                agent.remember(Transition {
                    state,
                    action: action_idx,
                    reward: result.reward,
                    next_state: result.state,
                    done: result.done,
                });

                total_steps += 1;
                agent.total_env_steps = total_steps;
                let loss = agent.train_step()?;
                if loss > 0.0 {
                    ep_loss += loss;
                    loss_count += 1;
                }

                ep_reward += result.reward as f64;
                ep_steps += 1;

                let kill_delta = result.kills.wrapping_sub(prev_kill_count);
                if kill_delta > 0 && kill_delta < 10 {
                    ep_kills += kill_delta as u32;
                }
                prev_kill_count = result.kills;

                let score_delta = result.score.saturating_sub(prev_score);
                ep_score += score_delta;
                game_total_score += score_delta;
                prev_score = result.score;
            }

            state = result.state;

            if let Some(win) = window.as_mut()
                && total_steps.is_multiple_of(4)
            {
                if last_title_update.elapsed().as_millis() > 250 {
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
                        ep_score
                    } else {
                        last_render_score
                    };
                    let overlay_top = if last_render_ep == 0 {
                        all_time_top_score
                    } else {
                        last_render_top
                    };
                    let overlay_kills = if last_render_ep == 0 {
                        ep_kills
                    } else {
                        last_render_kills
                    };
                    win.set_title(&format!(
                            "Kung Fu Master — Training | Ep {overlay_ep} | Steps {overlay_steps} | R {overlay_reward:.1} | Avg100 {overlay_avg:.1} | Score {overlay_score} | Top {overlay_top} | Kills {overlay_kills}"
                        ));
                    last_title_update = Instant::now();
                }
                let fb = env.frame_buffer();
                blit_rgba_to_u32(fb, &mut buf);
                win.update_with_buffer(&buf, 256, 240)?;
            }

            if result.done || ep_steps > 10_000 {
                if result.game_over {
                    need_reset = true;
                }
                break;
            }
        }

        if total_steps % 50_000 < ep_steps {
            let step_path = args
                .checkpoint_dir
                .join(format!("step_{total_steps}.safetensors"));
            agent.save(&step_path.to_string_lossy())?;
            save_checkpoint(
                &agent,
                best_reward,
                episode,
                total_steps,
                &args.checkpoint_dir,
            )?;
            save_recent_rewards(&recent_rewards, &args.checkpoint_dir)?;
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

        if recent_rewards.len() >= 100 && avg_reward > best_reward {
            best_reward = avg_reward;
            let best_path = args.checkpoint_dir.join("best.safetensors");
            agent.save(&best_path.to_string_lossy())?;
            save_checkpoint(
                &agent,
                best_reward,
                episode,
                total_steps,
                &args.checkpoint_dir,
            )?;
            save_recent_rewards(&recent_rewards, &args.checkpoint_dir)?;
        }

        let elapsed = t_start.elapsed().as_secs_f64();
        let fps = total_steps as f64 / elapsed;

        if game_total_score > all_time_top_score {
            all_time_top_score = game_total_score;
        }

        if episode.is_multiple_of(10) || ep_reward > best_reward - 1.0 {
            eprintln!(
                "Ep {episode:>5} | Steps {total_steps:>8} | R {ep_reward:>8.1} | \
                 Avg100 {avg_reward:>7.1} | Score {ep_score:>6} | Top {all_time_top_score:>6} | Kills {ep_kills:>3} | \
                 ε {eps:.4} | Loss {loss:.5} | FPS {fps:.0} | MinPg {min_pg}",
                eps = agent.epsilon,
                loss = avg_loss,
                min_pg = env.min_page_reached,
            );
            if env.reward_debug_enabled() {
                let breakdown = env.reward_breakdown();
                eprintln!(
                    "  R parts | score {:+.1} | hp {:+.1} | death {:+.1} | energy {:+.1} | move {:+.1} | floor {:+.1} | boss {:+.1} | time {:+.1}",
                    breakdown.score,
                    breakdown.hp,
                    breakdown.death,
                    breakdown.energy,
                    breakdown.movement,
                    breakdown.floor,
                    breakdown.boss,
                    breakdown.time,
                );
            }
        }

        last_render_ep = episode;
        last_render_steps = total_steps;
        last_render_reward = ep_reward;
        last_render_avg = avg_reward;
        last_render_score = ep_score;
        last_render_top = all_time_top_score;
        last_render_kills = ep_kills;
    }

    let final_path = args.checkpoint_dir.join("final.safetensors");
    agent.save(&final_path.to_string_lossy())?;
    save_checkpoint(
        &agent,
        best_reward,
        episode,
        total_steps,
        &args.checkpoint_dir,
    )?;
    save_recent_rewards(&recent_rewards, &args.checkpoint_dir)?;
    eprintln!(
        "\n✅ Training complete. {total_steps} steps in {:.1}s",
        t_start.elapsed().as_secs_f64()
    );
    Ok(())
}

// =============================================================================
// Play / Evaluate
// =============================================================================

fn play(args: &PlayArgs) -> Result<()> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  PLAYING — Kung Fu Master DQN Agent");
    eprintln!("═══════════════════════════════════════════════════════════");

    let device = if args.cpu {
        candle_core::Device::Cpu
    } else {
        candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
    };

    let env_config = EnvConfig {
        sticky_action_prob: 0.0,
        ..Default::default()
    };
    let mut env = NesEnv::new(args.rom.clone(), false, env_config, RewardConfig::default())?;
    env.set_clock_enabled(!args.no_clock);
    env.set_real_time(args.real_time);
    let mut agent = DqnAgent::new(&device, AgentConfig::default())?;
    agent.load(&args.model.to_string_lossy())?;
    agent.epsilon = 0.0;

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

    let mut buf = vec![0u32; 256 * 240];

    for ep in 0..args.episodes {
        let mut state = env.reset()?;
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
            if result.playing {
                total_reward += result.reward as f64;
                steps += 1;
            }
            state = result.state;

            let fb = env.frame_buffer();
            blit_rgba_to_u32(fb, &mut buf);
            window.update_with_buffer(&buf, 256, 240)?;

            if result.game_over || steps > 20_000 || !window.is_open() {
                break;
            }
        }

        eprintln!(
            "Episode {}: reward={total_reward:.1}, steps={steps}, score={}, top={}, kills={}",
            ep + 1,
            env.prev_state.score,
            env.prev_state.top_score,
            env.prev_state.kill_count,
        );

        if !window.is_open() {
            break;
        }
    }

    Ok(())
}

// =============================================================================
// RAM Explorer
// =============================================================================

fn explore(args: &ExploreArgs) -> Result<()> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  RAM EXPLORER — Kung Fu (NES via tetanes-core)");
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("Watch RAM values change as the game runs.");
    eprintln!("Press Start via keyboard if tetanes supports it,");
    eprintln!("or modify this to inject Start presses automatically.");
    eprintln!("═══════════════════════════════════════════════════════════");

    let env_config = EnvConfig {
        frame_skip: 1,
        sticky_action_prob: 0.0,
        ..Default::default()
    };
    let mut env = NesEnv::new(args.rom.clone(), false, env_config, RewardConfig::default())?;
    env.set_clock_enabled(!args.no_clock);
    env.set_real_time(args.real_time);

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
        ("Game Mode", ram::GAME_MODE),
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
    let mut buf = vec![0u32; 256 * 240];

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
        env.clock_frame()?;
        frame += 1;

        let fb = env.frame_buffer();
        blit_rgba_to_u32(fb, &mut buf);
        window.update_with_buffer(&buf, 256, 240)?;

        if frame.is_multiple_of(30) {
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
// Random Baseline
// =============================================================================

fn baseline(args: &BaselineArgs) -> Result<()> {
    eprintln!("Running random agent baseline...");

    let env_config = EnvConfig {
        sticky_action_prob: 0.0,
        ..Default::default()
    };
    let mut env = NesEnv::new(args.rom.clone(), true, env_config, RewardConfig::default())?;
    env.set_clock_enabled(!args.no_clock);
    env.set_real_time(args.real_time);
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

            if result.game_over || steps > 10_000 {
                break;
            }
        }

        eprintln!(
            "Random ep {}: reward={total_reward:.1}, steps={steps}, score={}, top={}, kills={}",
            ep + 1,
            env.prev_state.score,
            env.prev_state.top_score,
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
// Manual Play (Keyboard)
// =============================================================================

fn manual(args: &ManualArgs) -> Result<()> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  MANUAL — Kung Fu Master (Keyboard)");
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("Arrows: Move | Z: B (Punch) | X: A (Kick) | A: Select | S: Start");
    eprintln!("Space: Pause | Esc: Quit");

    let env_config = EnvConfig {
        frame_skip: 1,
        sticky_action_prob: 0.0,
        ..Default::default()
    };
    let mut env = NesEnv::new(args.rom.clone(), false, env_config, RewardConfig::default())?;
    env.set_clock_enabled(!args.no_clock);
    env.set_real_time(args.real_time);
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

    let mut buf = vec![0u32; 256 * 240];

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
            env.clock_frame()?;
        } else {
            env.step_pause()?;
        }

        let fb = env.frame_buffer();
        blit_rgba_to_u32(fb, &mut buf);
        window.update_with_buffer(&buf, 256, 240)?;
    }

    Ok(())
}

// =============================================================================
// CLI
// =============================================================================

#[derive(Parser)]
#[command(name = "kungfu-rl", about = "Kung Fu Master NES — DQN RL Agent")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Explore(ExploreArgs),
    Train(TrainArgs),
    TrainParallel(TrainParallelArgs),
    Play(PlayArgs),
    Manual(ManualArgs),
    Baseline(BaselineArgs),
}

#[derive(Parser)]
struct ExploreArgs {
    #[arg(long)]
    rom: PathBuf,
    #[arg(long, default_value_t = false)]
    no_clock: bool,
    #[arg(long, default_value_t = false)]
    real_time: bool,
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
    #[arg(long, default_value_t = false)]
    cpu: bool,
    #[arg(long, default_value_t = false)]
    no_clock: bool,
    #[arg(long, default_value_t = false)]
    real_time: bool,
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: PathBuf,
    #[arg(long)]
    resume: Option<PathBuf>,
}

#[derive(Parser)]
struct PlayArgs {
    #[arg(long)]
    rom: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "5")]
    episodes: usize,
    #[arg(long, default_value_t = false)]
    cpu: bool,
    #[arg(long, default_value_t = false)]
    no_clock: bool,
    #[arg(long, default_value_t = false)]
    real_time: bool,
}

#[derive(Parser)]
struct ManualArgs {
    #[arg(long)]
    rom: PathBuf,
    #[arg(long, default_value_t = false)]
    no_clock: bool,
    #[arg(long, default_value_t = false)]
    real_time: bool,
}

#[derive(Parser)]
struct BaselineArgs {
    #[arg(long)]
    rom: PathBuf,
    #[arg(long, default_value_t = false)]
    no_clock: bool,
    #[arg(long, default_value_t = false)]
    real_time: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".to_string()))
        .init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Explore(args) => explore(args),
        Commands::Train(args) => train(args),
        Commands::TrainParallel(args) => train_parallel::train_parallel(args),
        Commands::Play(args) => play(args),
        Commands::Manual(args) => manual(args),
        Commands::Baseline(args) => baseline(args),
    }
}
