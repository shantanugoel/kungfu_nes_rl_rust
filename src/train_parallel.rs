use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock, mpsc};
use std::time::Instant;

use crate::dqn::{AgentConfig, DqnAgent, DqnNet, Transition, save_checkpoint, save_recent_rewards};
use crate::env::{Action, EnvConfig, NesEnv, RewardConfig};
use crate::eval::run_eval;
use crate::{Features, STATE_DIM};

#[derive(Parser)]
pub struct TrainParallelArgs {
    #[arg(long)]
    pub rom: PathBuf,
    #[arg(long, default_value = "5000000")]
    pub timesteps: u64,
    #[arg(long, default_value = "4")]
    pub frame_skip: u32,
    #[arg(long, default_value_t = false)]
    pub cpu: bool,
    #[arg(long)]
    pub workers: Option<usize>,
    #[arg(long, default_value = "checkpoints")]
    pub checkpoint_dir: PathBuf,
    #[arg(long)]
    pub resume: Option<PathBuf>,
}

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

struct WorkerStats {
    episodes: u64,
    steps: u64,
    total_reward: f64,
    last_score: u32,
    last_kills: u8,
}

#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
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
    let env_config = EnvConfig {
        frame_skip,
        sticky_action_prob: sticky_prob,
        ..Default::default()
    };
    let mut env = NesEnv::new(rom_path, true, env_config, RewardConfig::default())?;

    let mut rng = SmallRng::from_os_rng();

    let local_varmap = VarMap::new();
    let local_vb = VarBuilder::from_varmap(&local_varmap, DType::F32, &Device::Cpu);
    let local_net = DqnNet::new(local_vb, &AgentConfig::default())?;
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
    let mut state: Features = [0.0f32; STATE_DIM];

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

pub fn train_parallel(args: &TrainParallelArgs) -> Result<()> {
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

    let mut agent = DqnAgent::new(
        &device,
        AgentConfig {
            total_timesteps: args.timesteps,
            ..AgentConfig::default()
        },
    )?;
    let mut best_reward = f64::NEG_INFINITY;
    let mut best_eval_reward = f64::NEG_INFINITY;
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

    std::fs::create_dir_all(&args.checkpoint_dir)?;

    let weight_version = Arc::new(AtomicU64::new(1));
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
    let mut last_eval_steps: u64 = (total_steps / 100_000) * 100_000;

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

        if total_steps >= last_eval_steps.saturating_add(100_000) {
            let eval_stats = run_eval(&agent, args.rom.clone(), args.frame_skip, true, 5)?;
            eprintln!(
                "Eval @ {total_steps} | eps=0 sticky=0 | avgR {:.2} | avgScore {:.0} | avgKills {:.1} | n={}",
                eval_stats.avg_reward,
                eval_stats.avg_score,
                eval_stats.avg_kills,
                eval_stats.episodes,
            );
            if eval_stats.avg_reward > best_eval_reward {
                best_eval_reward = eval_stats.avg_reward;
                let best_eval_path = args.checkpoint_dir.join("eval_best.safetensors");
                agent.save(&best_eval_path.to_string_lossy())?;
            }
            last_eval_steps = total_steps;
        }

        if total_steps - last_save_steps >= 50_000 {
            let mut total_worker_eps = 0u64;
            for ws in &worker_stats {
                if let Ok(s) = ws.read() {
                    total_worker_eps += s.episodes;
                }
            }
            episode = total_worker_eps;
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
            last_save_steps = total_steps;
        }
    }

    stop.store(true, Ordering::Relaxed);
    let final_path = args.checkpoint_dir.join("final.safetensors");
    agent.save(&final_path.to_string_lossy())?;
    let mut total_worker_eps = 0u64;
    for ws in &worker_stats {
        if let Ok(s) = ws.read() {
            total_worker_eps += s.episodes;
        }
    }
    episode = total_worker_eps;
    save_checkpoint(
        &agent,
        best_reward,
        episode,
        total_steps,
        &args.checkpoint_dir,
    )?;
    save_recent_rewards(&recent_rewards, &args.checkpoint_dir)?;

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
