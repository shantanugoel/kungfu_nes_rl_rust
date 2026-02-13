use anyhow::Result;

use crate::dqn::DqnAgent;
use crate::env::{Action, EnvConfig, NesEnv, RewardConfig};

pub struct EvalStats {
    pub avg_reward: f64,
    pub avg_score: f64,
    pub avg_kills: f64,
    pub episodes: usize,
}

pub fn run_eval(
    agent: &DqnAgent,
    rom: std::path::PathBuf,
    frame_skip: u32,
    clock_enabled: bool,
    episodes: usize,
) -> Result<EvalStats> {
    let env_config = EnvConfig {
        frame_skip,
        sticky_action_prob: 0.0,
        ..Default::default()
    };
    let mut env = NesEnv::new(rom, true, env_config, RewardConfig::default())?;
    env.set_clock_enabled(clock_enabled);
    env.set_real_time(false);

    let mut total_reward = 0.0f64;
    let mut total_score = 0u64;
    let mut total_kills = 0u64;

    let eval_episodes = episodes.max(1);

    for _ in 0..eval_episodes {
        let mut state = env.reset()?;
        let mut ep_reward = 0.0f64;
        let mut ep_steps = 0u64;
        let mut ep_score = 0u32;
        let mut prev_score = env.prev_state.score;
        let mut ep_kills = 0u32;
        let mut prev_kill_count = env.prev_state.kill_count;

        loop {
            let q_vals = agent.q_values(&state)?;
            let (action_idx, _) = q_vals
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            let result = env.step(Action::from_index(action_idx))?;
            if result.playing {
                ep_reward += result.reward as f64;
                ep_steps += 1;

                let kill_delta = result.kills.wrapping_sub(prev_kill_count);
                if kill_delta > 0 && kill_delta < 10 {
                    ep_kills += kill_delta as u32;
                }
                prev_kill_count = result.kills;

                let score_delta = result.score.saturating_sub(prev_score);
                ep_score += score_delta;
                prev_score = result.score;
            }
            state = result.state;

            if result.done || ep_steps > 10_000 {
                break;
            }
        }

        total_reward += ep_reward;
        total_score += ep_score as u64;
        total_kills += ep_kills as u64;
    }

    let denom = eval_episodes as f64;
    Ok(EvalStats {
        avg_reward: total_reward / denom,
        avg_score: total_score as f64 / denom,
        avg_kills: total_kills as f64 / denom,
        episodes: eval_episodes,
    })
}
