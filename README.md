# Kung Fu RL

A reinforcement learning agent that learns to play Kung Fu Master on the NES using Deep Q-Learning.

## What This Is

This is a DQN implementation in Rust that plays Kung Fu Master (also known as Spartan X). The agent learns by reading game state directly from the NES RAM - no screen capture needed. It extracts player position, enemy positions, HP, score, and other features to build an 84-dimensional state representation.

The agent outputs 13 actions: noop, movement (left/right/crouch/jump), and punch/kick combinations in different directions.

## Quick Start

First, make sure you have a copy of the Kung Fu Master NES ROM named `kungfu.nes` in the project root.

Train the agent:
```bash
cargo run --release -- train --rom kungfu.nes --timesteps 2000000
```

Watch it play with a trained model:
```bash
cargo run --release -- play --rom kungfu.nes --model checkpoints/best.safetensors
```

Play yourself with keyboard controls:
```bash
cargo run --release -- manual --rom kungfu.nes
```

Controls: Arrow keys to move, Z to punch, X to kick, S to start.

## Features

- **Dueling DQN architecture** - Separates value and advantage streams for more stable learning
- **Double DQN** - Uses online network to select actions, target network to evaluate (reduces overestimation)
- **Experience replay** - Stores transitions and samples randomly for training
- **Soft target updates** - Smoothly updates target network (tau=0.005)
- **Parallel training** - Run multiple environments simultaneously for faster data collection - WIP
- **Full checkpointing** - Save and resume training with optimizer hyperparameters, replay buffer, and metadata
- **Custom reward shaping** - Score points, damage enemies, complete floors, beat the boss

## Command Line Options

### Training
- `--timesteps` - How many steps to train for (default: 5,000,000)
- `--frame-skip` - Frames to skip between agent decisions (default: 4)
- `--render` - Show the game window during training
- `--cpu` - Force CPU-only mode (useful on macOS if Metal causes memory issues)
- `--checkpoint-dir` - Where to save model checkpoints

### Playing
- `--model` - Path to .safetensors checkpoint file
- `--episodes` - Number of episodes to play
- `--frame-skip` - Frames to skip between agent decisions (default: 4)
- `--epsilon` - Exploration rate during play (default: 0.0)

## Checkpoints

The trainer saves several files in your checkpoint directory:
- `best.safetensors` - Best model so far (based on 100-episode average reward)
- `model.safetensors` - Current online network
- `target.safetensors` - Target network
- `optimizer.json` - Optimizer hyperparameters (needed to resume with the same LR schedule)
- `replay.bin` - Experience replay buffer (can be large)
- `meta.json` - Training progress (episode count, epsilon, etc.)

You can resume training from any checkpoint with `--resume <directory>`.

## Architecture

The neural network takes 84 input features and outputs Q-values for 13 actions:
- Player position, HP, facing direction, stance
- Enemy positions (up to 4), types, energy levels
- Knife projectiles
- Floor level, boss HP, timer

Two hidden layers of 512 units with ReLU, then split into value and advantage streams (dueling architecture).

## Performance Notes

On macOS with Metal, you might see higher memory usage during long training runs. If this is problematic, use `--cpu` to run on CPU instead.

On Windows, make sure you have CUDA installed if you want GPU training.
