# RAM Map - Kung Fu (NES)

This document tracks the memory addresses used for state extraction in the `kungfu_rl_rs` project.

## Confirmed Addresses

| Address | Name | Size | Description | Potential Values / Meanings |
| :--- | :--- | :--- | :--- | :--- |
| `0x005C` | `PLAYER_LIVES` | 1 | Remaining lives. | Typically 0-5 (3 at start). |
| `0x00D4` | `PLAYER_X` | 1 | Player horizontal position. | 0-255 (screen-space). |
| `0x00D3` | `BOSS_X` | 1 | Boss horizontal position. | 0-255 (screen-space). |
| `0x00B6` | `PLAYER_Y` | 1 | Player vertical position. | 0-255 (screen-space). |
| `0x00B5` | `BOSS_Y` | 1 | Boss vertical position. | 0-255 (screen-space). |
| `0x04A6` | `PLAYER_HP` | 1 | Player health points (see Technical Notes). | 0-48 expected, `0xFF` sentinel during death. |
| `0x04A5` | `BOSS_HP` | 1 | Same as Player at least on floor 1. | 0-48, `0xFF` sentinel during death. |
| `0x036E` | `PLAYER_POSE` | 1 | Player pose/animation index. | Changes with actions and movement. Also in 0x0069|
| `0x036F` | `PLAYER_STATE` | 1 | Player direction and attack state (see Notes). | Low nibble: direction/stance; high nibble: attack type. |
| `0x0065` | `PAGE` | 1 | Screen/page number for horizontal scroll. | Increments/decrements on screen transitions. |
| `0x00CE - 0x00D1` | `ENEMY_X` | 4 | X positions for enemy slots 0 through 3. | 0-255 (screen-space). |
| `0x0087 - 0x008A` | `ENEMY_TYPE` | 4 | Enemy type identifiers for slots 0 through 3. | 0-7 observed (enemy class id). |
| `0x00B0 - 0x00B3` | `ENEMY_Y` | 4 | Y positions for enemy slots 0 through 3. | 0-255 (screen-space). |
| `0x00C0 - 0x00C3` | `ENEMY_FACING` | 4 | Facing direction for enemy slots 0 through 3. | 0/Non-zero (L/R). |
| `0x00C5` | `BOSS_FACING` | 1 | Facing direction for boss. | 0/Non-zero (L/R). |
| `0x00DF - 0x00E2` | `ENEMY_POSE` | 4 | Animation state for enemy slots 0 through 3. | Nonzero indicates active; `0x7F` seen as inactive. |
| `0x00B7 - 0x00BA` | `ENEMY_ATTACK` | 4 | Attack state for enemy slots 0 through 6. | Nonzero indicates active; `0x7F` seen as inactive. |
| `0x00BC` | `BOSS_ATTACK` | 1 | Attack state for Boss. | Cycles 0 through 9. |
| `0x04A0 - 0x04A3` | `ENEMY_ENERGY` | 4 | Enemy energy per slot 0 through 3. | See Enemy Energy Behavior. |
| `0x03D4 - 0x03D7` | `KNIFE_X` | 4 | Knife projectile X positions for slots 0 through 3. | 0-255; 0 or `0xF9` when off-screen. |
| `0x03D0 - 0x03D3` | `KNIFE_Y` | 4 | Knife projectile Y positions for slots 0 through 3. | 0-255 (screen-space). |
| `0x03EC - 0x03EF` | `KNIFE_STATE` | 4 | Knife active + facing for slots 0 through 3. | `0x00` inactive, `0x11` facing right, `0x01` facing left. |
| `0x03F0 - 0x03F3` | `KNIFE_THROW_SEQ` | 4 | Knife throw sequence counter for slots 0 through 3. | Increments on throw; cycles 0-3. |
| `0x03B1` | `KILL_COUNTER` | 1 | Total number of enemies defeated in current run. | Monotonic counter, wraps at 255. |
| `0x0531 - 0x0536` | `SCORE_DIGITS` | 6 | BCD-encoded score digits. | Each byte is a single digit 0-9 (low nibble). |
| `0x0501 - 0x0506` | `TOP_SCORE_DIGITS` | 6 | BCD-encoded score digits. | Each byte is a single digit 0-9 (low nibble). |
| `0x0390 - 0x0393` | `TIMER_DIGITS` | 4 | BCD-encoded timer digits. | Four-digit timer (e.g., 1079). |
| `0x005F` | `FLOOR` | 1 | Tracks the current stage/floor level. | Small integer that increments on floor transition (0-5 or 1-6). |
| `0x00E4` | `BOSS_ACTIVE` | 1 | Boss is active or not. Use other boss signals based on this. | 0 when not active, 1 when active, 0x7F when died |
| `0x373`  | `SHRUG_COUNTER` | 1 | Enemies (grabbers) hugging the player | Count of how many enemies are hugging the players right now, so can be between 0 to 4 |

## Unconfirmed / To Verify

| Address | Name | Status | Notes | Potential Values / Meanings |
| :--- | :--- | :--- | :--- | :--- |
| `0x0050` | `PLAYER_STATE` | To Verify | Legacy placeholder, superseded by `0x036F`. | Remove from code once unused. |

## Technical Notes

### Player HP Behavior
The `PLAYER_HP` address (`0x04A6`) occasionally reads `0xFF` during the death sequence or transition.
- Treatment: When reading this value for reinforcement learning observations, treat it as `0` (dead) or clamp to the expected maximum range to avoid reward spikes.
- Observed range: 0-48 for this ROM.

### Score Mapping
The address `0x0534` specifically maps to one of the decimal places in the HUD score. Any previous documentation suggesting this is `BOSS_HP` should be disregarded.

### Player State Encoding (`0x036F`)
Low nibble:
- 0: left standing
- 1: right standing
- 2: left facing down
- 3: right facing down
- 4: left facing jump
- 5: right facing jump

High nibble:
- 0: neutral
- 1: kicking
- 2: punching
- (possible 1/2 combo for attacking)

### Knife/Projectile Behavior
Use `KNIFE_STATE` to determine active projectiles; treat `KNIFE_X` and `KNIFE_Y` as valid only when the corresponding state is nonzero.

### Enemy Energy Behavior
- 1-hit enemies: `ENEMY_ENERGY` reads `0x00` while active; on death it becomes `0xFF` momentarily.
- 2-hit enemies: `ENEMY_ENERGY` starts at `0x01`; after the first hit it becomes `0x00`; on the killing hit it becomes `0xFF` momentarily.
