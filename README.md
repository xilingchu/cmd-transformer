# cmd-transformer

Records your shell command history into SQLite, trains a GPT-style character-level language model on it, and provides command completion via inference.

## Architecture

```
hook.zsh → cmd-record → history.db → train → model.safetensors → infer
```

| Component | Binary | Description |
|-----------|--------|-------------|
| zsh hook | — | Captures every command with cwd, exit code, duration, session |
| Recorder | `cmd-record` | Called by the hook; writes one row to SQLite per command |
| Trainer | `train` | Reads history, trains a transformer, saves weights |
| Inference | `infer` | Loads weights, completes a given command prefix |

## Model

GPT-style transformer, character-level tokenization.

- 4 transformer blocks, 4 attention heads, 128-dim embeddings
- Context window: 128 characters
- ~820K parameters
- Training data format: `[/path/to/cwd] command` — so the model learns the relationship between directories and commands

## Requirements

- Rust (stable, 1.88+)
- SQLite (statically linked via `rusqlite/bundled` — no system dependency)
- Optional: NVIDIA GPU with CUDA for faster training/inference

## Setup

### 1. Build

```bash
# CPU (default)
cargo build --release

# With CUDA support (NVIDIA GPU)
cargo build --release --features cuda
```

### 2. Install the zsh hook

```bash
mkdir -p ~/.config/cmd-transformer
cp hook.zsh ~/.config/cmd-transformer/hook.zsh
```

Add to `~/.zshrc`:

```zsh
source ~/.config/cmd-transformer/hook.zsh
```

The hook records every command asynchronously (`&!`) so it never blocks your prompt.

### 3. Set the binary path

The hook looks for `cmd-record` via the `CMD_RECORD_BIN` environment variable. The installed hook already points to:

```
/path/to/bash-transformer/target/release/cmd-record
```

If you move the project, update that path in `~/.config/cmd-transformer/hook.zsh`.

## Usage

### Record (automatic)

Once the hook is sourced, every command you run is recorded automatically. Data is stored at:

```
~/.local/share/cmd-transformer/history.db
```

### Train

```bash
./target/release/train
```

Trains on the full command history. Saves three files to `~/.local/share/cmd-transformer/`:

| File | Content |
|------|---------|
| `model.safetensors` | Model weights |
| `vocab.txt` | Character vocabulary (Unicode code points) |
| `config.txt` | Hyperparameters (layers, dims, block size) |

Training prints loss every 300 iterations. Recommended to retrain periodically as history grows. The model becomes meaningfully useful at a few thousand recorded commands.

### Infer

```bash
./target/release/infer <prefix> [--temperature 0.8] [--max-tokens 200]
```

Completes a command prefix character by character, stopping at newline.

```bash
# Basic completion
./target/release/infer "cargo"

# More deterministic output
./target/release/infer "git" --temperature 0.3

# Pass session context for better accuracy (cwd-aware)
./target/release/infer "[/home/user/projects/myapp] git "
```

**Temperature:** lower = more conservative/repetitive, higher = more creative/random. Default: 0.8.

## Database schema

```sql
CREATE TABLE commands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cmd         TEXT    NOT NULL,
    cwd         TEXT    NOT NULL,
    exit_code   INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    session     TEXT    NOT NULL,   -- PID-timestamp, unique per terminal window
    recorded_at INTEGER NOT NULL    -- Unix timestamp
);
```

Useful queries:

```bash
# Recent commands
sqlite3 ~/.local/share/cmd-transformer/history.db \
  "SELECT datetime(recorded_at,'unixepoch','localtime'), cwd, cmd FROM commands ORDER BY recorded_at DESC LIMIT 20"

# Commands by directory
sqlite3 ~/.local/share/cmd-transformer/history.db \
  "SELECT cmd, count(*) c FROM commands WHERE cwd LIKE '%projects%' GROUP BY cmd ORDER BY c DESC LIMIT 10"
```

## ML stack

- [Candle](https://github.com/huggingface/candle) — Hugging Face's pure-Rust tensor library (`candle-core` + `candle-nn`)
- No Python, no PyTorch
- Weights saved as [safetensors](https://github.com/huggingface/safetensors)
