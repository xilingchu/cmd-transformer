// train — reads command history from SQLite, trains a GPT-style char-level LM

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{self as nn, Module, Optimizer};
use rand::Rng;
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::PathBuf;

// ── Hyperparameters ───────────────────────────────────────────────────────────
const BLOCK_SIZE: usize = 128; // context length in characters
const N_EMBD: usize = 128;
const N_HEAD: usize = 4;
const N_LAYER: usize = 4;
const BATCH_SIZE: usize = 16;
const MAX_ITERS: usize = 3000;
const EVAL_INTERVAL: usize = 300;
const LR: f64 = 3e-4;

// ── Paths ─────────────────────────────────────────────────────────────────────
fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
        .join(".local/share/cmd-transformer")
}

fn db_path() -> PathBuf {
    data_dir().join("history.db")
}

// ── Data loading ──────────────────────────────────────────────────────────────
// Returns the full training text with format:
//   [/path/to/cwd] command\n
// Sessions are separated by a blank line so the model doesn't learn
// cross-session continuations.
fn load_commands() -> Result<(String, usize)> {
    let path = db_path();
    let conn =
        Connection::open(&path).with_context(|| format!("cannot open {:?}", path))?;
    let mut stmt = conn.prepare(
        "SELECT cmd, cwd, session FROM commands ORDER BY session, recorded_at",
    )?;

    let rows: Vec<(String, String, String)> = stmt
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get(1)?, row.get(2)?)))?
        .filter_map(|r| r.ok())
        .filter(|(cmd, _, _)| !cmd.trim().is_empty() && cmd.len() <= 256)
        .collect();

    let count = rows.len();
    let mut text = String::new();
    let mut cur_session = String::new();

    for (cmd, cwd, session) in rows {
        if session != cur_session {
            if !cur_session.is_empty() {
                text.push('\n'); // blank line between sessions
            }
            cur_session = session;
        }
        text.push_str(&format!("[{}] {}\n", cwd, cmd));
    }

    Ok((text, count))
}

// ── Tokenizer (character-level) ───────────────────────────────────────────────
struct Vocab {
    ch_to_id: HashMap<char, u32>,
    id_to_ch: Vec<char>,
}

impl Vocab {
    fn build(text: &str) -> Self {
        let chars: Vec<char> = {
            let mut set = std::collections::HashSet::new();
            text.chars().for_each(|c| {
                set.insert(c);
            });
            let mut v: Vec<char> = set.into_iter().collect();
            v.sort();
            v
        };
        let ch_to_id = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        Self { id_to_ch: chars, ch_to_id }
    }

    fn size(&self) -> usize {
        self.id_to_ch.len()
    }

    fn encode(&self, s: &str) -> Vec<u32> {
        s.chars()
            .filter_map(|c| self.ch_to_id.get(&c).copied())
            .collect()
    }

    // Save as Unicode code-points, one per line — unambiguous for any char
    fn save(&self, path: &PathBuf) -> Result<()> {
        let content: String = self
            .id_to_ch
            .iter()
            .map(|c| (*c as u32).to_string())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(path, content + "\n")?;
        Ok(())
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

struct CausalSelfAttention {
    c_attn: nn::Linear,
    c_proj: nn::Linear,
    n_head: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn new(vs: nn::VarBuilder, n_embd: usize, n_head: usize) -> Result<Self> {
        Ok(Self {
            c_attn: nn::linear(n_embd, 3 * n_embd, vs.pp("c_attn"))?,
            c_proj: nn::linear(n_embd, n_embd, vs.pp("c_proj"))?,
            n_head,
            head_dim: n_embd / n_head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let qkv = self.c_attn.forward(x)?;

        // Split into Q, K, V and reshape to (b, n_head, t, head_dim)
        let q = qkv
            .narrow(2, 0, c)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let k = qkv
            .narrow(2, c, c)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let v = qkv
            .narrow(2, c * 2, c)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let att = q
            .matmul(&k.transpose(D::Minus2, D::Minus1)?)?
            .affine(1.0 / scale, 0.0)?;

        // Causal mask: positions where j > i get -inf
        let mask_vals: Vec<f32> = (0..t * t)
            .map(|i| if (i % t) <= (i / t) { 0.0f32 } else { f32::NEG_INFINITY })
            .collect();
        let mask = Tensor::from_vec(mask_vals, (1, 1, t, t), x.device())?;
        let att = nn::ops::softmax(&att.broadcast_add(&mask)?, D::Minus1)?;

        // Weighted sum of values, merge heads
        let y = att
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c))?;
        Ok(self.c_proj.forward(&y)?)
    }
}

struct Mlp {
    c_fc: nn::Linear,
    c_proj: nn::Linear,
}

impl Mlp {
    fn new(vs: nn::VarBuilder, n_embd: usize) -> Result<Self> {
        Ok(Self {
            c_fc: nn::linear(n_embd, 4 * n_embd, vs.pp("c_fc"))?,
            c_proj: nn::linear(4 * n_embd, n_embd, vs.pp("c_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.c_proj.forward(&self.c_fc.forward(x)?.gelu()?)?)
    }
}

struct Block {
    ln1: nn::LayerNorm,
    attn: CausalSelfAttention,
    ln2: nn::LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn new(vs: nn::VarBuilder, n_embd: usize, n_head: usize) -> Result<Self> {
        Ok(Self {
            ln1: nn::layer_norm(n_embd, nn::LayerNormConfig::default(), vs.pp("ln1"))?,
            attn: CausalSelfAttention::new(vs.pp("attn"), n_embd, n_head)?,
            ln2: nn::layer_norm(n_embd, nn::LayerNormConfig::default(), vs.pp("ln2"))?,
            mlp: Mlp::new(vs.pp("mlp"), n_embd)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.add(&self.attn.forward(&self.ln1.forward(x)?)?)?;
        let x = x.add(&self.mlp.forward(&self.ln2.forward(&x)?)?)?;
        Ok(x)
    }
}

struct Gpt {
    wte: nn::Embedding,    // token embedding
    wpe: nn::Embedding,    // position embedding
    blocks: Vec<Block>,
    ln_f: nn::LayerNorm,
    lm_head: nn::Linear,
}

impl Gpt {
    fn new(
        vs: nn::VarBuilder,
        vocab_size: usize,
        n_embd: usize,
        n_head: usize,
        n_layer: usize,
        block_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            wte: nn::embedding(vocab_size, n_embd, vs.pp("wte"))?,
            wpe: nn::embedding(block_size, n_embd, vs.pp("wpe"))?,
            blocks: (0..n_layer)
                .map(|i| Block::new(vs.pp(format!("h.{i}")), n_embd, n_head))
                .collect::<Result<_>>()?,
            ln_f: nn::layer_norm(
                n_embd,
                nn::LayerNormConfig::default(),
                vs.pp("ln_f"),
            )?,
            lm_head: nn::linear(n_embd, vocab_size, vs.pp("lm_head"))?,
        })
    }

    fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let (_, t) = idx.dims2()?;
        let pos = Tensor::arange(0u32, t as u32, idx.device())?;
        let mut x = self.wte.forward(idx)?.broadcast_add(&self.wpe.forward(&pos)?)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(self.lm_head.forward(&self.ln_f.forward(&x)?)?)
    }
}

// ── Training helpers ──────────────────────────────────────────────────────────

fn get_batch(
    data: &[u32],
    block_size: usize,
    batch_size: usize,
    device: &Device,
    rng: &mut impl Rng,
) -> Result<(Tensor, Tensor)> {
    let n = data.len() - block_size - 1;
    let mut x = Vec::with_capacity(batch_size * block_size);
    let mut y = Vec::with_capacity(batch_size * block_size);
    for _ in 0..batch_size {
        let s = rng.gen_range(0..n);
        x.extend_from_slice(&data[s..s + block_size]);
        y.extend_from_slice(&data[s + 1..s + block_size + 1]);
    }
    Ok((
        Tensor::from_vec(x, (batch_size, block_size), device)?,
        Tensor::from_vec(y, (batch_size, block_size), device)?,
    ))
}

fn compute_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    Ok(nn::loss::cross_entropy(&logits.reshape((b * t, v))?, &targets.flatten_all()?)?)
}

// ── Device selection ──────────────────────────────────────────────────────────
fn detect_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        println!("Device: CUDA (GPU)");
        return Ok(Device::new_cuda(0)?);
    }

    println!("Device: CPU");
    Ok(Device::Cpu)
}

// ── Main ──────────────────────────────────────────────────────────────────────
fn main() -> Result<()> {
    println!("Loading commands from database...");
    let (text, count) = load_commands()?;
    anyhow::ensure!(count > 0, "no commands in database — run cmd-record first");
    println!("  {} commands loaded", count);

    let vocab = Vocab::build(&text);
    println!("  vocabulary: {} chars", vocab.size());

    let data: Vec<u32> = vocab.encode(&text);
    anyhow::ensure!(
        data.len() > BLOCK_SIZE * 2 + 2,
        "not enough data ({} tokens); record more commands first",
        data.len()
    );
    println!("  total tokens: {}", data.len());

    let split = (data.len() * 9) / 10;
    let train_data = &data[..split];
    let val_data = &data[split..];
    let can_eval = val_data.len() > BLOCK_SIZE + 1;
    println!(
        "  train: {}  val: {}{}",
        train_data.len(),
        val_data.len(),
        if can_eval { "" } else { " (too small — skipping val)" }
    );

    // Build model
    let device = detect_device()?;
    let varmap = nn::VarMap::new();
    let vs = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Gpt::new(vs, vocab.size(), N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE)?;

    let param_count: usize = varmap
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();
    println!("  parameters: {param_count}");

    let mut opt = nn::optim::AdamW::new(
        varmap.all_vars(),
        nn::optim::ParamsAdamW { lr: LR, ..Default::default() },
    )?;

    let mut rng = rand::thread_rng();

    println!("\nTraining ({MAX_ITERS} iters)...");
    for iter in 1..=MAX_ITERS {
        let (x, y) = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, &device, &mut rng)?;
        let train_loss = compute_loss(&model.forward(&x)?, &y)?;
        opt.backward_step(&train_loss)?;

        if iter % EVAL_INTERVAL == 0 || iter == 1 {
            let train_l = train_loss.to_scalar::<f32>()?;
            if can_eval {
                let (vx, vy) =
                    get_batch(val_data, BLOCK_SIZE, BATCH_SIZE, &device, &mut rng)?;
                let val_l = compute_loss(&model.forward(&vx)?, &vy)?
                    .to_scalar::<f32>()?;
                println!("  iter {iter:>5}/{MAX_ITERS}  train={train_l:.4}  val={val_l:.4}");
            } else {
                println!("  iter {iter:>5}/{MAX_ITERS}  train={train_l:.4}");
            }
        }
    }

    // Save everything needed for inference
    let dir = data_dir();
    std::fs::create_dir_all(&dir)?;

    let weights_path = dir.join("model.safetensors");
    varmap.save(&weights_path)?;
    println!("\nWeights → {:?}", weights_path);

    let vocab_path = dir.join("vocab.txt");
    vocab.save(&vocab_path)?;
    println!("Vocab   → {:?}", vocab_path);

    let config_path = dir.join("config.txt");
    std::fs::write(
        &config_path,
        format!(
            "vocab_size={}\nn_embd={N_EMBD}\nn_head={N_HEAD}\nn_layer={N_LAYER}\nblock_size={BLOCK_SIZE}\n",
            vocab.size()
        ),
    )?;
    println!("Config  → {:?}", config_path);

    Ok(())
}
