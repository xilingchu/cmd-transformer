// infer — load trained model and generate command completions

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{self as nn, Module};
use rand::Rng;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

// ── Paths ─────────────────────────────────────────────────────────────────────
fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
        .join(".local/share/cmd-transformer")
}

// ── Config ────────────────────────────────────────────────────────────────────
struct Config {
    vocab_size: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    block_size: usize,
}

impl Config {
    fn load(path: &PathBuf) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read {:?} — run train first", path))?;
        let mut map = HashMap::new();
        for line in text.lines() {
            if let Some((k, v)) = line.split_once('=') {
                map.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
        let get = |k: &str| -> Result<usize> {
            map.get(k)
                .with_context(|| format!("missing key '{}' in config", k))?
                .parse()
                .with_context(|| format!("invalid value for '{}'", k))
        };
        Ok(Self {
            vocab_size: get("vocab_size")?,
            n_embd: get("n_embd")?,
            n_head: get("n_head")?,
            n_layer: get("n_layer")?,
            block_size: get("block_size")?,
        })
    }
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────
struct Vocab {
    ch_to_id: HashMap<char, u32>,
    id_to_ch: Vec<char>,
}

impl Vocab {
    fn load(path: &PathBuf) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read {:?} — run train first", path))?;
        let id_to_ch: Vec<char> = text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| {
                l.parse::<u32>()
                    .ok()
                    .and_then(char::from_u32)
                    .with_context(|| format!("invalid code point: {}", l))
            })
            .collect::<Result<_>>()?;
        let ch_to_id = id_to_ch
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        Ok(Self { ch_to_id, id_to_ch })
    }

    fn encode(&self, s: &str) -> Vec<u32> {
        s.chars()
            .filter_map(|c| self.ch_to_id.get(&c).copied())
            .collect()
    }

    fn decode_char(&self, id: u32) -> Option<char> {
        self.id_to_ch.get(id as usize).copied()
    }
}

// ── Model (must match train exactly) ─────────────────────────────────────────

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

        let q = qkv
            .narrow(2, 0, c)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qkv
            .narrow(2, c, c)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qkv
            .narrow(2, c * 2, c)?
            .reshape((b, t, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let att = q
            .matmul(&k.transpose(D::Minus2, D::Minus1)?)?
            .affine(1.0 / scale, 0.0)?;

        let mask_vals: Vec<f32> = (0..t * t)
            .map(|i| if (i % t) <= (i / t) { 0.0f32 } else { f32::NEG_INFINITY })
            .collect();
        let mask = Tensor::from_vec(mask_vals, (1, 1, t, t), x.device())?;
        let att = nn::ops::softmax(&att.broadcast_add(&mask)?, D::Minus1)?;

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
    wte: nn::Embedding,
    wpe: nn::Embedding,
    blocks: Vec<Block>,
    ln_f: nn::LayerNorm,
    lm_head: nn::Linear,
    block_size: usize,
}

impl Gpt {
    fn new(vs: nn::VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            wte: nn::embedding(cfg.vocab_size, cfg.n_embd, vs.pp("wte"))?,
            wpe: nn::embedding(cfg.block_size, cfg.n_embd, vs.pp("wpe"))?,
            blocks: (0..cfg.n_layer)
                .map(|i| Block::new(vs.pp(format!("h.{i}")), cfg.n_embd, cfg.n_head))
                .collect::<Result<_>>()?,
            ln_f: nn::layer_norm(
                cfg.n_embd,
                nn::LayerNormConfig::default(),
                vs.pp("ln_f"),
            )?,
            lm_head: nn::linear(cfg.n_embd, cfg.vocab_size, vs.pp("lm_head"))?,
            block_size: cfg.block_size,
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

    // 给定当前 token 序列，返回下一个 token 的 logits (vocab_size,)
    fn next_logits(&self, tokens: &[u32], device: &Device) -> Result<Tensor> {
        let context: Vec<u32> = if tokens.len() > self.block_size {
            tokens[tokens.len() - self.block_size..].to_vec()
        } else {
            tokens.to_vec()
        };
        let t = context.len();
        let input = Tensor::from_vec(context, (1, t), device)?;
        let logits = self.forward(&input)?; // (1, t, vocab_size)
        Ok(logits.i((0, t - 1))?) // (vocab_size,)
    }
}

// ── Sampling ──────────────────────────────────────────────────────────────────

fn sample_token(logits: &Tensor, temperature: f64, rng: &mut impl Rng) -> Result<u32> {
    let logits = logits.affine(1.0 / temperature, 0.0)?;
    let probs = nn::ops::softmax(&logits, D::Minus1)?;
    let probs: Vec<f32> = probs.to_vec1()?;

    let r: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(i as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

// ── Device ────────────────────────────────────────────────────────────────────
fn detect_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        return Ok(Device::new_cuda(0)?);
    }
    Ok(Device::Cpu)
}

// ── CLI args ──────────────────────────────────────────────────────────────────
struct Args {
    prefix: String,
    temperature: f64,
    max_tokens: usize,
}

impl Args {
    fn parse() -> Result<Self> {
        let argv: Vec<String> = std::env::args().collect();
        let mut prefix = String::new();
        let mut temperature = 0.8f64;
        let mut max_tokens = 200usize;

        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "--temperature" | "-t" => {
                    i += 1;
                    temperature = argv.get(i)
                        .with_context(|| "--temperature requires a value")?
                        .parse()?;
                }
                "--max-tokens" | "-n" => {
                    i += 1;
                    max_tokens = argv.get(i)
                        .with_context(|| "--max-tokens requires a value")?
                        .parse()?;
                }
                arg if !arg.starts_with('-') => {
                    prefix = arg.to_string();
                }
                other => anyhow::bail!("unknown argument: {}", other),
            }
            i += 1;
        }

        anyhow::ensure!(!prefix.is_empty(), "usage: infer <prefix> [--temperature 0.8] [--max-tokens 200]");
        Ok(Self { prefix, temperature, max_tokens })
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
fn main() -> Result<()> {
    let args = Args::parse()?;
    let dir = data_dir();

    let cfg = Config::load(&dir.join("config.txt"))?;
    let vocab = Vocab::load(&dir.join("vocab.txt"))?;

    let device = detect_device()?;
    let vb = unsafe {
        nn::VarBuilder::from_mmaped_safetensors(
            &[dir.join("model.safetensors")],
            DType::F32,
            &device,
        )?
    };
    let model = Gpt::new(vb, &cfg)?;

    let mut tokens = vocab.encode(&args.prefix);
    anyhow::ensure!(
        !tokens.is_empty(),
        "prefix '{}' contains no known characters",
        args.prefix
    );

    // Print the prefix as-is, then stream generated characters
    print!("{}", args.prefix);
    std::io::stdout().flush()?;

    let mut rng = rand::thread_rng();
    for _ in 0..args.max_tokens {
        let logits = model.next_logits(&tokens, &device)?;
        let next = sample_token(&logits, args.temperature, &mut rng)?;

        match vocab.decode_char(next) {
            Some('\n') => break, // end of command
            Some(c) => {
                print!("{c}");
                std::io::stdout().flush()?;
            }
            None => break,
        }
        tokens.push(next);
    }
    println!();

    Ok(())
}
