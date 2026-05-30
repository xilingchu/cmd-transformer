use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn db_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("cmd-transformer")
        .join("history.db")
}

struct Args {
    cmd: String,
    cwd: String,
    exit_code: i32,
    duration: i64,
    session: String,
}

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = std::env::args().collect();
    let mut cmd = String::new();
    let mut cwd = String::new();
    let mut exit_code: i32 = 0;
    let mut duration: i64 = 0;
    let mut session = String::new();

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--cmd" => {
                i += 1;
                cmd = argv.get(i).cloned().unwrap_or_default();
            }
            "--cwd" => {
                i += 1;
                cwd = argv.get(i).cloned().unwrap_or_default();
            }
            "--exit" => {
                i += 1;
                exit_code = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--duration" => {
                i += 1;
                duration = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--session" => {
                i += 1;
                session = argv.get(i).cloned().unwrap_or_default();
            }
            _ => {}
        }
        i += 1;
    }

    Ok(Args { cmd, cwd, exit_code, duration, session })
}

fn ensure_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS commands (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cmd         TEXT    NOT NULL,
            cwd         TEXT    NOT NULL,
            exit_code   INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL,
            session     TEXT    NOT NULL,
            recorded_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_session     ON commands(session);
        CREATE INDEX IF NOT EXISTS idx_recorded_at ON commands(recorded_at);",
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;

    if args.cmd.is_empty() {
        return Ok(());
    }

    let path = db_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create DB directory {:?}", parent))?;
    }

    let conn = Connection::open(&path)
        .with_context(|| format!("cannot open database {:?}", path))?;

    ensure_schema(&conn)?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs() as i64;

    conn.execute(
        "INSERT INTO commands (cmd, cwd, exit_code, duration_ms, session, recorded_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![args.cmd, args.cwd, args.exit_code, args.duration, args.session, now],
    )?;

    Ok(())
}
