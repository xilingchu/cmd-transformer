# ─────────────────────────────────────────────────────────────
# cmd-transformer: zsh hook integration
# Add this to the end of your ~/.zshrc, or source this file:
#
# source ~/.config/cmd-transformer/hook.zsh
# ─────────────────────────────────────────────────────────────

# Path to the cmd-record binary; usually here after `cargo install`
# If using `cargo run`, set the full path instead:
# CMD_RECORD_BIN="/path/to/cmd-transformer/target/release/cmd-record"
CMD_RECORD_BIN="${CMD_RECORD_BIN:-cmd-record}"

# Check if cmd-record is available; silently disable if not, without affecting normal shell use
if ! command -v "$CMD_RECORD_BIN" &>/dev/null; then
    # Fall back to the project target directory
    _local_bin="$HOME/cmd-transformer/target/release/cmd-record"
    if [[ -x "$_local_bin" ]]; then
        CMD_RECORD_BIN="$_local_bin"
    else
        return 0  # Not found — exit silently without error
    fi
fi

# ── Internal variables (do not modify manually) ──────────────
_CMD_RECORD_CMD=""
_CMD_RECORD_CWD=""
_CMD_RECORD_TIME=0
# Unique session ID from PID + timestamp; refreshed on each terminal start
_CMD_RECORD_SESSION="${$}-$(date +%s)"

# ── preexec: fired before a command executes ─────────────────
# $1 = raw command string as typed by the user
_cmd_record_preexec() {
    _CMD_RECORD_CMD="$1"
    _CMD_RECORD_CWD="$PWD"
    # Millisecond timestamp (used to compute duration)
    _CMD_RECORD_TIME=$(( $(date +%s%N) / 1000000 ))
}

# ── precmd: fired after a command finishes ───────────────────
# $? holds the exit code of the previous command at this point
_cmd_record_precmd() {
    local exit_code=$?

    # Skip if there is no pending command (e.g. user pressed Enter on an empty line)
    [[ -z "$_CMD_RECORD_CMD" ]] && return

    # Compute elapsed time in milliseconds
    local now=$(( $(date +%s%N) / 1000000 ))
    local duration=$(( now - _CMD_RECORD_TIME ))
    [[ $duration -lt 0 ]] && duration=0

    # Run asynchronously so it never blocks the prompt
    # &! detaches the job; 2>/dev/null suppresses any error output
    "$CMD_RECORD_BIN" \
        --cmd      "$_CMD_RECORD_CMD" \
        --cwd      "$_CMD_RECORD_CWD" \
        --exit     "$exit_code"       \
        --duration "$duration"        \
        --session  "$_CMD_RECORD_SESSION" \
        2>/dev/null &!

    # Clear to prevent duplicate recording
    _CMD_RECORD_CMD=""
}

# ── Register hooks ────────────────────────────────────────────
autoload -Uz add-zsh-hook
add-zsh-hook preexec _cmd_record_preexec
add-zsh-hook precmd  _cmd_record_precmd
