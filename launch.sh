#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Create venv and install deps ────────────────────────────────────────────
if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
    echo "Creating virtual environment..."
    uv venv "$SCRIPT_DIR/.venv"
fi

echo "Installing dependencies..."
uv pip install --python "$SCRIPT_DIR/.venv/bin/python" 'langchain-openai>=0.3,<1' 'langgraph>=0.2,<1'

# ── Generate logs with current timestamps ────────────────────────────────────
echo "Generating logs..."
"$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/gen_logs.py"

# ── Resolve paths ───────────────────────────────────────────────────────────
VENV_DIR="$(readlink -f "$SCRIPT_DIR/.venv")"

# ── Launch in bwrap sandbox ─────────────────────────────────────────────────
# Note: --unshare-net is NOT used because the agent needs localhost access
# to reach the llama-server. The shell tool's command allowlist blocks
# networking tools (curl, wget, nc).
echo "Launching agent in sandbox..."
exec bwrap \
    --ro-bind /usr /usr \
    --ro-bind /bin /bin \
    --ro-bind /lib /lib \
    --ro-bind-try /lib64 /lib64 \
    --ro-bind /etc/alternatives /etc/alternatives \
    --ro-bind /etc/ld.so.cache /etc/ld.so.cache \
    --ro-bind-try /etc/ld.so.conf /etc/ld.so.conf \
    --ro-bind-try /etc/ld.so.conf.d /etc/ld.so.conf.d \
    --ro-bind-try /etc/ssl /etc/ssl \
    --proc /proc \
    --dev /dev \
    --tmpfs /tmp \
    --ro-bind "$VENV_DIR" /workspace/.venv \
    --ro-bind "$SCRIPT_DIR/run_agent.py" /workspace/run_agent.py \
    --ro-bind "$SCRIPT_DIR/demo_logs" /workspace/demo_logs \
    --unshare-pid \
    --unshare-ipc \
    --unshare-uts \
    --clearenv \
    --setenv PATH "/workspace/.venv/bin:/usr/bin:/bin" \
    --setenv VIRTUAL_ENV "/workspace/.venv" \
    --setenv HOME "/tmp" \
    --setenv LANG "C.UTF-8" \
    --setenv PYTHONDONTWRITEBYTECODE "1" \
    --setenv LOG_DIR "/workspace/demo_logs" \
    --setenv OPENAI_BASE_URL "${OPENAI_BASE_URL:-http://127.0.0.1:8080/v1}" \
    --setenv OPENAI_API_KEY "${OPENAI_API_KEY:-not-needed}" \
    --setenv OPENAI_MODEL "${OPENAI_MODEL:-local-model}" \
    --die-with-parent \
    --chdir /workspace \
    -- /workspace/.venv/bin/python /workspace/run_agent.py
