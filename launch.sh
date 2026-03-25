#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Create venv and install deps ────────────────────────────────────────────
if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
    echo "Creating virtual environment..."
    uv venv "$SCRIPT_DIR/.venv"
fi

echo "Installing dependencies..."
uv pip install --python "$SCRIPT_DIR/.venv/bin/python" 'langchain-openai>=0.3,<1' 'langgraph>=0.2,<1' 'fastembed>=0.4,<1' 'faiss-cpu>=1.7,<2'

# ── Generate logs with current timestamps ────────────────────────────────────
echo "Generating logs..."
"$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/gen_logs.py"

# ── Build RAG index (skip if already built and docs unchanged) ────────────────
INDEX_DIR="$SCRIPT_DIR/kb_index"
DOCS_CHANGED=false
if [[ ! -f "$INDEX_DIR/index.faiss" ]]; then
    DOCS_CHANGED=true
else
    for doc in "$SCRIPT_DIR/docs/"*.md; do
        [[ "$doc" -nt "$INDEX_DIR/index.faiss" ]] && DOCS_CHANGED=true && break
    done
fi
if $DOCS_CHANGED; then
    echo "Building knowledge base index..."
    DOCS_DIR="$SCRIPT_DIR/docs" "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/build_index.py"
fi

# ── Ensure output dir exists ───────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/output"

# ── Resolve paths ───────────────────────────────────────────────────────────
VENV_DIR="$(readlink -f "$SCRIPT_DIR/.venv")"

# ── Set up network bridge (socat) for sandbox ──────────────────────────────
# The sandbox uses --unshare-net for network isolation. A socat bridge
# forwards a Unix socket to the llama-server on the host, so the agent
# can only reach 127.0.0.1:8080 — nothing else.
BRIDGE_SOCK="$SCRIPT_DIR/.llama-bridge.sock"
rm -f "$BRIDGE_SOCK"
socat UNIX-LISTEN:"$BRIDGE_SOCK",fork TCP:127.0.0.1:${LLAMA_PORT:-8080} &
BRIDGE_PID=$!
cleanup() { kill "$BRIDGE_PID" 2>/dev/null; rm -f "$BRIDGE_SOCK"; }
trap cleanup EXIT

# Wait for socket to be created
for _ in $(seq 1 20); do [ -S "$BRIDGE_SOCK" ] && break; sleep 0.1; done
if [[ ! -S "$BRIDGE_SOCK" ]]; then
    echo "ERROR: socat bridge failed to start" >&2
    exit 1
fi

# ── Launch in bwrap sandbox ─────────────────────────────────────────────────
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
    --ro-bind /sys/devices/system/cpu /sys/devices/system/cpu \
    --proc /proc \
    --dev /dev \
    --tmpfs /tmp \
    --ro-bind "$VENV_DIR" /workspace/.venv \
    --ro-bind "$SCRIPT_DIR/run_agent.py" /workspace/run_agent.py \
    --ro-bind "$SCRIPT_DIR/demo_logs" /workspace/demo_logs \
    --ro-bind "$SCRIPT_DIR/kb_index" /workspace/kb_index \
    --bind "$SCRIPT_DIR/output" /workspace/output \
    --ro-bind "$BRIDGE_SOCK" /workspace/llama.sock \
    --unshare-pid \
    --unshare-ipc \
    --unshare-uts \
    --unshare-net \
    --clearenv \
    --setenv PATH "/workspace/.venv/bin:/usr/bin:/bin" \
    --setenv VIRTUAL_ENV "/workspace/.venv" \
    --setenv HOME "/tmp" \
    --setenv LANG "C.UTF-8" \
    --setenv PYTHONDONTWRITEBYTECODE "1" \
    --setenv LOG_DIR "/workspace/demo_logs" \
    --setenv KB_DIR "/workspace/kb_index" \
    --setenv ACTION_LOG "/workspace/output/actions.log" \
    --setenv OPENAI_BASE_URL "http://127.0.0.1:8080/v1" \
    --setenv OPENAI_API_KEY "${OPENAI_API_KEY:-not-needed}" \
    --setenv OPENAI_MODEL "${OPENAI_MODEL:-local-model}" \
    --die-with-parent \
    --chdir /workspace \
    -- /bin/sh -c 'socat TCP-LISTEN:8080,fork,bind=127.0.0.1 UNIX-CONNECT:/workspace/llama.sock & exec /workspace/.venv/bin/python /workspace/run_agent.py'
