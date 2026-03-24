"""ReAct agent for investigating logs on Jetson Orin Nano."""

import os
import re
import subprocess
from datetime import datetime, timezone

import json

import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_DIR = os.environ.get("LOG_DIR", "/workspace/demo_logs")
KB_DIR = os.environ.get("KB_DIR", "./kb_index")
EMAIL_LOG = os.environ.get("EMAIL_LOG", "/tmp/sent_emails.log")
MAX_OUTPUT = 2000
CMD_TIMEOUT = 10

SHELL_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
}

# ---------------------------------------------------------------------------
# Low-level tools (used by sub-agents)
# ---------------------------------------------------------------------------


@tool
def shell(command: str) -> str:
    """Run a shell command. cwd is the logs folder. Pipes allowed."""
    try:
        result = subprocess.run(
            command, shell=True, cwd=LOG_DIR, timeout=CMD_TIMEOUT,
            capture_output=True, text=True, env=SHELL_ENV,
        )
    except subprocess.TimeoutExpired:
        return f"Timed out after {CMD_TIMEOUT}s."

    output = result.stdout or result.stderr or "(no output)"
    if len(output) > MAX_OUTPUT:
        output = output[:MAX_OUTPUT] + f"\n... [truncated at {MAX_OUTPUT} chars]"
    return output


# ---------------------------------------------------------------------------
# RAG knowledge base (pre-built FAISS index from field manual)
# ---------------------------------------------------------------------------

_kb_index: faiss.IndexFlatIP | None = None
_kb_chunks: list[tuple[str, str]] | None = None
_kb_model = None


def _ensure_kb_loaded():
    """Load pre-built FAISS index and chunks."""
    global _kb_index, _kb_chunks
    if _kb_index is not None:
        return
    _kb_index = faiss.read_index(os.path.join(KB_DIR, "index.faiss"))
    with open(os.path.join(KB_DIR, "chunks.json")) as f:
        _kb_chunks = json.load(f)


def _get_model():
    """Lazy-load the embedding model."""
    global _kb_model
    if _kb_model is None:
        _fd = os.dup(2)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
        try:
            import onnxruntime as ort
            ort.set_default_logger_severity(3)
            from fastembed import TextEmbedding
        finally:
            os.dup2(_fd, 2)
            os.close(_fd)
        _kb_model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=os.path.join(KB_DIR, "model_cache"),
            local_files_only=True,
        )
    return _kb_model


@tool
def search_manual(query: str) -> str:
    """Search the field manual. Returns the most relevant sections."""
    _ensure_kb_loaded()
    model = _get_model()
    query_emb = np.array(list(model.embed([query])), dtype=np.float32)
    faiss.normalize_L2(query_emb)

    k = min(3, len(_kb_chunks))
    scores, indices = _kb_index.search(query_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        heading, body = _kb_chunks[idx]
        results.append(f"## {heading}\n{body}")
    return "\n\n---\n\n".join(results)


# ---------------------------------------------------------------------------
# Action tools (used by main agent directly)
# ---------------------------------------------------------------------------


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email alert (simulated — logs to file)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    email_block = (
        f"{'=' * 60}\n"
        f"Date: {timestamp}\n"
        f"From: jetson-log-agent@jetson-07\n"
        f"To: {recipient}\n"
        f"Subject: {subject}\n"
        f"{'=' * 60}\n"
        f"{body}\n\n"
    )
    with open(EMAIL_LOG, "a") as f:
        f.write(email_block)
    print(f"\n  \033[1m[EMAIL SENT]\033[0m To: {recipient} | Subject: {subject}", flush=True)
    return f"Email sent to {recipient} with subject '{subject}'"


@tool
def reboot_device(device: str, reason: str) -> str:
    """Reboot a Jetson device (simulated — logs to file)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_entry = f"[{timestamp}] REBOOT {device}: {reason}\n"
    with open(EMAIL_LOG, "a") as f:
        f.write(log_entry)
    print(f"\n  \033[1;31m[REBOOT]\033[0m {device} — {reason}", flush=True)
    return f"Reboot command sent to {device}. Reason: {reason}"


# ---------------------------------------------------------------------------
# Sub-agent helpers
# ---------------------------------------------------------------------------

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"


def _run_subagent(agent, label: str, color: str, user_msg: str,
                  recursion_limit: int = 15) -> str:
    """Stream a sub-agent, printing its steps, and return the final answer."""
    prefix = f"  {color}  [{label}]{RESET}"
    final_text = ""

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_msg}]},
        {"recursion_limit": recursion_limit},
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            if node_name == "agent":
                msg = update["messages"][-1]
                has_tools = hasattr(msg, "tool_calls") and msg.tool_calls
                content = msg.content if hasattr(msg, "content") else ""
                reasoning = getattr(msg, "additional_kwargs", {}).get("reasoning_content", "")
                if reasoning:
                    text = re.sub(r"</?think>", "", reasoning).strip()
                    if text:
                        for line in text.split("\n"):
                            print(f"{prefix} {DIM}{line}{RESET}")
                if has_tools:
                    if content:
                        text = re.sub(r"</?think>", "", content).strip()
                        if text:
                            for line in text.split("\n"):
                                print(f"{prefix} {DIM}{line}{RESET}")
                    for tc in msg.tool_calls:
                        args = tc.get("args", {})
                        if tc.get("name") == "shell":
                            print(f"{prefix} {CYAN}${RESET} {args.get('command', '')}")
                        elif tc.get("name") == "search_manual":
                            print(f"{prefix} {GREEN}search:{RESET} {args.get('query', '')}")
                        else:
                            print(f"{prefix} [{tc.get('name')}] {args}")
                elif content:
                    final_text = THINK_RE.sub("", content).strip()
            elif node_name == "tools":
                for msg in update["messages"]:
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    lines = content.strip().split("\n")
                    for line in lines[:4]:
                        print(f"{prefix} {DIM}{line}{RESET}")
                    if len(lines) > 4:
                        print(f"{prefix} {DIM}... ({len(lines) - 4} more lines){RESET}")

    return final_text or "(no result)"


# ---------------------------------------------------------------------------
# Sub-agent: log searcher
# ---------------------------------------------------------------------------

LOG_SEARCH_PROMPT = """\
You search Jetson Orin Nano log files for errors using shell commands.

STEP 1: Get current UTC time.
  date -u '+%Y-%m-%dT%H:%M'

STEP 2: Compute the ISO cutoff by subtracting the requested window from current time.
  date -u -d 'N minutes ago' '+%Y-%m-%dT%H:%M'
  date -u -d 'N hours ago' '+%Y-%m-%dT%H:%M'

STEP 3: Search app.log (ISO timestamps, first field). Levels are level=ERROR, level=WARN.
  awk -v cutoff=CUTOFF '$1 >= cutoff' app.log | grep -E 'level=ERROR|level=WARN'

STEP 4: Search thermal.log (syslog timestamps, cannot filter by time).
  grep -E 'ERROR|WARN' thermal.log

STEP 5: Search dmesg.log (seconds-since-boot, cannot filter by time).
  grep -E 'CRITICAL|WARNING' dmesg.log

Do NOT reason about quoting or syntax. Just run the commands above, substituting \
the cutoff value you computed. Return the output grouped by file."""

_log_agent = None


@tool
def search_logs(time_window: str) -> str:
    """Search all log files for errors within a time window."""
    return _run_subagent(_log_agent, "logs", CYAN,
                         f"Find errors in the {time_window}")


# ---------------------------------------------------------------------------
# Sub-agent: manual consultant
# ---------------------------------------------------------------------------

MANUAL_AGENT_PROMPT = """\
You are a field manual consultant. When given a query, search the manual \
and return a concise summary: the section title, severity, and recommended \
actions. Nothing else."""

_manual_agent = None


@tool
def consult_manual(query: str) -> str:
    """Consult the field manual about an issue. Returns severity and actions."""
    return _run_subagent(_manual_agent, "manual", GREEN,
                         query, recursion_limit=10)


# ---------------------------------------------------------------------------
# Main agent (router)
# ---------------------------------------------------------------------------

def _load_procedure() -> str:
    """Read the Investigation Procedure section from the field manual."""
    with open(os.path.join(KB_DIR, "chunks.json")) as f:
        chunks = json.load(f)
    for heading, body in chunks:
        if heading == "Investigation Procedure":
            return body
    raise RuntimeError("Investigation Procedure not found in field manual")


PROMPT = """\
You investigate Jetson Orin Nano hardware logs.

Files (all UTC):
- app.log: ISO timestamps, levels: ERROR/WARN/INFO
- thermal.log: syslog timestamps, levels: ERROR/WARN/INFO
- dmesg.log: [seconds-since-boot], keywords: CRITICAL/WARNING

PROCEDURE:
""" + _load_procedure()


def build_agents():
    global _log_agent, _manual_agent

    llm = ChatOpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        model=os.environ.get("OPENAI_MODEL", "local-model"),
        temperature=0.6,
    )

    _log_agent = create_react_agent(model=llm, tools=[shell],
                                     prompt=LOG_SEARCH_PROMPT)
    _manual_agent = create_react_agent(model=llm, tools=[search_manual],
                                        prompt=MANUAL_AGENT_PROMPT)

    return create_react_agent(
        model=llm,
        tools=[search_logs, consult_manual, send_email, reboot_device],
        prompt=PROMPT,
        checkpointer=MemorySaver(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
YELLOW = "\033[33m"

TOOL_PREVIEW_LINES = 6


def _print_thinking(text: str):
    """Show model reasoning dimmed."""
    text = re.sub(r"</?think>", "", text).strip()
    if not text:
        return
    for line in text.split("\n"):
        print(f"  {DIM}{line}{RESET}")


def _print_answer(text: str):
    text = THINK_RE.sub("", text).strip()
    if text:
        print(f"\n{text}\n")


def _print_tool_call(tc: dict):
    """Print a tool call with colored formatting."""
    name = tc.get("name", "")
    args = tc.get("args", {})
    if name == "search_logs":
        print(f"  {CYAN}[logs]{RESET} {args.get('time_window', '')}")
    elif name == "consult_manual":
        print(f"  {GREEN}[manual]{RESET} {args.get('query', '')}")
    elif name == "send_email":
        print(f"  {YELLOW}[email]{RESET} To: {args.get('recipient', '')} | {args.get('subject', '')}")
    elif name == "reboot_device":
        print(f"  \033[1;31m[reboot]\033[0m {args.get('device', '')} — {args.get('reason', '')}")
    else:
        parts = " ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"  [{name}] {parts}")


def _print_tool_result(content: str):
    """Print a compact preview of a tool result."""
    lines = content.strip().split("\n")
    preview = "\n".join(f"  {DIM}{l}{RESET}" for l in lines[:TOOL_PREVIEW_LINES])
    if len(lines) > TOOL_PREVIEW_LINES:
        preview += f"\n  {DIM}... ({len(lines) - TOOL_PREVIEW_LINES} more lines){RESET}"
    print(preview)


def main():
    print("Connecting to LLM server...")
    agent = build_agents()
    print("Jetson Log Agent ready. Type your question (or 'quit' to exit).\n")

    while True:
        try:
            question = input(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        print()
        config = {"configurable": {"thread_id": "session"}, "recursion_limit": 50}
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                if node_name == "agent":
                    msg = update["messages"][-1]
                    has_tools = hasattr(msg, "tool_calls") and msg.tool_calls
                    content = msg.content if hasattr(msg, "content") else ""
                    reasoning = getattr(msg, "additional_kwargs", {}).get("reasoning_content", "")
                    if reasoning:
                        _print_thinking(reasoning)
                    if has_tools:
                        if content:
                            _print_thinking(content)
                        for tc in msg.tool_calls:
                            _print_tool_call(tc)
                    elif content:
                        _print_answer(content)
                elif node_name == "tools":
                    for msg in update["messages"]:
                        content = msg.content if hasattr(msg, "content") else str(msg)
                        _print_tool_result(content)


if __name__ == "__main__":
    main()
