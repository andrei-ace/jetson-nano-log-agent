"""ReAct agent for investigating logs on Jetson Orin Nano."""

import json
import os
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

LOG_DIR = os.environ.get("LOG_DIR", "/workspace/demo_logs")
KB_DIR = os.environ.get("KB_DIR", "./kb_index")
ACTION_LOG = os.environ.get("ACTION_LOG",
                            os.environ.get("EMAIL_LOG", "/tmp/actions.log"))
MAX_OUTPUT = 4000
CMD_TIMEOUT = 10

SHELL_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
}

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[1;31m"


@contextmanager
def _suppress_onnx_warnings():
    """Redirect fd 2 to /dev/null during ONNX Runtime import."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


# --- Tools ---

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


# --- RAG ---

class KnowledgeBase:
    """FAISS index + embeddings + cross-encoder re-ranker for the field manual."""

    EMBED_MODEL = "BAAI/bge-small-en-v1.5"
    RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

    def __init__(self, kb_dir: str):
        self._dir = kb_dir
        self._index = None
        self._chunks: list[tuple[str, str]] | None = None
        self._embedder = None
        self._reranker = None

    def _load(self):
        if self._index is not None:
            return
        import faiss
        self._index = faiss.read_index(os.path.join(self._dir, "index.faiss"))
        with open(os.path.join(self._dir, "chunks.json")) as f:
            self._chunks = json.load(f)

    def _load_models(self):
        if self._embedder is not None:
            return
        with _suppress_onnx_warnings():
            import onnxruntime as ort
            ort.set_default_logger_severity(3)
            from fastembed import TextEmbedding
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        cache = os.path.join(self._dir, "model_cache")
        self._embedder = TextEmbedding(
            model_name=self.EMBED_MODEL, cache_dir=cache, local_files_only=True,
        )
        self._reranker = TextCrossEncoder(
            model_name=self.RERANK_MODEL, cache_dir=cache, local_files_only=True,
        )

    @property
    def chunks(self) -> list[tuple[str, str]]:
        self._load()
        return self._chunks

    def search(self, query: str, k: int = 3) -> list[tuple[str, str]]:
        import faiss
        import numpy as np

        self._load()
        self._load_models()

        # Stage 1: FAISS retrieves broad candidates
        query_emb = np.array(list(self._embedder.embed([query])), dtype=np.float32)
        faiss.normalize_L2(query_emb)
        n_candidates = min(k * 2, len(self._chunks))
        _, indices = self._index.search(query_emb, n_candidates)

        candidates = [self._chunks[idx] for idx in indices[0]]
        docs = [f"{heading}\n{body}" for heading, body in candidates]

        # Stage 2: cross-encoder re-ranks for precision
        scores = list(self._reranker.rerank(query, docs))
        ranked = sorted(zip(scores, candidates), reverse=True)

        return [chunk for _, chunk in ranked[:k]]


_kb = KnowledgeBase(KB_DIR)


@tool
def search_manual(query: str) -> str:
    """Search the field manual. Returns the most relevant sections."""
    results = []
    for heading, body in _kb.search(query):
        results.append(f"## {heading}\n{body}")
    return "\n\n---\n\n".join(results)


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
    with open(ACTION_LOG, "a") as f:
        f.write(email_block)
    print(f"\n  {BOLD}[EMAIL SENT]{RESET} To: {recipient} | Subject: {subject}",
          flush=True)
    return f"Email sent to {recipient} with subject '{subject}'"


@tool
def reboot_device(device: str, reason: str) -> str:
    """Reboot a Jetson device (simulated — logs to file)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_entry = f"[{timestamp}] REBOOT {device}: {reason}\n"
    with open(ACTION_LOG, "a") as f:
        f.write(log_entry)
    print(f"\n  {RED}[REBOOT]{RESET} {device} — {reason}", flush=True)
    return f"Reboot command sent to {device}. Reason: {reason}"


# --- Stream display ---

def _print_stream(prefix: str, chunk: dict):
    """Process one stream chunk and print formatted output."""
    for node_name, update in chunk.items():
        if node_name == "agent":
            msg = update["messages"][-1]
            has_tools = hasattr(msg, "tool_calls") and msg.tool_calls
            content = msg.content if hasattr(msg, "content") else ""
            reasoning = getattr(msg, "additional_kwargs", {}).get(
                "reasoning_content", "")
            if reasoning:
                _print_dimmed(prefix, reasoning)
            if has_tools:
                if content:
                    _print_dimmed(prefix, content)
                for tc in msg.tool_calls:
                    _print_tool(prefix, tc)
            elif content:
                text = THINK_RE.sub("", content).strip()
                if text and not prefix:
                    print(f"\n{text}\n")
        elif node_name == "tools":
            for msg in update["messages"]:
                content = msg.content if hasattr(msg, "content") else str(msg)
                _print_preview(prefix, content)


def _print_dimmed(prefix: str, text: str):
    text = re.sub(r"</?think>", "", text).strip()
    if not text:
        return
    for line in text.split("\n"):
        print(f"{prefix} {DIM}{line}{RESET}")


def _print_tool(prefix: str, tc: dict):
    name = tc.get("name", "")
    args = tc.get("args", {})
    fmt = {
        "shell": lambda: f"{CYAN}${RESET} {args.get('command', '')}",
        "search_manual": lambda: f"{GREEN}search:{RESET} {args.get('query', '')}",
        "search_logs": lambda: f"{CYAN}[logs]{RESET} {args.get('time_window', '')}",
        "consult_manual": lambda: f"{GREEN}[manual]{RESET} {args.get('query', '')}",
        "send_email": lambda: f"{YELLOW}[email]{RESET} To: {args.get('recipient', '')} | {args.get('subject', '')}",
        "reboot_device": lambda: f"{RED}[reboot]{RESET} {args.get('device', '')} — {args.get('reason', '')}",
    }
    line = fmt.get(name, lambda: f"[{name}] {args}")()
    print(f"{prefix} {line}")


def _print_preview(prefix: str, content: str, max_lines: int = 6):
    lines = content.strip().split("\n")
    for line in lines[:max_lines]:
        print(f"{prefix} {DIM}{line}{RESET}")
    if len(lines) > max_lines:
        print(f"{prefix} {DIM}... ({len(lines) - max_lines} more lines){RESET}")


# --- Sub-agents ---

def _run_subagent(agent, label: str, color: str, user_msg: str,
                  recursion_limit: int = 15) -> str:
    prefix = f"  {color}  [{label}]{RESET}"
    final_text = ""

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_msg}]},
        {"recursion_limit": recursion_limit},
        stream_mode="updates",
    ):
        _print_stream(prefix, chunk)
        # Capture final answer from agent node
        for node_name, update in chunk.items():
            if node_name == "agent":
                msg = update["messages"][-1]
                has_tools = hasattr(msg, "tool_calls") and msg.tool_calls
                content = msg.content if hasattr(msg, "content") else ""
                if not has_tools and content:
                    final_text = THINK_RE.sub("", content).strip()

    return final_text or "(no result)"


LOG_SEARCH_PROMPT = """\
Search Jetson log files for errors using shell commands.

STEP 1: Compute all cutoffs in one command.
  ISO_CUT=$(date -u -d 'N hours ago' '+%Y-%m-%dT%H:%M'); SYS_CUT=$(date -u -d 'N hours ago' '+%H:%M:%S'); DMESG_LAST=$(tail -1 dmesg.log | sed 's/\\[\\([0-9.]*\\)\\].*/\\1/'); DMESG_CUT=$(echo "$DMESG_LAST - N * 3600" | bc); echo "ISO=$ISO_CUT SYS=$SYS_CUT DMESG=$DMESG_CUT"
  Replace N with the number of hours. For minutes use 'N minutes ago' and N * 60.

STEP 2: Search each file (one command per file, substitute the cutoff values).
  awk -v c=ISO_CUT '$1>=c && /level=ERROR/' app.log
  awk -v c=ISO_CUT '$1>=c && /level=WARN/' app.log
  awk -v c=SYS_CUT '$3>=c && /ERROR|WARN/' thermal.log
  awk -F'[][]' -v c=DMESG_CUT '$2+0>=c && /CRITICAL|WARNING/' dmesg.log

Deduplicate: if the same error repeats, report it once with the count \
and time range."""

_log_agent = None


@tool
def search_logs(time_window: str) -> str:
    """Search all log files for errors within a time window."""
    return _run_subagent(_log_agent, "logs", CYAN,
                         f"Find errors in the {time_window}")


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


# --- Main agent ---

def _load_procedure() -> str:
    for heading, body in _kb.chunks:
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


# --- CLI ---

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
            _print_stream("", chunk)


if __name__ == "__main__":
    main()
