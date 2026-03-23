"""ReAct agent for investigating logs on Jetson Orin Nano."""

import os
import re
import subprocess
import sys

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_DIR = os.environ.get("LOG_DIR", "/workspace/demo_logs")
MAX_OUTPUT = 2000
CMD_TIMEOUT = 10

SHELL_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
}

# ---------------------------------------------------------------------------
# Shell tool
# ---------------------------------------------------------------------------


@tool
def shell(command: str) -> str:
    """Run a shell command. cwd is the logs folder. Pipes allowed.

    Examples:
        shell("date -u '+%Y-%m-%dT%H:%M:%SZ'")
        shell("grep ERROR *.log")
        shell("grep 'sess-a1b2c3' *.log")
        shell("awk '$1 >= \"2026-03-23T19:00\"' app.log | grep ERROR")
    """
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
# Agent
# ---------------------------------------------------------------------------

PROMPT = """\
You investigate Jetson Orin Nano hardware logs. Be concise.

Files (all UTC):
- app.log: ISO timestamps, levels: ERROR/WARN/INFO
- thermal.log: syslog timestamps, levels: ERROR/WARN/INFO
- dmesg.log: [seconds-since-boot], keywords: CRITICAL/WARNING

RULES:
1. For time questions, FIRST get current time, then filter.
2. Always check ALL three files.
3. Keep answers short — facts and evidence only.

COMMANDS:
  date -u '+%Y-%m-%dT%H:%M:%SZ'           # current UTC time
  grep ERROR app.log | awk '$1 >= "CUTOFF"' # filter app.log by time
  grep ERROR thermal.log                    # thermal.log errors
  grep CRITICAL dmesg.log                   # kernel critical events"""


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_old_thinking(state):
    """Pre-model hook: strip <think> from previous turns, keep current turn's thinking."""
    messages = state["messages"]
    # Find the last user message — everything before it is a previous turn
    last_user_idx = 0
    for i, msg in enumerate(messages):
        if hasattr(msg, "type") and msg.type == "human":
            last_user_idx = i
    cleaned = []
    for i, msg in enumerate(messages):
        if i < last_user_idx and hasattr(msg, "content") and isinstance(msg.content, str) and "<think>" in msg.content:
            msg = msg.model_copy(update={"content": THINK_RE.sub("", msg.content).strip()})
        cleaned.append(msg)
    return {"messages": cleaned}


def build_agent():
    llm = ChatOpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        model=os.environ.get("OPENAI_MODEL", "local-model"),
        temperature=0.6,
    )
    return create_react_agent(
        model=llm, tools=[shell], prompt=PROMPT,
        pre_model_hook=_strip_old_thinking,
        checkpointer=MemorySaver(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DIM = "\033[2m"
RESET = "\033[0m"


def _print_thinking(text: str):
    text = re.sub(r"</?think>", "", text).strip()
    if text:
        for line in text.split("\n"):
            print(f"{DIM}  {line}{RESET}")


def _print_answer(text: str):
    text = THINK_RE.sub("", text).strip()
    if text:
        print(f"\n{text}\n")


def main():
    print("Connecting to LLM server...")
    agent = build_agent()
    print("Jetson Log Agent ready. Type your question (or 'quit' to exit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        print()
        config = {"configurable": {"thread_id": "session"}}
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
                            cmd = tc.get("args", {}).get("command", "")
                            print(f"  $ {cmd}")
                    elif content:
                        _print_answer(content)
                elif node_name == "tools":
                    for msg in update["messages"]:
                        content = msg.content if hasattr(msg, "content") else str(msg)
                        lines = content.strip().split("\n")
                        preview = "\n".join(f"  {l}" for l in lines[:10])
                        if len(lines) > 10:
                            preview += f"\n  ... ({len(lines) - 10} more lines)"
                        print(preview)


if __name__ == "__main__":
    main()
