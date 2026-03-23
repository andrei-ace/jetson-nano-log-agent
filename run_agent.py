"""ReAct agent for investigating logs on Jetson Orin Nano."""

import os
import re
import shlex
import subprocess
import sys

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_DIR = os.environ.get("LOG_DIR", "/workspace/demo_logs")
MAX_OUTPUT = 8000
CMD_TIMEOUT = 10

ALLOWED_CMDS = frozenset(
    ["rg", "grep", "sed", "awk", "head", "tail", "cat", "ls", "wc",
     "find", "sort", "uniq", "cut", "tr", "diff", "xargs", "date"]
)

DANGEROUS_RE = re.compile(
    r"|".join([
        r"\brm\b", r"\bmv\b", r"\bcp\b", r"\bchmod\b", r"\bchown\b",
        r"\bsudo\b", r"\bcurl\b", r"\bwget\b", r"\bnc\b", r"\bncat\b",
        r"\bdd\b", r"\bmkfs\b", r"\bpython[23]?\b", r"\bperl\b",
        r"\bruby\b", r"\bbash\b", r"\bsh\b", r"\bzsh\b",
        r">>?\s", r"\|&", r"&\s*$", r"&\s*;",
        r"\.\./", r"`", r"\$\(",
    ])
)

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
    """Run a read-only shell command. Available: grep, awk, sed, head, tail, cat, ls, wc, find, sort, uniq, cut, tr, diff, xargs, date. Pipes allowed. cwd is the logs folder.

    Examples:
        shell("date -u '+%Y-%m-%dT%H:%M:%SZ'")
        shell("grep ERROR *.log")
        shell("grep 'sess-a1b2c3' *.log")
        shell("grep ERROR *.log | grep '2026-03-23T18:'")
        shell("awk '/throttle/{print}' thermal.log | head -20")
    """
    for segment in command.split("|"):
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError as exc:
            return f"Parse error: {exc}"
        if not tokens:
            continue
        cmd_name = os.path.basename(tokens[0])
        if cmd_name not in ALLOWED_CMDS:
            return f"Blocked: '{cmd_name}' is not allowed."

    if DANGEROUS_RE.search(command):
        return "Blocked: disallowed pattern."

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

Files: app.log (ISO timestamps), thermal.log (syslog timestamps), dmesg.log (boot-seconds). All UTC.

RULES:
1. For time questions ("past hour", "recently"), FIRST get current time, then filter.
2. Always use *.log to search all files.
3. Keep answers short — facts and evidence only.

TIMESTAMP FILTERING (app.log only — ISO timestamps sort lexically):
  shell("awk '$1 >= \"2026-03-23T19:00\"' app.log | grep ERROR")
  shell("awk '$1 >= \"2026-03-23T19:00\"' app.log | grep WARN | head -20")
NOTE: awk filters ONE file at a time. To search all files, run grep first:
  shell("grep ERROR *.log")
Then filter by time with awk on a specific file.
To get current time:
  shell("date -u '+%Y-%m-%dT%H:%M:%SZ'")"""


def build_agent():
    llm = ChatOpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        model=os.environ.get("OPENAI_MODEL", "local-model"),
        temperature=0.6,
    )
    return create_react_agent(model=llm, tools=[shell], prompt=PROMPT)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DIM = "\033[2m"
RESET = "\033[0m"


def _print_content(text: str):
    """Print model output with <think> blocks dimmed and the rest normal."""
    parts = re.split(r"(<think>.*?</think>)", text, flags=re.DOTALL)
    for part in parts:
        if part.startswith("<think>"):
            thinking = part[7:-8].strip()
            if thinking:
                for line in thinking.split("\n"):
                    print(f"{DIM}  {line}{RESET}")
        else:
            clean = part.strip()
            if clean:
                print(f"\n{clean}\n")


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
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                if node_name == "agent":
                    msg = update["messages"][-1]
                    # Show reasoning/thinking if present
                    reasoning = getattr(msg, "additional_kwargs", {}).get("reasoning_content", "")
                    if reasoning:
                        for line in reasoning.strip().split("\n"):
                            print(f"{DIM}  {line}{RESET}")
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            cmd = tc.get("args", {}).get("command", "")
                            print(f"  $ {cmd}")
                    if hasattr(msg, "content") and msg.content:
                        _print_content(msg.content)
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
