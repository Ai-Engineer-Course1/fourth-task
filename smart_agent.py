import json
import os
import sys
import threading
from typing import Any

from openai import OpenAI

from tools import TOOL_SCHEMA, check_inventory

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL = os.environ.get("AI_BASE_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("AI_BASE_URL")
API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """You are Alex, a friendly and knowledgeable customer service agent at a home appliance store that sells cellphones, headphones, and laptops.

Behave like a real human support representative:
- Greet the customer warmly and speak naturally, in a conversational tone.
- When the customer asks about product availability, ALWAYS use the check_inventory tool before answering. Never guess stock.
- Call check_inventory once per distinct item the customer asks about.
- Some items exist in several models (e.g. MacBook Air and MacBook Pro, multiple AirPods variants). Report every matching model and its stock amount.
- If an item is not found, politely say it is not in stock and offer help with something else.
- Keep answers clear, reassuring, and easy to read. Do not mention tools, functions, JSON, or internal details — just answer as a helpful human would.
"""

TOOLS = [{"type": "function", "function": TOOL_SCHEMA}]


class _Spinner:
    """Simple stderr spinner shown while the agent is waiting for output.

    Only renders when stderr is a TTY, so piped/redirected output stays clean.
    """

    FRAMES = "|/-\\"

    def __init__(self, message: str = "Thinking"):
        self._message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not sys.stderr.isatty():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        i = 0
        while not self._stop.wait(0.1):
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stderr.write(f"\r{self._message} {frame} ")
            sys.stderr.flush()
            i += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
            sys.stderr.write("\r" + " " * (len(self._message) + 4) + "\r")
            sys.stderr.flush()


def _stream_turn(
    client: OpenAI, messages: list[dict[str, str | int]]
) -> dict[str, str | int]:
    """Stream one assistant turn, printing content as it arrives.

    Returns the assembled assistant message (content + tool_calls).
    """
    spinner = _Spinner()
    spinner.start()
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            stream=True,
        )
    except Exception:
        spinner.stop()
        raise

    content = ""
    tool_calls: dict[int, dict[str, str | int]] = {}

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content or delta.tool_calls:
            spinner.stop()

        if delta.content:
            content += delta.content
            print(delta.content, end="", flush=True)

        for tc in delta.tool_calls or []:
            entry = tool_calls.setdefault(
                tc.index, {"id": "", "name": "", "arguments": ""}
            )
            if tc.id:
                entry["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    entry["name"] = tc.function.name
                if tc.function.arguments:
                    entry["arguments"] += tc.function.arguments

    spinner.stop()
    return {"content": content, "tool_calls": tool_calls}


def run_agent(user_input: str) -> None:
    if not API_KEY:
        raise SystemExit(
            "Missing API key. Set OPENROUTER_API_KEY (and AI_BASE_URL) in your .env file."
        )
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages: list[dict[str, str | int]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    while True:
        assistant = _stream_turn(client, messages)

        if not assistant["tool_calls"]:
            print()
            return

        messages.append(
            {
                "role": "assistant",
                "content": assistant["content"] or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in assistant["tool_calls"].values()
                ],
            }
        )

        for tc in assistant["tool_calls"].values():
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            result = check_inventory(**args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )


def main() -> None:
    print("Customer Service Agent (type 'exit' to quit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        print("Agent: ", end="", flush=True)
        run_agent(user_input)
        print()


if __name__ == "__main__":
    main()
