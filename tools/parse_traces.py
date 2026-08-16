"""Parse LangSmith trace exports into (user, assistant) conversation turns.

Supports two trace shapes found in LangSmith:
  1. LangGraph traces  - top-level run exposes ``messages[]`` with ``type``
     ``human``/``ai`` and ``content`` (with optional ``[38m ago]`` prefixes).
  2. Extraction traces - a ``RunnableSequence`` run whose input prompt contains
     a ``CONVERSATION:`` section with ``Speaker: message`` lines.

Stdlib only.
"""

import json
import re
from pathlib import Path
from typing import NamedTuple


class Turn(NamedTuple):
    user: str
    assistant: str


TIMESTAMP_RE = re.compile(r"^\[[^\]]+\]\s*")
SPEAKER_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 ]*?)\s*:\s*(.*)$")


def _strip_timestamp(text: str) -> str:
    return TIMESTAMP_RE.sub("", text, count=1).strip()


def _as_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("type"), str) and item["type"] == "text":
                    parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _turns_from_messages(messages: list) -> list[Turn]:
    """Pair up consecutive human/ai messages."""
    turns: list[Turn] = []
    user_buf: list[str] = []
    for m in messages:
        msg_type = m.get("type", "")
        content = _strip_timestamp(_as_text(m.get("content"))).strip()
        if msg_type == "human":
            user_buf.append(content)
        elif msg_type == "ai":
            assistant = content
            if user_buf:
                user = "\n".join(user_buf)
                user_buf = []
                turns.append(Turn(user, assistant))
            elif turns:
                # orphan assistant message - attach to previous turn context
                turns[-1] = Turn(turns[-1].user, f"{turns[-1].assistant}\n{assistant}")
    if user_buf and turns:
        turns[-1] = Turn(f"{turns[-1].user}\n{'\n'.join(user_buf)}", turns[-1].assistant)
    return turns


def _turns_from_conversation_block(conv: str) -> list[Turn]:
    """Parse ``User: ...`` / ``Companion: ...`` lines into turns."""
    turns: list[Turn] = []
    lines = conv.splitlines()
    user_parts: list[str] = []
    assistant_parts: list[str] = []
    current = None  # 'user' | 'assistant'

    def flush():
        nonlocal user_parts, assistant_parts
        if user_parts and assistant_parts:
            turns.append(
                Turn("\n".join(user_parts).strip(), "\n".join(assistant_parts).strip())
            )
        elif user_parts and turns:
            turns[-1] = Turn(turns[-1].user, "\n".join(user_parts).strip())
        user_parts = []
        assistant_parts = []

    for line in lines:
        match = SPEAKER_LINE_RE.match(line)
        if not match:
            (assistant_parts if current == "assistant" else user_parts).append(line)
            continue
        speaker, rest = match.group(1), match.group(2)
        if speaker.strip().lower() == "user":
            if current == "assistant":
                flush()
            current = "user"
            user_parts.append(rest)
        else:
            if current == "user":
                current = "assistant"
            assistant_parts.append(rest)
    flush()
    return turns


def _strip_context_prefix(text: str) -> str:
    """Strip '[CONTEXT: ...]' / '[Your original message was: ...]' platform noise."""
    lines = text.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[CONTEXT:") and stripped.endswith("]"):
            continue
        if stripped.startswith("[Your original message was:"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def parse_trace_file(path: str | Path) -> tuple[list[Turn], dict]:
    """Parse one trace export file.

    Returns (turns, metadata) where metadata carries trace_id, run names, and
    any production extraction output found in the export.
    """
    data = json.loads(Path(path).read_text())
    runs = data.get("runs", [])
    meta = {
        "file": str(path),
        "trace_id": data.get("trace_id"),
        "run_names": sorted({r.get("name", "") for r in runs}),
        "production_facts": [],
    }

    turns: list[Turn] = []
    # First pass: capture production extraction output (FactExtractionOutput) if present
    for run in runs:
        if run.get("name") == "RunnableLambda":
            out = (run.get("outputs") or {}).get("output")
            if isinstance(out, dict) and out.get("facts"):
                meta["production_facts"] = out.get("facts", [])

    # Second pass: extract the conversation
    for run in runs:
        name = run.get("name", "")
        # MasterGraphExecution: character-agent conversations (ai-companions projects)
        if name == "MasterGraphExecution" and not run.get("parent_run_id"):
            chat = _find_master_graph_messages(runs)
            if chat:
                turns = _turns_from_messages(chat)
                meta["shape"] = "mastergraph"
            break
        # LangGraph traces: full conversation lives in the top-level run
        if name == "LangGraph" and not run.get("parent_run_id"):
            messages = (run.get("outputs") or {}).get("messages") or (
                run.get("inputs") or {}
            ).get("messages") or []
            turns = _turns_from_messages(messages)
            if turns:
                meta["shape"] = "langgraph"
                break
        # Extraction traces: conversation embedded in the extraction prompt
        if name == "RunnableSequence":
            for m in (run.get("inputs") or {}).get("input", []):
                content = _as_text(m.get("content", ""))
                if "CONVERSATION:" in content:
                    turns = _turns_from_conversation_block(
                        content.split("CONVERSATION:", 1)[-1]
                    )
                    meta["shape"] = "extraction"
                    break
            if turns:
                break

    return turns, meta


def _find_master_graph_messages(runs: list) -> list:
    """Extract the full message list from a MasterGraphExecution trace.

    The current turn's user input and the assistant's final response are the
    last two entries; prior turns are the 'chat_history' on the character_agent
    node. Rebuild a single chronological message list.
    """
    root = None
    for r in runs:
        if r.get("name") == "MasterGraphExecution" and not r.get("parent_run_id"):
            root = r
            break
    if not root:
        return []

    character = None
    for r in runs:
        if r.get("parent_run_id") == root["id"] and r.get("name") == "character_agent":
            character = r
            break
    if not character:
        return []

    inputs = character.get("inputs") or {}
    chat_history = inputs.get("chat_history") or []
    messages = list(chat_history)

    user_input = _strip_context_prefix(_as_text(inputs.get("user_input", "")))
    if user_input:
        messages.append({"type": "human", "content": user_input})

    final_response = (root.get("outputs") or {}).get("final_response")
    final_response = _as_text(final_response) if final_response else ""
    if final_response:
        messages.append({"type": "ai", "content": final_response})

    return messages


def parse_trace_dir(directory: str | Path) -> list[tuple[list[Turn], dict]]:
    """Parse every *.json in a directory. Returns (turns, meta) pairs, skipping
    files with no conversation."""
    results = []
    for path in sorted(Path(directory).glob("*.json")):
        turns, meta = parse_trace_file(path)
        if turns:
            results.append((turns, meta))
    return results


if __name__ == "__main__":
    import sys

    for turns, meta in parse_trace_dir(sys.argv[1] if len(sys.argv) > 1 else "traces"):
        print(f"=== {meta['file']} [{meta.get('shape')}] turns={len(turns)} ===")
        for t in turns[:2]:
            print(f"  USER: {t.user[:120]}")
            print(f"  AISS: {t.assistant[:120]}")
