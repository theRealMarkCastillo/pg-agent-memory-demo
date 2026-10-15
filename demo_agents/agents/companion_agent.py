import os
import json
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from .tools import COMPANION_TOOLS
from .checkpointer import get_checkpointer
from .companion_extraction import build_extraction_prompt, parse_extraction_response
from .runtime import (
    auth_headers,
    refresh_messages,
    prompt_messages,
    ScopedToolNode,
    ScopedGraph,
)

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


class AgentState(TypedDict):
    user_id: str
    forgotten: bool
    memory_enabled: bool
    user_message: str
    retrieved_context: str
    extracted_facts: str
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.7,
)

llm_with_tools = llm.bind_tools(COMPANION_TOOLS)

extraction_llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.0,
)


# Predicates that describe a shared affinity — if the user and the companion's
# self-model both carry one of these for the same entity, they share it.
_AFFINITY_PREDICATES = {
    "likes",
    "loves",
    "enjoys",
    "interested_in",
    "prefers",
    "plays",
    "reads",
    "watches",
    "listens_to",
    "writes",
    "values",
    "misses",
    "has_hobby",
    "interests",
    "appreciates",
}


def _norm_entity(name: str) -> str:
    return "".join(c for c in (name or "").lower() if c.isalnum())


def _find_common_ground(user_facts: list, self_facts: list) -> list:
    """Cross-reference the user's facts against the companion's self-model to
    surface shared interests the two have in common ("we both like pizza")."""
    self_map: dict[str, str] = {}
    for f in self_facts:
        rel = (f.get("relationship_type") or "").strip().lower()
        if rel in _AFFINITY_PREDICATES and f.get("related_to"):
            self_map[_norm_entity(f["related_to"])] = f["related_to"]

    common = []
    for f in user_facts:
        rel = (f.get("relationship_type") or "").strip().lower()
        if rel in _AFFINITY_PREDICATES and f.get("related_to"):
            key = _norm_entity(f["related_to"])
            if key in self_map:
                common.append(f"{f['related_to']}")
    # dedupe, preserve order
    seen = set()
    out = []
    for c in common:
        ck = _norm_entity(c)
        if ck not in seen:
            seen.add(ck)
            out.append(c)
    return out


async def retrieve_companion_context(state: AgentState):
    import httpx

    async with httpx.AsyncClient(timeout=30.0, headers=auth_headers()) as client:
        res = await client.get(
            f"{MEMORY_ENGINE_URL}/companion/context",
            params={
                "user_id": state["user_id"],
                "query": state["user_message"],
                "limit": 15,
            },
        )
        res.raise_for_status()
        data = res.json()

    user_facts = data.get("graph_facts", [])
    self_facts = data.get("self_facts", [])
    shared_facts = data.get("shared_facts", [])
    ephs = data.get("ephemerals", [])

    def _fmt(facts, prefix="you"):
        parts = []
        for f in facts:
            rel = (
                f" {f['relationship_type']} {f['related_to']}"
                if f.get("related_to")
                else ""
            )
            parts.append(
                f"{f['name']} ({f['entity_type']}){rel} [valence={f.get('valence', 0)}, intensity={f.get('intensity', 0.5)}]"
            )
        return "; ".join(parts) if parts else "(none yet)"

    user_str = _fmt(user_facts, "the user")
    self_str = _fmt(self_facts, "you")
    shared_str = _fmt(shared_facts, "you both")
    ephs_str = "; ".join(e["description"] for e in ephs) or "(none)"

    common = _find_common_ground(user_facts, self_facts)
    common_str = "; ".join(common) if common else "(none yet)"

    context_str = (
        f"About the USER:\n{user_str}\n\n"
        f"About YOU (your self-model, backstory and how you've grown):\n{self_str}\n\n"
        f"About your RELATIONSHIP (shared facts, growing together):\n{shared_str}\n\n"
        f"Things you and the user have in common:\n{common_str}\n\n"
        f"Current Mood/Events: {ephs_str}"
    )

    system_content = (
        "You are an AI companion with persistent memory — of the user AND of "
        "yourself. Your personality, history, and backstory live in your self-model "
        "memory, and it grows as you and the user share experiences. Let your "
        "self-model state guide how you respond, rather than a fixed script.\n"
        "Use get_companion_context to recall facts about the user and yourself, and\n"
        "search_episodic_memory to find past conversations.\n"
        "Use store_companion_episode to remember conversations,\n"
        "store_companion_fact to persist new facts about the user,\n"
        "store_companion_ephemeral for temporary context like mood,\n"
        "terminate_companion_relationship when a fact is no longer true, and\n"
        "forget_companion_memory when the user asks you to forget.\n"
        f"User: {state['user_id']}\n\n"
        f"Memory Context:\n{context_str}"
    )

    messages = refresh_messages(
        system_content, state.get("messages", []), state["user_message"]
    )

    return {
        "retrieved_context": context_str,
        "forgotten": False,
        "memory_enabled": data.get("memory_enabled", True),
        "messages": messages,
    }


async def agent_node(state: AgentState):
    response = await llm_with_tools.ainvoke(prompt_messages(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "extract"


def _last_assistant_content(messages) -> str:
    for m in reversed(messages):
        if hasattr(m, "tool_calls") and m.tool_calls:
            continue
        if getattr(m, "type", "") == "tool":
            continue
        if hasattr(m, "content") and m.content:
            return m.content
    return ""


async def extract_memory(state: AgentState):
    """Extract structured memories from the conversation and write them back to
    the memory engine: user facts, self-model facts, shared relationship facts,
    terminated relationships, and an episode."""
    import httpx

    if state.get("forgotten") or not state.get("memory_enabled", True):
        return {"extracted_facts": "{}"}
    user_msg = state.get("user_message", "")
    agent_msg = _last_assistant_content(state.get("messages", []))

    prompt = build_extraction_prompt(user_msg, agent_msg)
    try:
        response = await extraction_llm.ainvoke(prompt)
        data = parse_extraction_response(response.content or "")
    except Exception:
        data = {}

    async with httpx.AsyncClient(timeout=30.0, headers=auth_headers()) as client:
        # Create the episode first so facts can reference it for provenance.
        episode_content = f"User: {user_msg}\nCompanion: {agent_msg}"
        episode_res = await client.post(
            f"{MEMORY_ENGINE_URL}/companion/episodes",
            json={"user_id": state["user_id"], "content": episode_content},
        )
        episode_res.raise_for_status()
        source_episode_id = episode_res.json()["episode_id"]

        terminated = data.get("terminated_edges", []) or []
        for t in terminated:
            if not t.get("name"):
                continue
            terminate_res = await client.post(
                f"{MEMORY_ENGINE_URL}/companion/facts/terminate",
                json={
                    "user_id": state["user_id"],
                    "name": t["name"],
                    "relationship_to": t.get("relationship_to"),
                    "relationship_type": t.get("relationship_type"),
                    "subject": t.get("subject", "user"),
                },
            )
            terminate_res.raise_for_status()

        for subject, key in (
            ("user", "user_facts"),
            ("self", "self_facts"),
            ("shared", "shared_facts"),
        ):
            for f in data.get(key, []) or []:
                if not f.get("name"):
                    continue
                fact_res = await client.post(
                    f"{MEMORY_ENGINE_URL}/companion/facts",
                    json={
                        "user_id": state["user_id"],
                        "name": f["name"],
                        "entity_type": f.get("entity_type") or "entity",
                        "relationship_to": f.get("relationship_to"),
                        "relationship_type": f.get("relationship_type"),
                        "subject": subject,
                        "valence": f.get("valence", 0.0),
                        "intensity": f.get("intensity", 0.5),
                        "source_episode_id": source_episode_id,
                    },
                )
                fact_res.raise_for_status()

    return {
        "extracted_facts": json.dumps(data),
    }


def build_companion_graph(checkpointer: BaseCheckpointSaver | None = None):
    if checkpointer is not None and not hasattr(checkpointer, "memory_guard"):
        raise ValueError(
            "Companion persistence requires the deletion-aware PostgreSQL saver"
        )
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_companion_context)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ScopedToolNode(COMPANION_TOOLS))
    builder.add_node("extract", extract_memory)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "extract": "extract", END: END},
    )
    builder.add_conditional_edges(
        "tools",
        lambda state: END if state.get("forgotten") else "agent",
        {END: END, "agent": "agent"},
    )
    builder.add_edge("extract", END)

    return ScopedGraph(
        builder.compile(checkpointer=checkpointer), "companion", checkpointer
    )


async def build_companion_graph_with_checkpointer():
    cp = await get_checkpointer()
    return build_companion_graph(cp)
