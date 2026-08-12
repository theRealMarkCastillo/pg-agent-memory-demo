import os
import json
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from .tools import COMPANION_TOOLS
from .checkpointer import get_checkpointer

MEMORY_ENGINE_URL = os.getenv("MEMORY_ENGINE_URL", "http://memory-engine:8000")


class AgentState(TypedDict):
    user_id: str
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


async def retrieve_companion_context(state: AgentState):
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(
            f"{MEMORY_ENGINE_URL}/companion/context",
            params={
                "user_id": state["user_id"],
                "query": state["user_message"],
                "limit": 15,
            },
        )
        data = res.json()

    facts = data.get("graph_facts", [])
    ephs = data.get("ephemerals", [])

    facts_str = "; ".join(
        f"{f['name']} ({f['entity_type']})"
        + (f" {f['relationship_type']} {f['related_to']}" if f.get("related_to") else "")
        for f in facts
    )
    ephs_str = "; ".join(e["description"] for e in ephs)

    context_str = f"Active Facts: {facts_str}\nCurrent Mood/Events: {ephs_str}"

    system_content = (
        "You are an AI companion with persistent memory.\n"
        "Use get_companion_context to recall facts about the user, and\n"
        "search_episodic_memory to find past conversations.\n"
        "Use store_companion_episode to remember conversations,\n"
        "store_companion_fact to persist new facts about the user,\n"
        "store_companion_ephemeral for temporary context like mood,\n"
        "terminate_companion_relationship when a fact is no longer true, and\n"
        "forget_companion_memory when the user asks you to forget.\n"
        f"User: {state['user_id']}\n\n"
        f"Memory Context:\n{context_str}"
    )

    existing = state.get("messages", [])
    if existing:
        messages = list(existing)
        if getattr(messages[0], "type", "") == "system":
            messages[0] = SystemMessage(content=system_content)
        else:
            messages.insert(0, SystemMessage(content=system_content))
        messages.append(HumanMessage(content=state["user_message"]))
    else:
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=state["user_message"]),
        ]

    return {
        "retrieved_context": context_str,
        "messages": messages,
    }


async def agent_node(state: AgentState):
    response = await llm_with_tools.ainvoke(state["messages"])
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
    the memory engine: new graph facts, terminated relationships, and an episode."""
    import httpx

    user_msg = state.get("user_message", "")
    agent_msg = _last_assistant_content(state.get("messages", []))

    prompt = f"""Analyze this conversation between a user and an AI companion. Extract durable,
identity-relevant memories. Ignore small talk and transient statements.

User said: {user_msg}
Companion said: {agent_msg}

Return ONLY valid JSON with this exact shape:
{{
  "new_facts": [
    {{"name": "<entity>", "entity_type": "<type>", "relationship_to": "<target or null>", "relationship_type": "<relation or null>"}}
  ],
  "terminated_edges": [
    {{"name": "<entity>", "relationship_to": "<target or null>", "relationship_type": "<relation or null>"}}
  ],
  "episode_content": "<1-2 sentence summary of the exchange>"
}}

Rules:
- new_facts: durable facts the user revealed (preferences, people, locations, goals, facts about themselves)
- terminated_edges: relationships that are NO LONGER true (e.g. user moved, changed job, dropped a goal)
- Use entity_type values like: person, location, goal, preference, hobby, job, pet, event
- If nothing meaningful, return empty lists.
"""

    try:
        response = await extraction_llm.ainvoke(prompt)
        content = response.content or ""
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
    except Exception:
        data = {"new_facts": [], "terminated_edges": [], "episode_content": ""}

    async with httpx.AsyncClient(timeout=30.0) as client:
        new_facts = data.get("new_facts", []) or []
        for f in new_facts:
            if not f.get("name"):
                continue
            await client.post(
                f"{MEMORY_ENGINE_URL}/companion/facts",
                json={
                    "user_id": state["user_id"],
                    "name": f["name"],
                    "entity_type": f.get("entity_type") or "entity",
                    "relationship_to": f.get("relationship_to"),
                    "relationship_type": f.get("relationship_type"),
                },
            )

        terminated = data.get("terminated_edges", []) or []
        for t in terminated:
            if not t.get("name"):
                continue
            await client.post(
                f"{MEMORY_ENGINE_URL}/companion/facts/terminate",
                json={
                    "user_id": state["user_id"],
                    "name": t["name"],
                    "relationship_to": t.get("relationship_to"),
                    "relationship_type": t.get("relationship_type"),
                },
            )

        episode_content = data.get("episode_content") or (
            f"User: {user_msg}\nCompanion: {agent_msg}"
        )
        await client.post(
            f"{MEMORY_ENGINE_URL}/companion/episodes",
            json={"user_id": state["user_id"], "content": episode_content},
        )

    return {
        "extracted_facts": json.dumps(data),
    }


def build_companion_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_companion_context)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(COMPANION_TOOLS))
    builder.add_node("extract", extract_memory)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "extract": "extract", END: END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("extract", END)

    return builder.compile(checkpointer=checkpointer)


async def build_companion_graph_with_checkpointer():
    cp = await get_checkpointer()
    return build_companion_graph(cp)
