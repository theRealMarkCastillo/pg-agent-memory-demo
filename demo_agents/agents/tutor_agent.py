import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from .tools import TUTOR_TOOLS
from .checkpointer import get_checkpointer
from .runtime import (
    auth_headers,
    refresh_messages,
    prompt_messages,
    ScopedToolNode,
    ScopedGraph,
)


class AgentState(TypedDict):
    user_id: str
    topic: str
    skill_gaps: str
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL_NAME"),
    temperature=0.5,
)

llm_with_tools = llm.bind_tools(TUTOR_TOOLS)


async def assess_skill_gaps(state: AgentState):
    import httpx

    async with httpx.AsyncClient(timeout=30.0, headers=auth_headers()) as client:
        res = await client.get(
            f"{os.getenv('MEMORY_ENGINE_URL', 'http://memory-engine:8000')}/tutor/gaps/{state['user_id']}"
        )
        res.raise_for_status()
        data = res.json()

    gaps_str = "\n".join(
        f"{s['skill_name']}: decayed_score={s['decayed_score']:.3f} [{s['status']} | Ready: {s.get('ready')} | Prerequisite: {s.get('prerequisite')}]"
        for s in data
    )

    system_content = (
        "You are an adaptive tutor agent with a skill-tree memory and forgetting-curve modeling.\n"
        "Use get_skill_gaps to assess a learner's current state, then recommend and teach.\n"
        "Use update_skill_progress to record improved proficiency after the learner demonstrates mastery.\n"
        f"Learner: {state['user_id']}\n\n"
        f"Skill Gaps (with Ebbinghaus decay):\n{gaps_str}"
    )

    messages = refresh_messages(
        system_content, state.get("messages", []), state["topic"]
    )

    return {
        "skill_gaps": gaps_str,
        "messages": messages,
    }


async def agent_node(state: AgentState):
    response = await llm_with_tools.ainvoke(prompt_messages(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_tutor_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(AgentState)
    builder.add_node("assess", assess_skill_gaps)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ScopedToolNode(TUTOR_TOOLS))

    builder.set_entry_point("assess")
    builder.add_edge("assess", "agent")
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")

    return ScopedGraph(
        builder.compile(checkpointer=checkpointer), "tutor", checkpointer
    )


async def build_tutor_graph_with_checkpointer():
    cp = await get_checkpointer()
    return build_tutor_graph(cp)
