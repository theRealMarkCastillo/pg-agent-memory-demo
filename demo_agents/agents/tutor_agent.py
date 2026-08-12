import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from .tools import TUTOR_TOOLS
from .checkpointer import get_checkpointer


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

    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(
            f"{os.getenv('MEMORY_ENGINE_URL', 'http://memory-engine:8000')}/tutor/gaps/{state['user_id']}"
        )
        data = res.json()

    gaps_str = "\n".join(
        f"{s['skill_name']}: decayed_score={s['decayed_score']:.3f} [{s['status']}]"
        for s in data
    )

    system_content = (
        "You are an adaptive tutor agent with a skill-tree memory and forgetting-curve modeling.\n"
        "Use get_skill_gaps to assess a learner's current state, then recommend and teach.\n"
        "Use update_skill_progress to record improved proficiency after the learner demonstrates mastery.\n"
        f"Learner: {state['user_id']}\n\n"
        f"Skill Gaps (with Ebbinghaus decay):\n{gaps_str}"
    )

    existing = state.get("messages", [])
    if existing:
        messages = list(existing)
        if getattr(messages[0], "type", "") == "system":
            messages[0] = SystemMessage(content=system_content)
        else:
            messages.insert(0, SystemMessage(content=system_content))
        messages.append(HumanMessage(content=f"Topic request: {state['topic']}\nAssess gaps, recommend, and teach."))
    else:
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"Topic request: {state['topic']}\nAssess gaps, recommend, and teach."),
        ]

    return {
        "skill_gaps": gaps_str,
        "messages": messages,
    }


async def agent_node(state: AgentState):
    response = await llm_with_tools.ainvoke(state["messages"])
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
    builder.add_node("tools", ToolNode(TUTOR_TOOLS))

    builder.set_entry_point("assess")
    builder.add_edge("assess", "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


async def build_tutor_graph_with_checkpointer():
    cp = await get_checkpointer()
    return build_tutor_graph(cp)
