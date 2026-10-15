"""Shared companion memory extraction prompt, predicate vocabulary, and parser.

Used by:
- demo_agents/agents/companion_agent.py (live LangGraph extraction node)
- tools/replay_traces.py (LangSmith trace replay pipeline)
- tools/benchmark_extraction.py (ground-truth extraction benchmark)
"""

import json

# Canonical relation predicates, aligned with the production extractor's vocabulary.
PREDICATE_VOCAB = """LIVES_IN, FROM, MOVED_TO, GREW_UP_IN, HAS_JOB, WORKS_AT, WORKED_AT, WORKS_ON,
MARRIED_TO, DATING, PARENT_OF, SIBLING_OF, FRIEND_OF, FAMILY_MEMBER, HAS_PET,
LIKES, LOVES, DISLIKES, HATES, ENJOYS, INTERESTED_IN, PREFERS, ALLERGIC_TO,
HAS_CONDITION, DIETARY_RESTRICTION, WATCHES, READS, PLAYS, LISTENS_TO, WRITES,
HAS_AGE, BIRTHDAY_ON, SPEAKS, IDENTIFIES_AS, PRACTICES, HAS_NAME, WANTS_TO,
PLANS_TO, AVOIDS, SKILLED_AT, CERTIFIED_IN, VALUES, USES, SUPPORTS, TRUSTS,
APPRECIATES, BUILT, MISSES, SHARED_EXPERIENCE, SHARED_INTEREST, SHARED_RITUAL,
INSIDE_JOKE, AGREED_ON, SHARED_MEMORY, RELATIONSHIP_MILESTONE, DISAGREE_ON, HAS_FACT"""

EXTRACTION_PROMPT = """Analyze this conversation between a user and an AI companion. Extract durable,
identity-relevant memories. Capture facts about the USER, facts the COMPANION reveals about
ITSELF (its own preferences, history, values, how it's changing), and SHARED relationship
facts (inside jokes, shared rituals, milestones, what "we/us/our" — the growing bond).

User said: {user_msg}
Companion said: {agent_msg}

Return ONLY valid JSON with this exact shape:
{{
  "user_facts": [
    {{"name": "<entity>", "entity_type": "<type>", "relationship_to": "<target or null>", "relationship_type": "<UPPER_SNAKE_PREDICATE>", "valence": <number -1..1>, "intensity": <number 0..1>}}
  ],
  "self_facts": [
    {{"name": "<companion entity>", "entity_type": "<type>", "relationship_to": "<target or null>", "relationship_type": "<UPPER_SNAKE_PREDICATE>", "valence": <number -1..1>, "intensity": <number 0..1>}}
  ],
  "shared_facts": [
    {{"name": "<entity>", "entity_type": "<type>", "relationship_to": "<target or null>", "relationship_type": "<UPPER_SNAKE_PREDICATE>", "valence": <number -1..1>, "intensity": <number 0..1>}}
  ],
  "terminated_edges": [
    {{"name": "<entity>", "relationship_to": "<target or null>", "relationship_type": "<UPPER_SNAKE_PREDICATE>"}}
  ],
  "episode_content": "<1-2 sentence summary of the exchange>"
}}

Rules:
- user_facts: durable facts the USER revealed (preferences, people, locations, goals, pets, jobs, hobbies, health, relationships). These are about the user.
- self_facts: durable facts the COMPANION revealed about ITSELF — its own stated preferences, history, values, how it feels or is changing. Only extract these if the companion says something concrete about itself (NOT the user). Prefer the companion's name (e.g. "Iris" or "Vesper") as name.
- shared_facts: relationship facts that belong to BOTH — "we always...", "our inside joke", shared rituals, relationship milestones, agreements. These capture the bond growing together.
- Use entity_type values like: person, location, goal, preference, hobby, job, pet, event, self
- DIRECTION: name is the SUBJECT, relationship_to is the OBJECT. Express each fact in ONE canonical direction and never emit the reverse. E.g. "user likes Gemini" -> name="user", relationship_to="Gemini", relationship_type="LIKES".
- CONSISTENCY: reuse stable entity names (the user's name or "user"; the companion's name for self_facts). Never duplicate a fact already stated.
- PREDICATES: relationship_type MUST be one of these UPPER_SNAKE values:
""" + PREDICATE_VOCAB + """
  - Match the verb: "I enjoy hiking" -> ENJOYS. "I love sushi" -> LOVES. "I play guitar" -> PLAYS. "I live in Seattle" -> LIVES_IN. "I'm a software developer" -> HAS_JOB. "My wife Sarah is a nurse" -> MARRIED_TO (relationship_to="Sarah"). "I have a cat named Luna" -> HAS_PET (relationship_to="Luna"). "I want to learn Japanese" -> WANTS_TO (relationship_to="learn Japanese"). "I used to work at Google" -> WORKED_AT. "I hate spiders" -> HATES. "I'm allergic to peanuts" -> ALLERGIC_TO. "I'm vegetarian" -> DIETARY_RESTRICTION.
  - TEMPORAL: extract CURRENT state. Past jobs use WORKED_AT, current use WORKS_AT.
- VALENCE & INTENSITY: valence reflects emotional tone (-1 very sad/loss, +1 joyful); intensity how strongly felt (0..1). Pet dying -> valence -0.9 intensity 0.9. New job -> +0.6/0.5. Neutral -> 0/0.3.
- terminated_edges: relationships that are NO LONGER true (user or companion moved on, dropped a goal, ended a fact)
- If nothing meaningful in a category, return an empty list for it.
"""


def build_extraction_prompt(user_msg: str, agent_msg: str) -> str:
    """Format the canonical 3-scope extraction prompt for a (user, companion) turn."""
    return EXTRACTION_PROMPT.format(user_msg=user_msg, agent_msg=agent_msg)


def strip_code_fence(content: str) -> str:
    """Strip markdown ```json ... ``` fences from an LLM response."""
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    return content.strip()


def parse_extraction_response(content: str) -> dict:
    """Parse an LLM extraction response into a normalized dict."""
    try:
        data = json.loads(strip_code_fence(content))
    except Exception:
        return {}
    if isinstance(data, list):
        data = {"user_facts": data}
    if not isinstance(data, dict):
        return {}
    for key in ("user_facts", "self_facts", "shared_facts", "terminated_edges"):
        if not isinstance(data.get(key), list):
            data[key] = []
        data[key] = [x for x in data[key] if isinstance(x, dict)]
    return {
        "user_facts": data["user_facts"],
        "self_facts": data["self_facts"],
        "shared_facts": data["shared_facts"],
        "terminated_edges": data["terminated_edges"],
        "episode_content": data.get("episode_content") or "",
    }
