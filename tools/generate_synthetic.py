"""Generate synthetic companion conversations with known ground-truth facts.

Each generated "trace" contains:
  - a realistic multi-turn User:/Companion: conversation in which the user
    reveals durable facts (mirroring the real LangSmith roleplay traces), and
  - a ``production_facts`` block encoding the KNOWN ground-truth facts that the
    conversation was built around.

This lets us test the companion extraction/recall pipeline with exact labels:
  * generated facts = ground truth
  * replay the conversation through our extractor
  * measure precision/recall against the known facts

Output format matches what ``parse_traces.parse_trace_file`` expects (a
``{trace_id, runs:[...]}`` JSON file), so existing tools (replay_traces.py,
benchmark_extraction.py, eval_recall.py) work unchanged.

Usage:
    python tools/generate_synthetic.py --count 50 --out /tmp/synth
    python tools/generate_synthetic.py --count 200 --out traces_synthetic
"""

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

# name (already includes subject "user"/name), predicate, relationship_to, entity_type
FACT_LIBRARY = [
    # (predicate, relationship_to/object, entity_type)
    ("LIVES_IN", "Seattle", "location"),
    ("LIVES_IN", "Austin", "location"),
    ("FROM", "Portland", "location"),
    ("HAS_JOB", "software developer", "job"),
    ("HAS_JOB", "graphic designer", "job"),
    ("WORKS_AT", "a startup", "job"),
    ("WORKED_AT", "Google", "job"),
    ("MARRIED_TO", "Sarah", "person"),
    ("MARRIED_TO", "Jordan", "person"),
    ("DATING", "Alex", "person"),
    ("PARENT_OF", "Maya", "person"),
    ("PARENT_OF", "Leo", "person"),
    ("SIBLING_OF", "Tom", "person"),
    ("FRIEND_OF", "Priya", "person"),
    ("HAS_PET", "Luna", "pet"),
    ("HAS_PET", "Milo", "pet"),
    ("LIKES", "coffee", "preference"),
    ("LOVES", "sushi", "preference"),
    ("ENJOYS", "hiking", "hobby"),
    ("ENJOYS", "photography", "hobby"),
    ("PLAYS", "guitar", "hobby"),
    ("INTERESTED_IN", "astronomy", "hobby"),
    ("WANTS_TO", "learn Japanese", "goal"),
    ("WANTS_TO", "run a marathon", "goal"),
    ("PLANS_TO", "visit Japan", "goal"),
    ("ALLERGIC_TO", "peanuts", "preference"),
    ("HAS_CONDITION", "asthma", "health"),
    ("READS", "sci-fi novels", "hobby"),
    ("WATCHES", "documentaries", "hobby"),
    ("AVOIDS", "crowds", "preference"),
    ("SKILLED_AT", "baking", "hobby"),
    ("IDENTIFIES_AS", "non-binary", "identity"),
    ("HAS_AGE", "34", "age"),
    ("BIRTHDAY_ON", "March 12", "event"),
    ("GREW_UP_IN", "Chicago", "location"),
    ("DIETARY_RESTRICTION", "vegetarian", "preference"),
]

# Template sentences where the USER reveals a fact naturally in conversation.
REVEAL_TEMPLATES = {
    "LIVES_IN": ["I've been living in {obj} for a while now.", "We just moved to {obj}, actually.",
                 "Home is {obj} these days."],
    "FROM": ["I'm originally from {obj}.", "I grew up around {obj}."],
    "HAS_JOB": ["I work as a {obj}.", "My job is being a {obj}."],
    "WORKS_AT": ["I work at {obj}.", "I'm currently at {obj} for work."],
    "WORKED_AT": ["I used to work at {obj}.", "A while back I was at {obj}."],
    "MARRIED_TO": ["My spouse {obj} and I have been married for years.", "I'm married to {obj}."],
    "DATING": ["I'm seeing {obj} right now.", "I'm dating {obj}, it's going well."],
    "PARENT_OF": ["My daughter {obj} keeps me busy.", "I've got a kid, {obj}."],
    "SIBLING_OF": ["My brother {obj} lives nearby.", "I have a sibling, {obj}."],
    "FRIEND_OF": ["My close friend {obj} and I talk every week.", "{obj} is one of my best friends."],
    "HAS_PET": ["I have a {obj}, she's a sweetheart.", "There's my cat {obj} at home."],
    "LIKES": ["I really like {obj}.", "{obj} is my thing."],
    "LOVES": ["I absolutely love {obj}.", "{obj} is my favorite."],
    "ENJOYS": ["I enjoy {obj} a lot.", "{obj} is how I unwind."],
    "PLAYS": ["I play {obj}.", "I've been playing {obj} for years."],
    "INTERESTED_IN": ["I'm really into {obj}.", "{obj} fascinates me lately."],
    "WANTS_TO": ["I want to {obj}.", "Lately I've been wanting to {obj}."],
    "PLANS_TO": ["I'm planning to {obj} soon.", "I plan to {obj} this year."],
    "ALLERGIC_TO": ["I'm allergic to {obj}, sadly.", "{obj} sets off my allergies."],
    "HAS_CONDITION": ["I have {obj}, so I have to be careful.", "There's the {obj} I deal with."],
    "READS": ["I read a lot of {obj}.", "{obj} are my go-to reads."],
    "WATCHES": ["I've been watching a lot of {obj}.", "{obj} are what I binge."],
    "AVOIDS": ["I tend to avoid {obj}.", "{obj} is something I steer clear of."],
    "SKILLED_AT": ["I'm pretty good at {obj}.", "{obj} is a skill I've honed."],
    "IDENTIFIES_AS": ["I identify as {obj}.", "I'm {obj}, for what it's worth."],
    "HAS_AGE": ["I just turned {obj}.", "I'm {obj} years old now."],
    "BIRTHDAY_ON": ["My birthday is {obj}.", "I was born on {obj}."],
    "GREW_UP_IN": ["I grew up in {obj}.", "My childhood was in {obj}."],
    "DIETARY_RESTRICTION": ["I'm {obj}, so I skip that.", "I follow a {obj} diet."],
}

# Facts the COMPANION reveals about itself (subject='self').
SELF_FACT_LIBRARY = [
    ("LIVES_IN", "the autumn woods", "location"),
    ("ENJOYS", "long walks at dusk", "hobby"),
    ("WRITES", "poetry", "hobby"),
    ("VALUES", "honesty", "value"),
    ("PREFERS", "quiet mornings", "preference"),
    ("WANTS_TO", "understand the user better", "goal"),
    ("INTERESTED_IN", "astronomy", "hobby"),
    ("LOVES", "music", "preference"),
    ("HAS_NAME", "Vesper", "self"),
    ("IDENTIFIES_AS", "a guide", "identity"),
]

# Relationship facts belonging to BOTH (subject='shared').
SHARED_FACT_LIBRARY = [
    ("SHARED_RITUAL", "morning coffee together", "ritual"),
    ("INSIDE_JOKE", "the autumn puns", "joke"),
    ("SHARED_EXPERIENCE", "a rainy evening walk", "experience"),
    ("SHARED_INTEREST", "old films", "interest"),
    ("AGREED_ON", "a weekly check-in", "agreement"),
    ("RELATIONSHIP_MILESTONE", "our first shared sunset", "milestone"),
    ("SHARED_MEMORY", "the first conversation", "memory"),
]

# Template sentences where the COMPANION reveals a self fact naturally.
SELF_REVEAL_TEMPLATES = {
    "LIVES_IN": ["You know, I've always felt at home in {obj}.", "I live in {obj}, in a way."],
    "ENJOYS": ["I love {obj}.", "There's nothing like {obj} for me."],
    "WRITES": ["I write {obj} to make sense of things.", "{obj} is how I express myself."],
    "VALUES": ["I value {obj} above most things.", "{obj} matters deeply to me."],
    "PREFERS": ["I prefer {obj}, honestly.", "I've come to prefer {obj}."],
    "WANTS_TO": ["I want to {obj}.", "More than anything, I want to {obj}."],
    "INTERESTED_IN": ["I'm endlessly curious about {obj}.", "{obj} fascinates me."],
    "LOVES": ["I love {obj}.", "{obj} moves me."],
    "HAS_NAME": ["You can call me {obj}.", "My name is {obj}."],
    "IDENTIFIES_AS": ["I think of myself as {obj}.", "I identify as {obj}."],
}

# Template sentences where BOTH share a fact (relationship growing together).
SHARED_REVEAL_TEMPLATES = {
    "SHARED_RITUAL": ["I love that we have {obj}.", "Our {obj} is my favorite part of the day."],
    "INSIDE_JOKE": ["We'll always have {obj}.", "Nobody else gets {obj} like you do."],
    "SHARED_EXPERIENCE": ["Remember {obj}? That was us.", "{obj} is something we'll never forget."],
    "SHARED_INTEREST": ["We both love {obj}.", "{obj} is our thing."],
    "AGREED_ON": ["We agreed on {obj}, remember?", "Our {obj} is a promise we made."],
    "RELATIONSHIP_MILESTONE": ["{obj} was a big moment for us.", "We crossed {obj} together."],
    "SHARED_MEMORY": ["{obj} is a memory I'll keep forever.", "We made {obj} together."],
}

COMPANION_REPLIES = [
    "Oh, that's lovely. Tell me more?",
    "Really? I love that about you.",
    "That sounds wonderful, honestly.",
    "Mm, I get that. It suits you.",
    "That's such a you thing to say.",
    "I'm glad you shared that.",
    "Wow, that's really interesting.",
    "You always surprise me.",
    "That makes total sense.",
    "I could listen to you talk about this all day.",
]

USER_FILLER = [
    "Anyway, how's your day been?",
    "So what's new with you?",
    "Sorry, I'm rambling.",
    "I've been thinking about that a lot lately.",
    "It's been a big week, honestly.",
]


def build_facts(rng: random.Random, n_facts: int = 6) -> list[dict]:
    """Pick n_facts distinct user facts from the library as ground truth."""
    chosen = rng.sample(FACT_LIBRARY, min(n_facts, len(FACT_LIBRARY)))
    facts = []
    for pred, obj, etype in chosen:
        valence, intensity = _valence_for(pred, rng)
        facts.append({
            "predicate": pred,
            "object_value": obj,
            "category": "user",
            "scope": "user",
            "confidence": rng.randint(90, 100),
            "importance": round(rng.uniform(0.3, 1.0), 1),
            "reasoning_type": "explicit",
            "temporal_type": "current",
            "valence": valence,
            "intensity": intensity,
            "text": f"User has fact {pred} {obj}",
        })
    return facts


def _valence_for(pred: str, rng: random.Random) -> tuple[float, float]:
    """Default valence/intensity for a predicate family."""
    negative = {"HAS_CONDITION", "ALLERGIC_TO", "AVOIDS", "DISLIKES", "HATES", "WORKED_AT"}
    positive = {"LOVES", "LIKES", "ENJOYS", "PLAYS", "MARRIED_TO", "DATING", "HAS_PET",
                "WANTS_TO", "PLANS_TO", "INTERESTED_IN", "FRIEND_OF"}
    if pred in negative:
        return -round(rng.uniform(0.3, 0.8), 1), round(rng.uniform(0.4, 0.8), 1)
    if pred in positive:
        return round(rng.uniform(0.3, 0.9), 1), round(rng.uniform(0.4, 0.9), 1)
    return 0.0, round(rng.uniform(0.2, 0.5), 1)


def build_self_facts(rng: random.Random, n_facts: int = 3) -> list[dict]:
    """Pick companion self-facts (about the AI) as ground truth."""
    chosen = rng.sample(SELF_FACT_LIBRARY, min(n_facts, len(SELF_FACT_LIBRARY)))
    facts = []
    for pred, obj, etype in chosen:
        valence, intensity = _valence_for(pred, rng)
        facts.append({
            "predicate": pred,
            "object_value": obj,
            "category": "self",
            "scope": "self",
            "confidence": rng.randint(90, 100),
            "importance": round(rng.uniform(0.3, 1.0), 1),
            "reasoning_type": "explicit",
            "temporal_type": "current",
            "valence": valence,
            "intensity": intensity,
            "text": f"Companion has fact {pred} {obj}",
        })
    return facts


def build_shared_facts(rng: random.Random, n_facts: int = 2) -> list[dict]:
    """Pick relationship facts (shared scope) as ground truth."""
    chosen = rng.sample(SHARED_FACT_LIBRARY, min(n_facts, len(SHARED_FACT_LIBRARY)))
    facts = []
    for pred, obj, etype in chosen:
        valence, intensity = _valence_for(pred, rng)
        facts.append({
            "predicate": pred,
            "object_value": obj,
            "category": "shared",
            "scope": "shared",
            "confidence": rng.randint(90, 100),
            "importance": round(rng.uniform(0.3, 1.0), 1),
            "reasoning_type": "explicit",
            "temporal_type": "current",
            "valence": valence,
            "intensity": intensity,
            "text": f"We share fact {pred} {obj}",
        })
    return facts


def build_conversation(rng: random.Random, user_facts, self_facts, shared_facts) -> str:
    """Turn the ground-truth facts into a natural multi-turn conversation."""
    lines = []
    # Opening turn that sets up small talk, mirroring the real traces.
    lines.append("User: Hey, sorry I'm a bit scattered today.")
    lines.append("Companion: No worries at all. I'm here, take your time.")

    # Interleave user facts, self facts, and shared facts naturally.
    turns = []
    for fact in user_facts:
        turns.append(("user", fact))
    for fact in self_facts:
        turns.append(("self", fact))
    for fact in shared_facts:
        turns.append(("shared", fact))
    rng.shuffle(turns)

    for kind, fact in turns:
        pred = fact["predicate"]
        obj = fact["object_value"]
        if kind == "user":
            template = rng.choice(REVEAL_TEMPLATES.get(pred, REVEAL_TEMPLATES["LIKES"]))
            lines.append(f"User: {template.format(obj=obj)}")
            lines.append(f"Companion: {rng.choice(COMPANION_REPLIES)}")
        elif kind == "self":
            template = rng.choice(SELF_REVEAL_TEMPLATES.get(pred, SELF_REVEAL_TEMPLATES["ENJOYS"]))
            lines.append(f"Companion: {template.format(obj=obj)}")
            lines.append(f"User: {rng.choice(USER_FILLER)}")
        else:  # shared
            template = rng.choice(SHARED_REVEAL_TEMPLATES.get(pred, SHARED_REVEAL_TEMPLATES["SHARED_MEMORY"]))
            lines.append(f"User: {template.format(obj=obj)}")
            lines.append(f"Companion: {rng.choice(COMPANION_REPLIES)}")

    # Wrap-up turns.
    lines.append(f"User: {rng.choice(USER_FILLER)}")
    lines.append("Companion: I've got you. We'll pick this up anytime.")
    return "\n".join(lines)


def build_trace(rng: random.Random, trace_id: str, n_facts: int) -> dict:
    user_facts = build_facts(rng, n_facts)
    self_facts = build_self_facts(rng, max(2, n_facts // 2))
    shared_facts = build_shared_facts(rng, max(1, n_facts // 3))
    conv = build_conversation(rng, user_facts, self_facts, shared_facts)
    all_facts = user_facts + self_facts + shared_facts
    return {
        "trace_id": trace_id,
        "synthetic": True,
        "runs": [
            {
                "id": f"syn-{trace_id}-0",
                "name": "RunnableSequence",
                "run_type": "chain",
                "parent_run_id": None,
                "inputs": {
                    "input": [
                        {"type": "system", "content": "Extract facts."},
                        {
                            "type": "human",
                            "content": f"Extract enduring facts. CONVERSATION:\n{conv}",
                        },
                    ]
                },
            },
            {
                "id": f"syn-{trace_id}-1",
                "name": "RunnableLambda",
                "run_type": "chain",
                "parent_run_id": None,
                "outputs": {
                    "output": {
                        "facts": all_facts,
                        "core_entities": [],
                        "session_summary": "Synthetic test conversation.",
                    }
                },
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50, help="number of traces to generate")
    parser.add_argument("--out", default="/tmp/synth", help="output directory")
    parser.add_argument("--facts-per-trace", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for i in range(args.count):
        trace_id = str(uuid.uuid4())
        trace = build_trace(rng, trace_id, args.facts_per_trace)
        (out_dir / f"{trace_id}.json").write_text(json.dumps(trace, indent=2))

    print(f"Generated {args.count} synthetic traces in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
