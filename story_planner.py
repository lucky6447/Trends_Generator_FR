"""
TrendCurrent Story Planner v1
Maps locked evidence to one concrete story.
It never deletes or rewrites evidence.
"""

import json
from config import MODEL
from ollama import chat

PLANNER_CTX = 4096
PLANNER_TOKENS = 650
PLANNER_BATCH = 256
PLANNER_THREADS = 16


class StoryPlannerError(Exception):
    pass


def _fid(f, i):
    if isinstance(f, dict):
        for k in ("id", "fact_id", "factId"):
            v = str(f.get(k, "")).strip()
            if v:
                return v
    return f"F{i}"


def _ids(x):
    if not isinstance(x, list):
        raise StoryPlannerError("Planner returned invalid fact list.")
    return [str(v).strip() for v in x if str(v).strip()]


def _validate(p, facts):
    valid = {_fid(f, i) for i, f in enumerate(facts, 1)}
    groups = {
        "primary": _ids(p.get("primary_fact_ids")),
        "development": _ids(p.get("development_fact_ids")),
        "support": _ids(p.get("support_fact_ids")),
        "separate": _ids(p.get("separate_fact_ids")),
    }
    all_ids = sum(groups.values(), [])
    if len(all_ids) != len(set(all_ids)):
        raise StoryPlannerError("Fact classified more than once.")
    if set(all_ids) != valid:
        raise StoryPlannerError("Planner did not classify every locked fact exactly once.")
    if not groups["primary"]:
        raise StoryPlannerError("Planner found no primary fact.")

    order = _ids(p.get("writer_fact_order"))
    selected = groups["primary"] + groups["development"] + groups["support"]
    if set(order) != set(selected) or len(order) != len(selected):
        raise StoryPlannerError("Writer fact order does not match selected facts.")

    story = str(p.get("primary_story", "")).strip()
    if not story:
        raise StoryPlannerError("Planner returned empty primary story.")

    return {
        "primary_story": story,
        "primary_fact_ids": groups["primary"],
        "development_fact_ids": groups["development"],
        "support_fact_ids": groups["support"],
        "separate_fact_ids": groups["separate"],
        "writer_fact_order": order,
    }


def _prompt(topic, facts):
    rows = []
    for i, f in enumerate(facts, 1):
        rows.append(
            f"FACT {_fid(f, i)}\\n"
            f"{json.dumps(f, ensure_ascii=False, separators=(',', ':'))}"
        )

    return f"""
You are TrendCurrent's strict event editor.

MAIN TOPIC:
{topic}

LOCKED VERIFIED FACTS:
{chr(10).join(rows)}

TASK
----
Identify ONE concrete event represented by the MAIN TOPIC and classify EVERY fact exactly once.

The goal is NOT to collect everything related to the same subject.
The goal is to isolate the SINGLE concrete news event that the article is about.

PRIMARY
-------
Directly states the central event.

DEVELOPMENT
-----------
Use ONLY if the fact is explicitly about the SAME concrete event and describes a real
development of that event.

A DEVELOPMENT must pass BOTH tests:
1. EVENT IDENTITY: it refers to the exact same event/announcement/decision/incident.
2. EVENT CONNECTION: the fact explicitly describes what happened to, because of, or as
   a direct next step of that event.

If either test fails -> SEPARATE.

Do NOT classify as DEVELOPMENT merely because the fact:
- concerns the same person, company, film, team, product, place or topic;
- appeared in the same source;
- was reported at the same time;
- is background information;
- is another review or reaction;
- concerns another project, release, match, transaction, dispute or event;
- is a general consequence that is not explicitly tied to the event.

SUPPORT
-------
Extremely narrow class.

Use only for a factual detail that identifies or directly explains the exact same event:
date, location, participants, amount, status, or similarly necessary detail.

If the fact could stand as a separate news item, it is NOT SUPPORT.

SEPARATE
--------
Use this aggressively.

Any fact describing another concrete event goes here, even if:
- the entity is identical;
- the topic is identical;
- the source connects them;
- it is interesting;
- it happened recently;
- it would make the article longer.

IMPORTANT
---------
When uncertain between DEVELOPMENT and SEPARATE -> SEPARATE.

When uncertain between SUPPORT and SEPARATE -> SEPARATE.

Do not invent relationships.
Do not use outside knowledge.
Do not rewrite facts.
Do not delete facts.
Every fact MUST be classified exactly once.

PRIMARY STORY
-------------
Write a precise one-sentence description of the concrete event.
Never use broad wording such as "various reactions and developments".

WRITER ORDER
------------
writer_fact_order must contain ONLY:
PRIMARY + DEVELOPMENT + SUPPORT.

SEPARATE facts remain preserved for diagnostics but are NOT writer facts.

Return JSON only:
{{
  "primary_story":"",
  "primary_fact_ids":[],
  "development_fact_ids":[],
  "support_fact_ids":[],
  "separate_fact_ids":[],
  "writer_fact_order":[]
}}
"""


def plan_story(topic, evidence):
    facts = evidence.get("facts")
    if not isinstance(facts, list) or not facts:
        raise StoryPlannerError("No locked facts.")

    r = chat(
        model=MODEL,
        messages=[{"role": "user", "content": _prompt(topic, facts)}],
        options={
            "temperature": 0.0,
            "top_p": 0.8,
            "top_k": 30,
            "num_ctx": PLANNER_CTX,
            "num_predict": PLANNER_TOKENS,
            "num_batch": PLANNER_BATCH,
            "num_thread": PLANNER_THREADS,
        },
        format={
            "type": "object",
            "properties": {
                "primary_story": {"type": "string"},
                "primary_fact_ids": {"type": "array", "items": {"type": "string"}},
                "development_fact_ids": {"type": "array", "items": {"type": "string"}},
                "support_fact_ids": {"type": "array", "items": {"type": "string"}},
                "separate_fact_ids": {"type": "array", "items": {"type": "string"}},
                "writer_fact_order": {"type": "array", "items": {"type": "string"}}
            },
            "required": [
                "primary_story","primary_fact_ids","development_fact_ids",
                "support_fact_ids","separate_fact_ids","writer_fact_order"
            ]
        },
    )

    try:
        p = json.loads(r.message.content or "")
    except Exception as e:
        raise StoryPlannerError(f"Invalid planner JSON: {e}") from e

    p = _validate(p, facts)
    print(
        "[STORY PLANNER] "
        f"primary={len(p['primary_fact_ids'])} "
        f"development={len(p['development_fact_ids'])} "
        f"support={len(p['support_fact_ids'])} "
        f"separate={len(p['separate_fact_ids'])} "
        f"| {p['primary_story']}"
    )
    return p


def attach_plan(evidence, plan):
    out = dict(evidence)
    out["story_planner"] = plan
    return out
