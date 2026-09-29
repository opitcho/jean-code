---
name: research
description: Research a question on the web and get a cited report. Use when the user wants something investigated, compared or surveyed rather than answered from memory.
---

# Research

## 1. Scope the question
Don't start research on an unclear question. Ask the user, in one message, only what you can't infer
and what would change the research:
- the decision or purpose it informs
- scope: what's in and out; the time range; sources to prefer or avoid
- depth: quick (≈$0.25), standard (≈$0.50) or deep (≈$2)
- the deliverable: what the report must answer

If you don't know the field well enough to ask good questions, look first: one searcher, or a few
searches of your own, to find the main options and terms. A question that already answers all of
these goes straight to step 2.

If nobody can answer you (you are a subagent), don't ask: pick the most likely reading, write the
brief with your assumptions under Purpose, and start.

## 2. Write the brief
Write the brief with these headings: Question, Purpose, In scope, Out of scope, Time range, Depth,
Deliverable. End your reply with it and ask for a yes. Do this even when the request was clear: the
brief is where the user catches a misreading before it costs a run. Spawn nothing in that reply;
start step 3 only once the user agrees, with their edits folded in.

## 3. Run it
`spawn("research", brief, background=True, max_cost=…)`, with max_cost 0.25 for quick, 0.50 for
standard and 2 for deep. The task is the brief, exactly as the user agreed to it. Tell the user it's
running, and carry on with anything else they ask.

The research run has its own lead and researchers, and writes the report to a file. Its result
arrives in the conversation as a `<subagent-result>` block with the next user message. Only call
`subagent(id)` if the user asks how it's going.

If `research` isn't listed under "## Subagents" (you are a subagent yourself), research it yourself
with web_search and fetch_page, and reply with the report, citing the URL of each fact.

## 4. Relay the result
When the result arrives, give the user the Summary and the report's path. A citation marked ⚠, like
[3⚠], rests on a quote that couldn't be checked against its page: say which claims those are. If the
status is "stopped" or "failed", say so: the report holds whatever was written before it stopped.
Read the report file only when the user wants more than the Summary.
